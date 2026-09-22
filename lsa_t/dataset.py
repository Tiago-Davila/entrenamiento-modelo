"""
dataset.py — Dataset classes para LSA-T y LSA-X
Lee los keypoints preprocesados generados por 01_prepare_data.py.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import (
    DATA_DIR, VOCAB_PATH,
    MAX_FRAMES, INPUT_DIM,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
    MAX_DEC_LEN, BATCH_SIZE,
)
from sequence_contract import normalize_keypoints, pad_or_crop


# ── Vocabulario ───────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, vocab_path: Path = VOCAB_PATH):
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulario no encontrado en {vocab_path}.\n"
                "Correr primero: python 01_prepare_data.py"
            )
        self.tokens: list[str] = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.token2id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Texto en espanol -> lista de IDs (sin BOS/EOS)."""
        return [self.token2id.get(w, UNK_ID) for w in text.lower().split()]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Lista de IDs -> texto."""
        special = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
        out = []
        for i in ids:
            if skip_special and i in special:
                continue
            if 0 <= i < len(self.tokens):
                out.append(self.tokens[i])
        return " ".join(out)


# ── Augmentacion ─────────────────────────────────────────────────────────────

def augment_keypoints(kps: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """
    Augmentacion sobre keypoints normalizados [T, 42, 3].
    - Flip horizontal (invertir manos)
    - Ruido gaussiano en coordenadas
    - Time-warp (stretching temporal leve)
    """
    if np.random.rand() > prob:
        return kps

    kps = kps.copy()

    # Flip horizontal (intercambiar mano izq <-> der y negar x)
    if np.random.rand() < 0.5:
        left, right = kps[:, :21, :], kps[:, 21:, :]
        kps[:, :21, :], kps[:, 21:, :] = right.copy(), left.copy()
        kps[:, :, 0] *= -1   # negar coordenada x

    # Ruido gaussiano
    if np.random.rand() < 0.5:
        noise_scale = np.random.uniform(0.01, 0.03)
        kps[:, :, :2] += np.random.randn(*kps[:, :, :2].shape).astype(np.float32) * noise_scale

    # Time-warp: reescalar tiempo +-10%
    if np.random.rand() < 0.5:
        T     = kps.shape[0]
        warp  = np.random.uniform(0.9, 1.1)
        new_T = max(5, int(T * warp))
        idx   = np.linspace(0, T - 1, new_T)
        # Interpolar
        kps_new = np.zeros((new_T, kps.shape[1], kps.shape[2]), dtype=kps.dtype)
        for j in range(kps.shape[1]):
            for c in range(kps.shape[2]):
                kps_new[:, j, c] = np.interp(idx, np.arange(T), kps[:, j, c])
        kps = kps_new

    return kps


# ── Dataset ───────────────────────────────────────────────────────────────────

class LSADataset(Dataset):
    """
    Lee los keypoints cacheados en NPZ (generados por 01_prepare_data.py).
    Cada archivo NPZ contiene: 'keypoints' [T, 42, 3] y 'label' (string).
    """

    def __init__(
        self,
        split: str,               # "train", "val", "test"
        vocab: Vocabulary,
        augment: bool = False,
        max_frames: int = MAX_FRAMES,
        max_token_len: int = MAX_DEC_LEN,
    ):
        self.vocab         = vocab
        self.augment       = augment
        self.max_frames    = max_frames
        self.max_token_len = max_token_len

        split_dir = DATA_DIR / "processed" / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{split}' no encontrado en {split_dir}.\n"
                "Correr primero: python 01_prepare_data.py"
            )

        self.files = sorted(split_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise ValueError(f"No se encontraron archivos .npz en {split_dir}")

        print(f"[dataset] {split}: {len(self.files)} muestras")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx], allow_pickle=True)
        kps  = data["keypoints"].astype(np.float32)  # [T, 42, 3]
        label = str(data["label"])

        # Augmentacion (solo en train)
        if self.augment:
            kps = augment_keypoints(kps)

        # Padding / crop a max_frames y aplanar a [T, 126]
        kps  = pad_or_crop(kps, self.max_frames)           # [T, 42, 3]
        kps  = kps.reshape(self.max_frames, INPUT_DIM)     # [T, 126]

        # Mascara de padding (frames con todos ceros)
        pad_mask = (kps.sum(axis=-1) == 0)  # [T] True = frame es padding

        # Tokenizar label
        token_ids = self.vocab.encode(label)[: self.max_token_len - 2]
        # Agregar BOS y EOS
        tgt_in  = [BOS_ID] + token_ids            # entrada al decoder
        tgt_out = token_ids + [EOS_ID]             # objetivo (shifted)

        return {
            "keypoints":  torch.tensor(kps,      dtype=torch.float32),
            "pad_mask":   torch.tensor(pad_mask, dtype=torch.bool),
            "tgt_in":     torch.tensor(tgt_in,   dtype=torch.long),
            "tgt_out":    torch.tensor(tgt_out,  dtype=torch.long),
            "label":      label,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Padding dinamico de las secuencias de tokens del decoder.
    Los keypoints ya son de longitud fija (max_frames).
    """
    keypoints = torch.stack([b["keypoints"] for b in batch])   # [B, T, 126]
    pad_mask  = torch.stack([b["pad_mask"]  for b in batch])   # [B, T]
    labels    = [b["label"] for b in batch]

    max_len  = max(len(b["tgt_in"])  for b in batch)
    tgt_in   = torch.zeros(len(batch), max_len, dtype=torch.long)
    tgt_out  = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    tgt_mask = torch.ones(len(batch), max_len, dtype=torch.bool)  # True = padding

    for i, b in enumerate(batch):
        L = len(b["tgt_in"])
        tgt_in[i,  :L] = b["tgt_in"]
        tgt_out[i, :L] = b["tgt_out"]
        tgt_mask[i, :L] = False   # no es padding

    return {
        "keypoints": keypoints,
        "pad_mask":  pad_mask,
        "tgt_in":    tgt_in,
        "tgt_out":   tgt_out,
        "tgt_mask":  tgt_mask,
        "labels":    labels,
    }


def make_loaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader, DataLoader]:
    vocab = Vocabulary()
    train_ds = LSADataset("train", vocab, augment=True)
    val_ds   = LSADataset("val",   vocab, augment=False)
    test_ds  = LSADataset("test",  vocab, augment=False)

    kwargs = dict(
        batch_size  = batch_size,
        collate_fn  = collate_fn,
        num_workers = 0,    # 0 es mas estable con XPU en Linux
        pin_memory  = False,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )
