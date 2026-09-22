"""
01_prepare_data.py — Descarga y preprocesa LSA-T y LSA-X
Crea los archivos .npz en data/processed/{train,val,test}/

Uso:
    python 01_prepare_data.py             # solo LSA-T
    python 01_prepare_data.py --lsax      # LSA-T + LSA-X (recomendado)
"""
import argparse
import json
import os
import sys
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR, VOCAB_PATH,
    MP_LEFT_HAND_START, MP_RIGHT_HAND_START, MP_HAND_LANDMARKS,
    SPECIAL_TOKENS, MIN_FREQ,
    LSAX_ZENODO_BASE, LSAX_SUBSETS,
)
from dataset import normalize_keypoints, pad_or_crop


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_hand_keypoints_from_pose(pose_data) -> np.ndarray | None:
    """
    Extrae los 42 landmarks de manos de un objeto Pose (pose-format).
    Retorna [T, 42, 3] o None si no hay datos de manos.
    """
    try:
        # pose.body.data: [T, 1, total_landmarks, coords]
        # Puede ser un tensor de torch (.numpy()) o ya un np.ndarray/MaskedArray
        raw = pose_data.body.data
        data = raw.numpy() if hasattr(raw, "numpy") else np.ma.filled(raw, 0.0)
        T    = data.shape[0]

        left_start  = MP_LEFT_HAND_START
        right_start = MP_RIGHT_HAND_START
        N           = MP_HAND_LANDMARKS

        left_hand  = data[:, 0, left_start  : left_start  + N, :3]  # [T, 21, 3]
        right_hand = data[:, 0, right_start : right_start + N, :3]  # [T, 21, 3]

        kps = np.concatenate([left_hand, right_hand], axis=1)  # [T, 42, 3]
        return kps.astype(np.float32)
    except Exception as e:
        print(f"    [warn] Error extrayendo keypoints de pose: {e}")
        return None


def save_sample(out_dir: Path, name: str, kps: np.ndarray, label: str):
    """Guarda una muestra como archivo .npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{name}.npz",
        keypoints=kps,
        label=label,
    )


# ── LSA-T (HuggingFace pedroodb/glosl-lsat) ──────────────────────────────────

def prepare_lsat():
    """
    Descarga y procesa LSA-T desde HuggingFace.
    Requiere: pandas, huggingface_hub, pose-format

    NOTA: el repo pedroodb/glosl-lsat ya no trae un loading script (solo
    `annotations.csv` + `poses/*.pose` + `videos/`), por lo que
    `datasets.load_dataset(..., trust_remote_code=True)` cae al builder
    generico "videofolder" (un unico split "train" con solo la columna
    "video"), sin texto ni poses. Por eso se descargan los archivos crudos
    directamente en vez de usar `load_dataset`.
    """
    print("\n=== Preparando LSA-T (pedroodb/glosl-lsat) ===")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: Instalar dependencias primero:")
        print("  pip install pandas huggingface_hub pose-format")
        sys.exit(1)

    REPO_ID = "pedroodb/glosl-lsat"
    SPLIT_MAP = {"train": "train", "dev": "val", "test": "test"}  # split HF -> carpeta local

    print("[1/3] Descargando annotations.csv...")
    ann_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="annotations.csv")
    df = pd.read_csv(ann_path)
    df = df[df["split"].isin(SPLIT_MAP)]
    print(f"    Filas: {len(df)}  Splits: {df['split'].value_counts().to_dict()}")

    # Construir vocabulario desde el train set
    print("[2/3] Construyendo vocabulario...")
    word_counts: Counter = Counter()
    for text in df.loc[df["split"] == "train", "text"]:
        word_counts.update(str(text).lower().split())

    vocab = SPECIAL_TOKENS + [
        w for w, c in word_counts.most_common() if c >= MIN_FREQ
    ]
    VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"    Vocabulario: {len(vocab)} tokens "
          f"(filtrado singletons: {sum(1 for c in word_counts.values() if c < MIN_FREQ)} palabras eliminadas)")

    # Descargar poses y guardar .npz por split
    print("[3/3] Descargando poses y guardando .npz...")
    total = 0
    for split_hf, split_name in SPLIT_MAP.items():
        split_df = df[df["split"] == split_hf]
        out_dir = DATA_DIR / "processed" / split_name
        ok = 0
        for i, row in enumerate(tqdm(split_df.itertuples(), total=len(split_df), desc=f"  {split_name}")):
            label = str(row.text).lower().strip()
            try:
                pose_path = hf_hub_download(
                    repo_id=REPO_ID, repo_type="dataset", filename=f"poses/{row.id}.pose",
                )
            except Exception as e:
                print(f"    [warn] No se pudo descargar pose de '{row.id}': {e}")
                continue

            kps = _read_pose(Path(pose_path).read_bytes())
            if kps is None or kps.shape[0] < 5:
                continue  # skip frames insuficientes

            kps_norm = normalize_keypoints(kps)
            save_sample(out_dir, f"{split_name}_{i:05d}", kps_norm, label)
            ok += 1

        print(f"    {split_name}: {ok} muestras guardadas")
        total += ok

    print(f"\nLSA-T listo: {total} muestras totales en {DATA_DIR / 'processed'}")


def _read_pose(pose_raw) -> np.ndarray | None:
    """
    Intenta leer keypoints desde distintos formatos que puede tener el dataset:
    - bytes / bytearray: formato .pose binario (pose-format)
    - np.ndarray / list: array de keypoints ya extraidos
    - dict con 'data'
    """
    from pose_format import Pose

    if isinstance(pose_raw, (bytes, bytearray)):
        try:
            pose = Pose.read(bytes(pose_raw))
            return extract_hand_keypoints_from_pose(pose)
        except Exception:
            return None

    if isinstance(pose_raw, (np.ndarray, list)):
        arr = np.array(pose_raw, dtype=np.float32)
        # Puede ser [T, 543, 3] o [T, 42, 3] o [T, 126]
        if arr.ndim == 2:
            # [T, 126] -> reshape a [T, 42, 3]
            if arr.shape[1] == 126:
                return arr.reshape(arr.shape[0], 42, 3)
            # [T, N*3] -> reshape
            if arr.shape[1] % 3 == 0:
                N = arr.shape[1] // 3
                arr = arr.reshape(arr.shape[0], N, 3)
        if arr.ndim == 3:
            T, N, C = arr.shape
            if N == 543:
                left  = arr[:, MP_LEFT_HAND_START  : MP_LEFT_HAND_START  + 21, :3]
                right = arr[:, MP_RIGHT_HAND_START : MP_RIGHT_HAND_START + 21, :3]
                return np.concatenate([left, right], axis=1).astype(np.float32)
            if N == 42 and C >= 3:
                return arr[:, :, :3].astype(np.float32)
        return None

    if isinstance(pose_raw, dict) and "data" in pose_raw:
        return _read_pose(pose_raw["data"])

    if isinstance(pose_raw, str) and Path(pose_raw).exists():
        try:
            pose = Pose.read(Path(pose_raw).read_bytes())
            return extract_hand_keypoints_from_pose(pose)
        except Exception:
            return None

    return None


# ── LSA-X (Zenodo) ────────────────────────────────────────────────────────────

def prepare_lsax():
    """
    Descarga y procesa los subsets de LSA-X desde Zenodo.
    Solo para pre-entrenamiento: guarda en data/processed/pretrain/
    """
    print("\n=== Preparando LSA-X (Zenodo 19087120) ===")
    try:
        import h5py
        import urllib.request
    except ImportError:
        print("ERROR: Instalar h5py: pip install h5py")
        sys.exit(1)

    COCO_TO_MP_BODY = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16}

    def coco_to_mediapipe_hands(coco_kps: np.ndarray) -> np.ndarray | None:
        """
        LSA-X almacena keypoints en formato COCO upper body.
        Retorna [T, 42, 3] con ceros en las manos (COCO no tiene manos en detalle).
        Util para pre-entrenamiento del encoder sobre el cuerpo.
        """
        if coco_kps is None or len(coco_kps) == 0:
            return None
        # coco_kps: [T, N_joints, 3] o [T, N_joints*3]
        kps = np.array(coco_kps, dtype=np.float32)
        if kps.ndim == 2:
            kps = kps.reshape(kps.shape[0], -1, 3)
        T = kps.shape[0]
        # Crear placeholder de 42 joints (manos) con ceros
        # Usar muñecas (COCO 9=l_wrist, 10=r_wrist) como ancla
        hands = np.zeros((T, 42, 3), dtype=np.float32)
        N_src = kps.shape[1]
        if N_src > 9:
            hands[:, 0]  = kps[:, 9,  :3]  # ancla muñeca izq en primer landmark
            hands[:, 21] = kps[:, 10, :3]  # ancla muñeca der
        return hands

    out_dir = DATA_DIR / "processed" / "pretrain"
    total   = 0

    for subset in LSAX_SUBSETS:
        h5_path = DATA_DIR / subset
        if not h5_path.exists():
            url = f"{LSAX_ZENODO_BASE}/{subset}/content"
            print(f"Descargando {subset} (~2-5 GB)...")
            try:
                urllib.request.urlretrieve(url, h5_path, reporthook=_dl_progress)
                print()
            except Exception as e:
                print(f"  [error] No se pudo descargar {subset}: {e}")
                print(f"  Descargar manualmente desde: https://zenodo.org/records/19087120")
                continue

        print(f"Procesando {subset}...")
        try:
            with h5py.File(h5_path, "r") as f:
                # La estructura del h5 varia por subset; intentar detectarla
                items = _iter_lsax_h5(f)
                for i, (kps, label) in enumerate(tqdm(items, desc=f"  {subset}")):
                    hand_kps = coco_to_mediapipe_hands(kps)
                    if hand_kps is None or len(hand_kps) < 5:
                        continue
                    if not label or len(label.strip()) == 0:
                        continue
                    kps_norm = normalize_keypoints(hand_kps)
                    safe_name = Path(subset).stem + f"_{i:05d}"
                    save_sample(out_dir, safe_name, kps_norm, label.lower().strip())
                    total += 1
        except Exception as e:
            print(f"  [error] Procesando {subset}: {e}")

    print(f"\nLSA-X listo: {total} muestras en {out_dir}")


def _iter_lsax_h5(f):
    """Itera sobre (keypoints, label) en un archivo h5 de LSA-X."""
    # Estructuras comunes en LSA-X:
    # - f["keypoints"][i], f["labels"][i]
    # - f["data/keypoints"], f["data/text"]
    # - grupos por video ID
    for key in f.keys():
        g = f[key]
        if isinstance(g, type(f)):  # es un grupo
            # Intentar leer keypoints y label del grupo
            kps_key   = next((k for k in g.keys() if "kp" in k.lower() or "pose" in k.lower() or "joint" in k.lower()), None)
            label_key = next((k for k in g.keys() if "label" in k.lower() or "text" in k.lower() or "sub" in k.lower()), None)
            if kps_key and label_key:
                try:
                    kps   = np.array(g[kps_key])
                    label = str(g[label_key][()].decode("utf-8") if isinstance(g[label_key][()], bytes) else g[label_key][()])
                    yield kps, label
                except Exception:
                    pass
        elif hasattr(g, "shape"):
            # Dataset plano: intentar con claves hermanas
            pass


def _dl_progress(count, block_size, total_size):
    pct = count * block_size / total_size * 100
    print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparacion de datos LSA")
    parser.add_argument("--lsax",  action="store_true",
                        help="Incluir LSA-X para pre-entrenamiento (recomendado)")
    parser.add_argument("--only-lsax", action="store_true",
                        help="Solo procesar LSA-X (si LSA-T ya esta listo)")
    args = parser.parse_args()

    if not args.only_lsax:
        prepare_lsat()

    if args.lsax or args.only_lsax:
        prepare_lsax()

    print("\n=== Resumen ===")
    for split in ["train", "val", "test", "pretrain"]:
        d = DATA_DIR / "processed" / split
        if d.exists():
            n = len(list(d.glob("*.npz")))
            print(f"  {split:10s}: {n:6d} muestras")

    print(f"\nVocabulario: {len(json.loads(VOCAB_PATH.read_text())) if VOCAB_PATH.exists() else 'NO GENERADO'} tokens")
    print("\nListo. Siguiente paso: python 02_train.py")
