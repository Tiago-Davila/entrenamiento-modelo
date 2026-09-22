"""
02_train.py — Entrenamiento del modelo LSA con Intel Arc 140V (IPEX/XPU)
Tambien compatible con CUDA (Kaggle/Colab) y CPU.

Fases:
  A - Pre-entrenamiento en LSA-X (--phase a)  [opcional]
  B - Entrenamiento principal en LSA-T (--phase b) [principal]
  C - Fine-tuning con augmentacion (--phase c)

Uso:
    python 02_train.py --phase b              # fase principal (recomendado)
    python 02_train.py --phase a --epochs 25  # pre-entrenamiento LSA-X
    python 02_train.py --phase c --epochs 20  # fine-tuning
    python 02_train.py --phase b --smoke      # smoke test (5 pasos, verificar setup)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CHECKPOINTS_DIR, DATA_DIR, VOCAB_PATH,
    D_MODEL, NHEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS, DIM_FFN, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, LR_WARMUP_RATIO,
    WEIGHT_DECAY, GRAD_CLIP, PATIENCE, LABEL_SMOOTHING,
    PAD_ID, BOS_ID, EOS_ID,
)
from model import LSAModel, get_device
from dataset import Vocabulary, LSADataset, make_loaders, collate_fn


# ── IPEX / Device setup ───────────────────────────────────────────────────────

def setup_device():
    """
    Detecta el device y configura IPEX si esta disponible.
    Retorna (device_str, use_ipex, amp_dtype).
    """
    device = get_device()
    use_ipex = False
    amp_dtype = torch.float16  # default para CUDA

    if device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            use_ipex = True
            amp_dtype = torch.bfloat16   # XPU prefiere BF16 (sin GradScaler)
            print("[setup] Intel Extension for PyTorch cargado correctamente")
            print(f"[setup] AMP dtype: bfloat16 (recomendado para Arc 140V)")
        except ImportError:
            print("[warn] IPEX no instalado — usando XPU sin optimizaciones IPEX")
            print("       Instalar: pip install intel-extension-for-pytorch")

    elif device == "cuda":
        amp_dtype = torch.float16
        print(f"[setup] CUDA disponible — AMP dtype: float16")

    else:
        print("[warn] Entrenando en CPU — sera muy lento.")
        print("       Verificar instalacion de IPEX: pip install intel-extension-for-pytorch")

    return device, use_ipex, amp_dtype


# ── Scheduler con warmup ──────────────────────────────────────────────────────

def make_scheduler(optimizer, total_steps: int, warmup_ratio: float = LR_WARMUP_RATIO):
    """Cosine annealing con warmup lineal."""
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Evaluacion BLEU ───────────────────────────────────────────────────────────

def evaluate_bleu(model: LSAModel, loader: DataLoader,
                  vocab: Vocabulary, device: str, max_batches: int = 50) -> float:
    """Calcula BLEU-4 sobre un subset del loader (rapido)."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        print("    [warn] sacrebleu no instalado — usando WER como proxy")
        return evaluate_wer_proxy(model, loader, vocab, device, max_batches)

    model.eval()
    bleu = BLEU(effective_order=True)
    hypotheses, references = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            kps    = batch["keypoints"].to(device)
            labels = batch["labels"]

            # Encoder
            embeddings, _ = model.encoder(kps, batch["pad_mask"].to(device))

            # Decodificacion greedy
            pred_ids = model.decoder.greedy_decode(
                embeddings, BOS_ID, EOS_ID, max_len=64
            )

            for ids, ref in zip(pred_ids, labels):
                hyp = vocab.decode(ids)
                hypotheses.append(hyp)
                references.append(ref)

    if not hypotheses:
        return 0.0

    score = bleu.corpus_score(hypotheses, [references])
    return score.score


def evaluate_wer_proxy(model, loader, vocab, device, max_batches) -> float:
    """WER simplificado como proxy de BLEU cuando sacrebleu no esta."""
    model.eval()
    correct_words, total_words = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            kps = batch["keypoints"].to(device)
            embeddings, _ = model.encoder(kps, batch["pad_mask"].to(device))
            pred_ids = model.decoder.greedy_decode(embeddings, BOS_ID, EOS_ID)
            for ids, ref in zip(pred_ids, batch["labels"]):
                hyp_words = vocab.decode(ids).split()
                ref_words = ref.split()
                correct_words += sum(h == r for h, r in zip(hyp_words, ref_words))
                total_words   += max(len(ref_words), 1)
    return (correct_words / max(total_words, 1)) * 100.0


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device, use_ipex, amp_dtype = setup_device()

    # Vocabulario y datasets
    vocab      = Vocabulary()
    vocab_size = len(vocab)
    print(f"[data] Vocabulario: {vocab_size} tokens")

    # Segun la fase, elegir el dataset de entrenamiento
    if args.phase == "a":
        print("[data] Fase A: pre-entrenamiento en LSA-X")
        pretrain_dir = DATA_DIR / "processed" / "pretrain"
        if not pretrain_dir.exists() or not list(pretrain_dir.glob("*.npz")):
            print("ERROR: Datos LSA-X no encontrados.")
            print("       Correr: python 01_prepare_data.py --lsax")
            sys.exit(1)
        train_ds = LSADataset("pretrain", vocab, augment=True)
        ckpt_prefix = "phase_a"
        n_epochs    = args.epochs or 25
        lr          = 1e-4
    elif args.phase == "b":
        print("[data] Fase B: entrenamiento principal en LSA-T")
        ckpt_prefix = "phase_b"
        n_epochs    = args.epochs or EPOCHS
        lr          = LR
        train_ds    = LSADataset("train", vocab, augment=False)
    elif args.phase == "c":
        print("[data] Fase C: fine-tuning con augmentacion")
        ckpt_prefix = "phase_c"
        n_epochs    = args.epochs or 20
        lr          = 1e-5
        train_ds    = LSADataset("train", vocab, augment=True)
    else:
        print(f"ERROR: Fase '{args.phase}' invalida. Usar a, b, o c.")
        sys.exit(1)

    val_ds  = LSADataset("val",  vocab, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"[data] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Modelo
    model = LSAModel(vocab_size=vocab_size)
    print(f"[model] {model.count_params()}")

    # Cargar checkpoint previo si existe (continuar entrenamiento)
    last_ckpt = CHECKPOINTS_DIR / f"{ckpt_prefix}_last.pt"
    if not args.fresh and last_ckpt.exists():
        print(f"[model] Cargando checkpoint: {last_ckpt}")
        model.load_state_dict(torch.load(last_ckpt, map_location="cpu"))

    elif args.phase in ("b", "c"):
        # Intentar cargar checkpoint de fase anterior
        prev_map = {"b": "phase_a_best.pt", "c": "phase_b_best.pt"}
        prev_ckpt = CHECKPOINTS_DIR / prev_map[args.phase]
        if prev_ckpt.exists():
            print(f"[model] Cargando pesos de fase anterior: {prev_ckpt}")
            model.load_state_dict(torch.load(prev_ckpt, map_location="cpu"))

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98),
    )

    # Aplicar optimizaciones IPEX (Arc 140V)
    if use_ipex:
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(
            model, optimizer=optimizer, dtype=amp_dtype, level="O1"
        )
        print(f"[ipex] Modelo optimizado con IPEX (dtype={amp_dtype})")

    # Scheduler
    total_steps = n_epochs * len(train_loader)
    scheduler   = make_scheduler(optimizer, total_steps)

    # Loss con label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING
    )

    # GradScaler solo para CUDA FP16
    use_scaler = (device == "cuda" and amp_dtype == torch.float16)
    scaler     = torch.cuda.amp.GradScaler() if use_scaler else None

    # Contexto AMP
    def amp_context():
        if device == "xpu":
            return torch.xpu.amp.autocast(dtype=amp_dtype, enabled=True)
        elif device == "cuda":
            return torch.cuda.amp.autocast(dtype=amp_dtype)
        else:
            import contextlib
            return contextlib.nullcontext()

    print(f"\n[train] Iniciando Fase {args.phase.upper()} — {n_epochs} epocas, "
          f"lr={lr:.1e}, device={device}")
    if args.smoke:
        print("[train] SMOKE TEST: solo 5 pasos por epoca")

    best_bleu    = 0.0
    no_improve   = 0
    history      = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for step, batch in enumerate(train_loader):
            if args.smoke and step >= 5:
                break

            kps      = batch["keypoints"].to(device)   # [B, T, 126]
            tgt_in   = batch["tgt_in"].to(device)       # [B, S]
            tgt_out  = batch["tgt_out"].to(device)      # [B, S]
            src_pad  = batch["pad_mask"].to(device)     # [B, T]
            tgt_pad  = batch["tgt_mask"].to(device)     # [B, S]

            optimizer.zero_grad()

            with amp_context():
                dec_logits, _ = model(kps, tgt_in, src_pad, tgt_pad)
                # dec_logits: [B, S, vocab_size]
                # tgt_out:    [B, S]
                B, S, V = dec_logits.shape
                loss = criterion(dec_logits.reshape(B * S, V), tgt_out.reshape(B * S))

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(step + 1, 1)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        # Evaluar BLEU en validacion
        bleu = evaluate_bleu(model, val_loader, vocab, device, max_batches=30)

        history.append({"epoch": epoch, "loss": avg_loss, "bleu": bleu, "lr": lr_now})
        print(f"Epoca {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  "
              f"BLEU-4={bleu:.2f}  lr={lr_now:.2e}  t={elapsed:.0f}s")

        # Guardar checkpoint ultimo
        state = _get_model_state(model, use_ipex)
        torch.save(state, CHECKPOINTS_DIR / f"{ckpt_prefix}_last.pt")

        # Early stopping y mejor modelo
        if bleu > best_bleu:
            best_bleu  = bleu
            no_improve = 0
            torch.save(state, CHECKPOINTS_DIR / f"{ckpt_prefix}_best.pt")
            print(f"  -> Nuevo mejor BLEU-4: {best_bleu:.2f}  (guardado)")
        else:
            no_improve += 1
            if no_improve >= PATIENCE and not args.smoke:
                print(f"\nEarly stopping: {PATIENCE} epocas sin mejora en BLEU-4")
                break

    # Guardar historial
    hist_path = CHECKPOINTS_DIR / f"{ckpt_prefix}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\n[done] Fase {args.phase.upper()} finalizada.")
    print(f"       Mejor BLEU-4: {best_bleu:.2f}")
    print(f"       Checkpoint:   {CHECKPOINTS_DIR / f'{ckpt_prefix}_best.pt'}")
    print(f"       Siguiente:    python 02_train.py --phase {_next_phase(args.phase)}")


def _get_model_state(model, use_ipex: bool) -> dict:
    """Extrae state_dict limpio, compatible con CPU."""
    if use_ipex:
        # IPEX puede envolver el modelo; intentar extraer el original
        try:
            return model._model.state_dict()
        except AttributeError:
            pass
    return model.state_dict()


def _next_phase(phase: str) -> str:
    return {"a": "b", "b": "c", "c": "b --fresh (re-entrenar)"}[phase]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento LSA")
    parser.add_argument("--phase",  choices=["a", "b", "c"], default="b",
                        help="Fase de entrenamiento (default: b)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Numero de epocas (overrides config)")
    parser.add_argument("--smoke",  action="store_true",
                        help="Smoke test: 5 pasos por epoca para verificar el setup")
    parser.add_argument("--fresh",  action="store_true",
                        help="Ignorar checkpoints existentes y empezar desde cero")
    args = parser.parse_args()

    train(args)
