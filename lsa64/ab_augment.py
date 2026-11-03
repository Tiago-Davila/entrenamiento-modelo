#!/usr/bin/env python3
"""
ab_augment.py — Mide si una configuración de augmentación mejora la
generalización, con varias semillas y split por sujeto (contrato v3).

Una sola corrida no alcanza: la varianza entre semillas puede ser mayor que
el efecto que se busca medir. Usa el mismo Augmenter y el mismo
entrenamiento que train_keras_tflite.py (augmentación nueva por época).

Variantes:
    sin_aug   sin augmentación
    aug       AugmentConfig por defecto
    aug_yaw   + giro sobre el eje vertical (p_yaw=0,5), que depende de la z
              de MediaPipe y por eso se evalúa aparte antes de activarlo

Uso:
    cd lsa64
    python ab_augment.py --data ../data/processed_v3 --seeds 3 --arch conv_tx --epochs 80
    python ab_augment.py --data ../data/processed_v3 --variantes aug aug_yaw
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import keras

import train_keras_tflite as tk
from eva_contract import load_catalog

VARIANTES = {
    "sin_aug": {"no_augment": True, "p_yaw": 0.0},
    "aug": {"no_augment": False, "p_yaw": 0.0},
    "aug_yaw": {"no_augment": False, "p_yaw": 0.5},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variantes", nargs="+", choices=sorted(VARIANTES), default=["sin_aug", "aug"])
    ap.add_argument("--arch", choices=("lstm", "conv_tx"), default="conv_tx")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--val-subjects", nargs="*", type=int, default=[9])
    ap.add_argument("--test-subjects", nargs="*", type=int, default=[10])
    ap.add_argument("--out", default="ab_augment.json")
    args = ap.parse_args()

    # Hiperparámetros por defecto del script de entrenamiento.
    base = Namespace(
        arch=args.arch, lr=1e-3, batch=64, hidden=128, dim=192, dropout=0.3,
        weight_decay=0.05, label_smoothing=0.1, paciencia=40, workers=4,
        verbose=False, epochs=args.epochs)

    _, catalog = load_catalog()
    X, y, sujetos, clases = tk.cargar(Path(args.data), catalog)
    n = len(clases)
    m_te = np.isin(sujetos, args.test_subjects)
    m_va = np.isin(sujetos, args.val_subjects)
    m_tr = ~(m_te | m_va)

    resultados = {}
    for nombre in args.variantes:
        cfg = Namespace(**vars(base), **VARIANTES[nombre])
        accs = []
        for seed in range(args.seeds):
            modelo, _ = tk.entrenar(cfg, X[m_tr], y[m_tr], X[m_va], y[m_va], n,
                                    args.epochs, seed, early_stopping=True)
            accs.append(tk.evaluar(modelo, X[m_te], y[m_te], n, catalog)["accuracy"])
            keras.backend.clear_session()
            print(f"  {nombre} semilla {seed}: test {accs[-1]:.4f}")
        resultados[nombre] = {"test": accs, "media": round(float(np.mean(accs)), 4),
                              "desvio": round(float(np.std(accs)), 4)}
        print(f"{nombre}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    Path(args.out).write_text(json.dumps({
        "arch": args.arch, "epochs": args.epochs, "seeds": args.seeds,
        "sujetosVal": args.val_subjects, "sujetosTest": args.test_subjects,
        "resultados": resultados,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
