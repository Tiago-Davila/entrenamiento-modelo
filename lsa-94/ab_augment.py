#!/usr/bin/env python3
"""
ab_augment.py — Compara variantes de entrenamiento de Eva LSA-94 con varias
semillas, por grupo de señas (lsa64, temporales, abecedario).

Una sola corrida no alcanza: con 4 repeticiones por letra, la varianza entre
semillas puede ser mayor que el efecto que se busca medir. Solo se adopta una
mejora si la diferencia supera el desvío entre semillas.

Cada variante se entrena, por semilla, en:
  - los folds de repeticiones del sujeto 012 (--folds-reps): deja afuera la
    repetición r en las 94 clases. Es lo único que mide las clases nuevas.
  - opcionalmente folds LOSO de LSA64 (--folds-loso): verifica que las
    mejoras no empeoren las 64 señas entre personas.
Sin early stopping y con épocas fijas, como los CV de train_keras_tflite.py.

Variantes (acumulativas, de v3 a LSA-94 completo):
    v3            features v3, sin balanceo, augmentación v3
    balance       + muestreo balanceado (--balance 0.5)
    perfil        + perfil fuerte para clases de un solo señante
    aug_mano      + dedos, giro de mano y deriva lenta
    forma         + FeaturesEva94 (forma de mano)  = configuración por defecto
    forma_ema     + promedio exponencial de pesos
    forma_bal0    forma con balanceo total (--balance 0)

Uso:
    cd lsa-94
    ..\\venv\\Scripts\\python.exe ab_augment.py --data .\\data\\processed_94 --seeds 2 --epochs 80
    ..\\venv\\Scripts\\python.exe ab_augment.py --data .\\data\\processed_94 \\
        --variantes v3 forma --folds-loso 10 --out ab_v3_vs_forma.json
"""

from __future__ import annotations

import entorno  # noqa: F401  (KERAS_BACKEND antes de importar keras)

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import keras

import train_keras_tflite as tk
from eva_contract import load_catalog, load_groups

_V3 = {"features": "v3", "balance": 1.0, "sin_perfil_fuerte": True,
       "sin_aug_mano": True, "ema": False}
VARIANTES = {
    "v3": _V3,
    "balance": {**_V3, "balance": 0.5},
    "perfil": {**_V3, "balance": 0.5, "sin_perfil_fuerte": False},
    "aug_mano": {**_V3, "balance": 0.5, "sin_perfil_fuerte": False, "sin_aug_mano": False},
    "forma": {"features": "forma", "balance": 0.5, "sin_perfil_fuerte": False,
              "sin_aug_mano": False, "ema": False},
    "forma_ema": {"features": "forma", "balance": 0.5, "sin_perfil_fuerte": False,
                  "sin_aug_mano": False, "ema": True},
    "forma_bal0": {"features": "forma", "balance": 0.0, "sin_perfil_fuerte": False,
                   "sin_aug_mano": False, "ema": False},
}


def resumir(corridas: dict, variantes, seeds: int, folds, grupos) -> dict:
    """Por variante, tipo de fold y grupo: media ± desvío entre semillas de la
    accuracy sobre todos los folds de ese tipo."""
    resumen = {}
    for nombre in variantes:
        for tipo in sorted({t for t, _, _ in folds}):
            por_seed: dict[str, list[float]] = {}
            for seed in range(seeds):
                rs = [corridas[f"{nombre}|{seed}|{tipo}{f}"] for t, f, _ in folds if t == tipo]
                for g in grupos:
                    gs = [r["porGrupo"][g] for r in rs if g in r["porGrupo"]]
                    if gs:
                        nn = sum(v["n"] for v in gs)
                        por_seed.setdefault(g, []).append(
                            sum(v["accuracy"] * v["n"] for v in gs) / nn)
            resumen.setdefault(nombre, {})[tipo] = {
                g: {"media": round(float(np.mean(v)), 4), "desvio": round(float(np.std(v)), 4)}
                for g, v in por_seed.items()}
    return resumen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--variantes", nargs="+", choices=list(VARIANTES),
                    default=["v3", "balance", "perfil", "aug_mano", "forma"])
    ap.add_argument("--arch", choices=("lstm", "conv_tx"), default="conv_tx")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--sujeto-reps", type=int, default=12)
    ap.add_argument("--folds-reps", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--folds-loso", nargs="*", type=int, default=[])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("ab_augment.json"))
    args = ap.parse_args()

    # Hiperparámetros por defecto del script de entrenamiento.
    base = dict(arch=args.arch, lr=1e-3, batch=64, hidden=128, dim=192, dropout=0.3,
                weight_decay=0.05, label_smoothing=0.1, paciencia=40, workers=args.workers,
                verbose=False, epochs=args.epochs, no_augment=False, p_yaw=0.0)

    _, catalog = load_catalog()
    grupos = load_groups()
    X, y, sujetos, reps, clases = tk.cargar(Path(args.data), catalog)
    n = len(clases)
    folds = ([("reps", r, (sujetos == args.sujeto_reps) & (reps == r)) for r in args.folds_reps] +
             [("loso", s, sujetos == s) for s in args.folds_loso])
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    # Se guarda tras cada corrida: relanzar el mismo comando retoma lo pendiente.
    previo = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    corridas = previo.get("corridas", {})

    for nombre in args.variantes:
        cfg = Namespace(**base, **VARIANTES[nombre])
        for seed in range(args.seeds):
            for tipo, f, m_te in folds:
                clave = f"{nombre}|{seed}|{tipo}{f}"
                if clave in corridas:
                    continue
                t0 = time.time()
                modelo, hist = tk.entrenar(cfg, X[~m_te], y[~m_te], X[m_te], y[m_te], n,
                                        args.epochs, seed, early_stopping=False,
                                        sujetos_tr=sujetos[~m_te])
                r = tk.evaluar(modelo, X[m_te], y[m_te], n, catalog, grupos)
                keras.backend.clear_session()
                corridas[clave] = {
                    "tipo": tipo, "accuracy": r["accuracy"],
                    "porGrupo": {g: {"accuracy": v["accuracy"], "n": v["n"]}
                                 for g, v in r["porGrupo"].items()},
                    "paresMasConfundidos": r["paresMasConfundidos"][:5],
                    "minutos": round((time.time() - t0) / 60, 1),
                    "historia": tk.historia_redondeada(hist),
                }
                args.out.write_text(json.dumps({"config": config, "corridas": corridas},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
                txt = "  ".join(f"{g} {v['accuracy']:.3f}" for g, v in r["porGrupo"].items())
                print(f"  {clave}: {txt}  ({corridas[clave]['minutos']} min)", flush=True)

    resumen = resumir(corridas, args.variantes, args.seeds, folds, grupos)
    print("\n" + "=" * 72)
    print("RESULTADO (media ± desvío entre semillas, accuracy sobre todos los folds)")
    print("=" * 72)
    for nombre, tipos in resumen.items():
        for tipo, gs in tipos.items():
            txt = "  ".join(f"{g} {v['media']:.3f}±{v['desvio']:.3f}" for g, v in gs.items())
            print(f"{nombre:11s} [{tipo}]  {txt}")

    args.out.write_text(json.dumps({"config": config, "corridas": corridas, "resumen": resumen},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
