#!/usr/bin/env python3
"""
ab_augment.py — Mide si la augmentación mejora la generalización del
clasificador de señas.

Entrena N semillas SIN augmentación y N semillas CON augmentación, con split
por sujeto, y compara. Una sola corrida no alcanza: la varianza entre semillas
puede ser mayor que el efecto que se busca medir.

CALIBRACIÓN CONTRA LA BRECHA MEDIDA
El parámetro p_drop no es arbitrario. Se eligió a partir de la comparación
entre LSA64 (guantes, laboratorio) y LSA-T (señantes reales, sin guantes):

    componente        LSA64      LSA-T
    mano izquierda    26.3%      17.4%
    mano derecha       4.4%      16.3%   <-- la brecha importante

El modelo entrenó viendo la mano derecha casi siempre presente. En condiciones
reales falta ~16% del tiempo, y nunca aprendió a tolerarlo. p_drop simula esa
pérdida de detección durante el entrenamiento.

Uso:
    python ab_augment.py --data ./data_tasks --seeds 3
    python ab_augment.py --data ./data_tasks --seeds 3 --p-drop 0.15 0.25
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

FRAMES = 40
COORDS = 201
OFF_MANO_IZQ, OFF_MANO_DER, OFF_POSE = 0, 63, 126
IDX_HOMBRO_IZQ = OFF_POSE + 11 * 3
IDX_HOMBRO_DER = OFF_POSE + 12 * 3


# --------------------------------------------------------------------------
# Augmentación — cada transformación responde a una diferencia medida
# --------------------------------------------------------------------------

def a_puntos(x):
    """(T, 201) -> (T, 67, 3)"""
    return x.reshape(x.shape[0], COORDS // 3, 3)


def a_plano(p):
    return p.reshape(p.shape[0], COORDS)


def drop_mano(x, rng, p_izq, p_der):
    """Borra una mano entera en cuadros sueltos.

    Simula que MediaPipe pierda la detección, que es lo que ocurre sin guantes.
    Es la transformación más importante para esta brecha: el modelo nunca vio
    desaparecer la mano derecha durante el entrenamiento.
    """
    out = x.copy()
    T = out.shape[0]
    m_izq = rng.random(T) < p_izq
    m_der = rng.random(T) < p_der
    out[m_izq, OFF_MANO_IZQ:OFF_MANO_DER] = 0.0
    out[m_der, OFF_MANO_DER:OFF_POSE] = 0.0
    return out


# Pares izquierda/derecha de MediaPipe Pose dentro de los landmarks 0..24.
# Al espejar hay que intercambiarlos: si no, el bloque de mano izquierda queda
# con la mano derecha mientras el landmark de muñeca izquierda sigue siendo el
# izquierdo negado, y ambos dejan de corresponderse.
PARES_POSE = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
              (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def espejar(x):
    """Intercambia manos, intercambia pares izq/der de la pose e invierte x.

    Los 10 sujetos de LSA64 son diestros. Espejar da robustez para personas
    zurdas, que hoy fallan de forma sistemática.
    """
    p = a_puntos(x).copy()
    # manos
    izq = p[:, 0:21].copy()
    p[:, 0:21] = p[:, 21:42]
    p[:, 21:42] = izq
    # pose: intercambiar pares anatómicos (offset 42 en espacio de puntos)
    for a, b in PARES_POSE:
        tmp = p[:, 42 + a].copy()
        p[:, 42 + a] = p[:, 42 + b]
        p[:, 42 + b] = tmp
    p[:, :, 0] *= -1.0                      # x invertida (ya está centrada)
    out = a_plano(p)
    out[out == -0.0] = 0.0
    return out


def jitter(x, rng, sigma):
    """Ruido gaussiano sobre coordenadas detectadas.
    Los keypoints sin guantes son más ruidosos. No se toca lo que está en cero.
    """
    out = x.copy()
    mask = out != 0.0
    out[mask] += rng.normal(0.0, sigma, size=int(mask.sum())).astype(np.float32)
    return out


def escalar(x, rng, rango):
    """Escala global. Simula distancia a la cámara distinta."""
    f = float(rng.uniform(*rango))
    out = x.copy()
    mask = out != 0.0
    out[mask] *= f
    return out


def deformar_tiempo(x, rng, maxpct):
    """Reinterpola la secuencia con velocidad distinta y vuelve a 40 cuadros.
    Los señantes nativos señan más rápido que los sujetos de laboratorio.
    """
    T = x.shape[0]
    f = 1.0 + float(rng.uniform(-maxpct, maxpct))
    n = max(8, int(round(T * f)))
    src = np.linspace(0, T - 1, n)
    tmp = np.empty((n, COORDS), dtype=np.float32)
    for c in range(COORDS):
        tmp[:, c] = np.interp(src, np.arange(T), x[:, c])
    # volver a T con la MISMA aritmética entera del contrato
    idx = [(i * (n - 1)) // (T - 1) for i in range(T)]
    return tmp[idx]


def aumentar(x, rng, cfg):
    out = x
    if rng.random() < cfg["p_time"]:
        out = deformar_tiempo(out, rng, cfg["time_pct"])
    if rng.random() < cfg["p_scale"]:
        out = escalar(out, rng, cfg["scale"])
    if cfg["jitter"] > 0:
        out = jitter(out, rng, cfg["jitter"])
    out = drop_mano(out, rng, cfg["p_drop_izq"], cfg["p_drop_der"])
    if rng.random() < cfg["p_mirror"]:
        out = espejar(out)
    return out.astype(np.float32)


def expandir(X, y, cfg, factor, seed):
    """Devuelve el conjunto original más `factor` copias aumentadas."""
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(factor):
        Xs.append(np.stack([aumentar(x, rng, cfg) for x in X]))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# --------------------------------------------------------------------------

def cargar(d: Path):
    X = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy")
    s = np.load(d / "subjects.npy")
    clases = sorted(set(y.tolist()))
    mapa = {c: i for i, c in enumerate(clases)}
    return X, np.array([mapa[v] for v in y], dtype=np.int32), s, len(clases)


def construir(n_clases, hidden=128, dropout=0.3):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES, COORDS)),
        tf.keras.layers.LSTM(hidden, return_sequences=True, unroll=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(hidden, unroll=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_clases),
    ])


def entrenar(Xtr, ytr, Xva, yva, Xte, yte, n_clases, seed, args):
    tf.keras.utils.set_random_seed(seed)
    m = construir(n_clases, args.hidden, args.dropout)
    m.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=args.epochs, batch_size=args.batch, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_accuracy", patience=args.paciencia,
                  restore_best_weights=True)])
    _, va = m.evaluate(Xva, yva, verbose=0)
    _, te = m.evaluate(Xte, yte, verbose=0)
    tr_acc = float(h.history["accuracy"][-1])
    return {"val": float(va), "test": float(te), "train": tr_acc,
            "epocas": len(h.history["loss"])}, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--factor", type=int, default=2,
                    help="copias aumentadas por muestra original")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--paciencia", type=int, default=15)
    ap.add_argument("--val-subjects", nargs="*", type=int, default=[9])
    ap.add_argument("--test-subjects", nargs="*", type=int, default=[10])
    # calibrado contra la brecha medida LSA64 vs LSA-T
    ap.add_argument("--p-drop", nargs=2, type=float, default=[0.10, 0.15],
                    metavar=("IZQ", "DER"),
                    help="prob. de borrar cada mano por cuadro (default 0.10 0.15)")
    ap.add_argument("--p-mirror", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.004)
    ap.add_argument("--p-scale", type=float, default=0.5)
    ap.add_argument("--p-time", type=float, default=0.5)
    ap.add_argument("--time-pct", type=float, default=0.20)
    ap.add_argument("--out", default="ab_augment.json")
    args = ap.parse_args()

    cfg = {
        "p_drop_izq": args.p_drop[0], "p_drop_der": args.p_drop[1],
        "p_mirror": args.p_mirror, "jitter": args.jitter,
        "p_scale": args.p_scale, "scale": (0.85, 1.15),
        "p_time": args.p_time, "time_pct": args.time_pct,
    }

    d = Path(args.data)
    X, y, suj, n_clases = cargar(d)
    val_s, test_s = set(args.val_subjects), set(args.test_subjects)
    m_te = np.isin(suj, list(test_s)); m_va = np.isin(suj, list(val_s))
    m_tr = ~(m_te | m_va)
    Xtr, ytr = X[m_tr], y[m_tr]
    Xva, yva = X[m_va], y[m_va]
    Xte, yte = X[m_te], y[m_te]

    print(f"Datos: {X.shape} | clases {n_clases}")
    print(f"Split por sujeto -> train {len(ytr)} | val {len(yva)} (suj "
          f"{sorted(val_s)}) | test {len(yte)} (suj {sorted(test_s)})")
    print(f"\nCeros actuales en train: mano_izq "
          f"{np.mean(Xtr[...,0:63]==0):.1%} | mano_der {np.mean(Xtr[...,63:126]==0):.1%}")

    Xa, ya = expandir(Xtr, ytr, cfg, args.factor, seed=999)
    print(f"Tras augmentación:       mano_izq "
          f"{np.mean(Xa[...,0:63]==0):.1%} | mano_der {np.mean(Xa[...,63:126]==0):.1%}")
    print(f"  (objetivo aproximado, medido en señantes reales: 17% y 16%)")
    print(f"  train {len(ytr)} -> {len(ya)} muestras\n")

    res = {"sin": [], "con": []}
    mejor_test, mejor_modelo = -1.0, None

    for i in range(args.seeds):
        seed = 42 + i
        print(f"--- semilla {seed} ---")
        r0, _ = entrenar(Xtr, ytr, Xva, yva, Xte, yte, n_clases, seed, args)
        res["sin"].append(r0)
        print(f"  SIN aug: val {r0['val']:.4f}  test {r0['test']:.4f}  "
              f"train {r0['train']:.4f}  ({r0['epocas']} épocas)")

        Xa, ya = expandir(Xtr, ytr, cfg, args.factor, seed=seed)
        r1, m1 = entrenar(Xa, ya, Xva, yva, Xte, yte, n_clases, seed, args)
        res["con"].append(r1)
        print(f"  CON aug: val {r1['val']:.4f}  test {r1['test']:.4f}  "
              f"train {r1['train']:.4f}  ({r1['epocas']} épocas)")
        if r1["test"] > mejor_test:
            mejor_test, mejor_modelo = r1["test"], m1

    print("\n" + "=" * 58)
    print("RESULTADO")
    print("=" * 58)
    for k, etiqueta in (("sin", "SIN augmentación"), ("con", "CON augmentación")):
        te = [r["test"] for r in res[k]]
        va = [r["val"] for r in res[k]]
        tr = [r["train"] for r in res[k]]
        print(f"{etiqueta}")
        print(f"   test  {np.mean(te):.4f} ± {np.std(te):.4f}   "
              f"(min {min(te):.4f}, max {max(te):.4f})")
        print(f"   val   {np.mean(va):.4f} ± {np.std(va):.4f}")
        print(f"   gap train-val {np.mean(tr)-np.mean(va):+.4f}   "
              f"(menor = menos sobreajuste)")

    d_te = np.mean([r["test"] for r in res["con"]]) - np.mean([r["test"] for r in res["sin"]])
    sd = max(np.std([r["test"] for r in res["sin"]]),
             np.std([r["test"] for r in res["con"]]))
    print(f"\nDiferencia en test: {d_te:+.4f}")
    if abs(d_te) < sd:
        print("  MENOR que la desviación entre semillas: NO concluyente.")
        print("  Correr más semillas o ajustar la intensidad antes de decidir.")
    elif d_te > 0:
        print("  La augmentación MEJORA de forma consistente.")
    else:
        print("  La augmentación EMPEORA. Bajar la intensidad (--p-drop, --jitter)")
        print("  o revisar si el espejado es apropiado para estas señas.")

    Path(args.out).write_text(json.dumps(
        {"config": cfg, "factor": args.factor, "semillas": args.seeds,
         "resultados": res}, indent=2), encoding="utf-8")
    print(f"\nGuardado en {args.out}")

    if mejor_modelo is not None:
        ruta = d / "modelo_aug.keras"
        mejor_modelo.save(ruta)
        print(f"Mejor modelo con augmentación: {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())