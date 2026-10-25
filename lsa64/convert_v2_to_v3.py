#!/usr/bin/env python3
"""
convert_v2_to_v3.py — Pasa un dataset Eva v2 ya procesado al contrato v3 sin
volver a correr MediaPipe.

Es exacto porque v2 guarda crudo - centro_de_hombros y los hombros siguen en el
vector (ver eva_contract.v2_to_v3). Lo único que v2 no guardó es el tamaño de
imagen: se pasa con --width/--height y, si se indica --videos, se verifica
contra los videos originales antes de convertir.

Uso:
    python convert_v2_to_v3.py --src ../data/processed --out ../data/processed_v3 \
        --videos ./data/processed/raw/all
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from eva_contract import (
    CONTRACT_VERSION,
    LEFT_SHOULDER_POINT,
    POINTS,
    RIGHT_SHOULDER_POINT,
    v2_to_v3,
)


def verificar_resolucion(videos: Path, width: int, height: int, muestras: int) -> None:
    import cv2

    archivos = sorted(videos.glob("*.mp4"))
    if not archivos:
        raise SystemExit(f"No hay videos en {videos}")
    paso = max(1, len(archivos) // muestras)
    for f in archivos[::paso][:muestras]:
        cap = cv2.VideoCapture(str(f))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if (w, h) != (width, height):
            raise SystemExit(f"{f.name} es {w}x{h}, no {width}x{height}. No se convierte.")
    print(f"Resolución verificada en {min(muestras, len(archivos))} videos: {width}x{height}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="carpeta v2 con X.npy, y.npy, subjects.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--videos", type=Path, default=None,
                    help="videos originales, para verificar la resolución")
    ap.add_argument("--muestras", type=int, default=20)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if args.videos:
        verificar_resolucion(args.videos, args.width, args.height, args.muestras)
    else:
        print(f"AVISO: resolución {args.width}x{args.height} asumida, sin verificar.")

    X = np.load(src / "X.npy").astype(np.float32)
    X3 = np.stack([v2_to_v3(x, args.width, args.height) for x in X]).astype(np.float32)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X3)
    for nombre in ("y.npy", "subjects.npy", "reps.npy"):
        if (src / nombre).exists():
            shutil.copy2(src / nombre, out / nombre)
    sizes = np.tile(np.array([args.width, args.height], dtype=np.int32), (len(X3), 1))
    np.save(out / "sizes.npy", sizes)

    # --- control: el ancho de hombros debe quedar en ~1 ---
    p = X3.reshape(len(X3), -1, POINTS, 3)[:, :, :, :2]
    ancho = np.linalg.norm(p[:, :, LEFT_SHOULDER_POINT] - p[:, :, RIGHT_SHOULDER_POINT], axis=-1)
    medianas = np.median(ancho, axis=1)
    sujetos = np.load(src / "subjects.npy") if (src / "subjects.npy").exists() else None

    meta_src = src / "meta.json"
    meta = json.loads(meta_src.read_text(encoding="utf-8")) if meta_src.exists() else {}
    meta.update({
        "contractVersion": CONTRACT_VERSION,
        "coordenadas": X3.shape[2],
        "normalizacion": "v3: pixeles (x*W, y*H, z*W), centro en hombros, / mediana del ancho de hombros",
        "convertidoDesde": str(src.resolve()),
        "imagen": {"width": args.width, "height": args.height,
                   "verificada": bool(args.videos)},
    })
    meta.pop("centrado", None)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"X v3: {X3.shape}  rango [{X3.min():+.2f}, {X3.max():+.2f}]  std {X3.std():.3f}")
    print(f"Ancho de hombros (mediana por secuencia): {medianas.mean():.4f} ± {medianas.std():.1e}")
    print(f"Ancho de hombros por cuadro: min {ancho.min():.3f}  max {ancho.max():.3f}")
    if sujetos is not None:
        print("Rango x de manos por sujeto (debería ser parecido entre sujetos):")
        manos = X3.reshape(len(X3), -1, POINTS, 3)[:, :, :42, 0]
        for s in sorted(set(sujetos.tolist())):
            m = manos[sujetos == s]
            m = m[m != 0]
            print(f"  sujeto {s:2d}: p5 {np.percentile(m, 5):+.2f}  p95 {np.percentile(m, 95):+.2f}")
    print(f"Guardado en {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
