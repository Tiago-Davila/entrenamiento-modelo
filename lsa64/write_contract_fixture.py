#!/usr/bin/env python3
"""
write_contract_fixture.py — Fixture de paridad del contrato v3 Python <-> Kotlin.

Cada caso trae cuadros CRUDOS (como los entrega MediaPipe: x,y en [0,1] de la
imagen), el tamaño de imagen y el tensor 40x168 que produce
eva_contract.build_input_v3. El test unitario de Android
(KeypointContractTest) recorre el mismo archivo con KeypointContract y exige
el mismo tensor.

Los cuadros se derivan de una muestra real de LSA64 v3, reproyectada a otras
cámaras y distancias, así los valores tienen forma de seña y no de ruido.

Uso:
    python write_contract_fixture.py --data ../data/processed_v3 \
        --out ../../helpi/app/src/test/resources/contract_v3_fixture.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eva_contract import (
    CONTRACT_VERSION,
    COORDS,
    FRAMES,
    LEFT_SHOULDER_POINT,
    POINTS,
    RIGHT_SHOULDER_POINT,
    build_input_v3,
)


def reproyectar(x_v3: np.ndarray, T: int, width: int, height: int,
                ancho_hombros: float, centro: tuple[float, float]) -> np.ndarray:
    """Muestra v3 (40,168) -> T cuadros crudos normalizados para otra cámara."""
    pts = x_v3.reshape(FRAMES, POINTS, 3).astype(np.float64)
    presente = np.any(pts != 0, axis=-1)
    src = np.round(np.linspace(0, FRAMES - 1, T)).astype(int)
    pts, presente = pts[src], presente[src]
    px = pts * ancho_hombros
    raw = np.zeros_like(px)
    raw[..., 0] = (px[..., 0] + centro[0]) / width
    raw[..., 1] = (px[..., 1] + centro[1]) / height
    raw[..., 2] = px[..., 2] / width
    raw[~presente] = 0.0
    return raw.reshape(T, COORDS).astype(np.float32)


def caso(nombre: str, raw: np.ndarray, width: int, height: int) -> dict:
    d = {"nombre": nombre, "width": width, "height": height,
         "cuadros": [[float(v) for v in f] for f in raw]}
    try:
        d["tensor"] = [[float(v) for v in f] for f in build_input_v3(raw, width, height)]
    except ValueError as e:
        d["error"] = str(e)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed_v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = np.load(Path(args.data) / "X.npy")
    a, b, c = X[rng.choice(len(X), size=3, replace=False)]
    hl, hr = LEFT_SHOULDER_POINT * 3, RIGHT_SHOULDER_POINT * 3

    casos = []
    # 1. celular vertical, cerca, T=46 (caso del muestreo entero i=13 -> 15)
    casos.append(caso("vertical_480x640_T46", reproyectar(a, 46, 480, 640, 260.0, (250.0, 300.0)), 480, 640))
    # 2. celular horizontal 16:9, lejos
    casos.append(caso("horizontal_1280x720_T40", reproyectar(b, 40, 1280, 720, 300.0, (620.0, 330.0)), 1280, 720))
    # 3. secuencia corta (T<40: repite el último cuadro)
    casos.append(caso("corta_960x540_T12", reproyectar(c, 12, 960, 540, 210.0, (500.0, 260.0)), 960, 540))
    # 4. mano izquierda perdida en un tramo y hombros faltantes en algunos cuadros
    r = reproyectar(a, 55, 960, 540, 230.0, (470.0, 280.0))
    r[10:25, 0:63] = 0.0
    r[[0, 1, 30, 54], hl:hl + 3] = 0.0
    r[[31, 32], hr:hr + 3] = 0.0
    casos.append(caso("huecos_960x540_T55", r, 960, 540))
    # 5. cantidad impar de cuadros válidos (los demás casos tienen 40: par)
    r = reproyectar(b, 40, 1280, 720, 280.0, (640.0, 360.0))
    r[5, hl:hl + 3] = 0.0
    casos.append(caso("mediana_impar_1280x720", r, 1280, 720))
    # 6. sin hombros: la secuencia es inválida
    r = reproyectar(c, 40, 960, 540, 200.0, (480.0, 270.0))
    r[:, hl:hl + 3] = 0.0
    casos.append(caso("sin_hombros_invalida", r, 960, 540))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "descripcion": "Paridad del contrato Eva v3: cuadros crudos -> tensor 40x168.",
        "contractVersion": CONTRACT_VERSION,
        "frames": FRAMES,
        "coordenadas": COORDS,
        "tolerancia": 1e-5,
        "casos": casos,
    }), encoding="utf-8")
    print(f"{len(casos)} casos -> {out.resolve()} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
