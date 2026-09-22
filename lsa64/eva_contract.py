"""Contrato único de Eva LSA64 v2.

El modelo descarta los landmarks faciales de Pose (0..10). Conserva ambas
manos y el tronco/brazos de Pose (11..24), por lo que cada cuadro tiene 168
coordenadas. Este archivo es la única fuente de verdad para preprocesamiento,
augmentación y entrenamiento.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FRAMES = 40
HAND_LANDMARKS = 21
POSE_SOURCE_INDICES = tuple(range(11, 25))
POSE_LANDMARKS = len(POSE_SOURCE_INDICES)

LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = HAND_LANDMARKS * 3
POSE_OFFSET = HAND_LANDMARKS * 2 * 3
COORDS = POSE_OFFSET + POSE_LANDMARKS * 3

# Pose 11 y 12 pasan a ser las posiciones locales 0 y 1 del bloque reducido.
LEFT_SHOULDER_OFFSET = POSE_OFFSET
RIGHT_SHOULDER_OFFSET = POSE_OFFSET + 3

CATALOG_PATH = Path(__file__).with_name("catalogo_lsa64.json")


def load_catalog(path: Path = CATALOG_PATH) -> tuple[str, list[str]]:
    """Carga el catálogo oficial y falla si su estructura no es exacta."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    glosses = raw.get("glosas")
    if not isinstance(version, str) or not isinstance(glosses, list):
        raise ValueError(f"Catálogo inválido: {path}")
    if len(glosses) != 64 or any(not isinstance(g, str) or not g.strip() for g in glosses):
        raise ValueError(f"El catálogo debe contener exactamente 64 glosas: {path}")
    return version, glosses


def sample_indices(total: int) -> list[int]:
    """Muestreo entero exacto, igual al que deberá implementar Android v2."""
    if total <= 0:
        return []
    if total >= FRAMES:
        return [(i * (total - 1)) // (FRAMES - 1) for i in range(FRAMES)]
    return list(range(total)) + [total - 1] * (FRAMES - total)


def is_present(frame: np.ndarray) -> np.ndarray:
    """Máscara por landmark: (x,y,z)==0 representa ausencia."""
    return np.any(frame.reshape(-1, 3) != 0.0, axis=1)


def center_frame(frame: np.ndarray) -> np.ndarray:
    """Centra x/y en hombros y conserva ceros de landmarks no detectados."""
    if frame.shape != (COORDS,):
        raise ValueError(f"Se esperaba un cuadro de {COORDS}, llegó {frame.shape}")
    out = frame.astype(np.float32, copy=True)
    left = out[LEFT_SHOULDER_OFFSET:LEFT_SHOULDER_OFFSET + 3]
    right = out[RIGHT_SHOULDER_OFFSET:RIGHT_SHOULDER_OFFSET + 3]
    if not np.any(left) or not np.any(right):
        return out
    cx = (left[0] + right[0]) / 2.0
    cy = (left[1] + right[1]) / 2.0
    points = out.reshape(-1, 3)
    present = np.any(points != 0.0, axis=1)
    points[present, 0] -= cx
    points[present, 1] -= cy
    return out


def center_sequence(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != COORDS:
        raise ValueError(f"Se esperaba (T, {COORDS}), llegó {sequence.shape}")
    return np.stack([center_frame(frame) for frame in sequence]).astype(np.float32)
