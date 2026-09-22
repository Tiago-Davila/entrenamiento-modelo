"""Contrato de keypoints del modelo secuencial LSA-T.

Este módulo es la referencia del productor Python. El espejo Kotlin vive en
``SequenceKeypointContract.kt`` y debe mantenerse compatible con estos valores.

Entrada por cuadro: 42 landmarks de manos, primero izquierda y luego derecha,
con ejes intercalados ``[x0, y0, z0, x1, y1, z1, ...]``.

La normalización conserva el comportamiento usado para generar el dataset:

* centro por cuadro calculado sobre x/y de los 42 landmarks, incluyendo ceros;
* se resta ese centro solo en x/y;
* una única escala global es el máximo absoluto x/y de toda la secuencia;
* z no se centra ni se escala;
* las secuencias largas se muestrean con división entera;
* las secuencias cortas se rellenan con cuadros completamente cero.
"""

from __future__ import annotations

import numpy as np


MAX_FRAMES = 75
LANDMARKS = 42
COORDS_PER_LANDMARK = 3
INPUT_DIM = LANDMARKS * COORDS_PER_LANDMARK


def _as_keypoints(kps: np.ndarray) -> np.ndarray:
    values = np.asarray(kps)
    if values.ndim == 2:
        if values.shape[1] != INPUT_DIM:
            raise ValueError(
                f"entrada [T,C] con C={values.shape[1]}; se esperaba {INPUT_DIM}",
            )
        values = values.reshape(values.shape[0], LANDMARKS, COORDS_PER_LANDMARK)
    if values.ndim != 3 or tuple(values.shape[1:]) != (LANDMARKS, COORDS_PER_LANDMARK):
        raise ValueError(
            f"entrada {tuple(values.shape)}; se esperaba [T, {LANDMARKS}, 3]",
        )
    if values.shape[0] == 0:
        raise ValueError("la secuencia debe tener al menos un cuadro")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"dtype {values.dtype}; se esperaba un tipo flotante")
    if not np.isfinite(values).all():
        raise ValueError("la secuencia contiene NaN o infinito")
    return values.astype(np.float32, copy=False)


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """Normaliza una secuencia absoluta y devuelve ``[T, 42, 3]`` float32."""
    values = _as_keypoints(kps)
    normalized = values.copy()

    for frame_index in range(normalized.shape[0]):
        frame = normalized[frame_index]
        center = frame[:, :2].mean(axis=0)
        frame[:, :2] -= center

    scale = float(np.abs(normalized[:, :, :2]).max())
    if scale > 1e-6:
        normalized[:, :, :2] /= scale
    return normalized


def sample_indices(total_frames: int, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Devuelve índices enteros reproducibles para una secuencia larga."""
    if total_frames <= 0:
        raise ValueError("total_frames debe ser positivo")
    if max_frames <= 0:
        raise ValueError("max_frames debe ser positivo")
    if total_frames <= max_frames:
        return np.arange(total_frames, dtype=np.int64)
    return np.asarray(
        [(i * (total_frames - 1)) // (max_frames - 1) for i in range(max_frames)],
        dtype=np.int64,
    )


def pad_or_crop(kps: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Ajusta una secuencia normalizada a ``max_frames`` con padding cero."""
    values = _as_keypoints(kps)
    total_frames = values.shape[0]
    if total_frames == max_frames:
        return values.copy()
    if total_frames > max_frames:
        return values[sample_indices(total_frames, max_frames)].copy()

    padding = np.zeros(
        (max_frames - total_frames, LANDMARKS, COORDS_PER_LANDMARK),
        dtype=np.float32,
    )
    return np.concatenate([values, padding], axis=0)


def build_input_tensor(kps: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Cadena completa: normalización, temporal y aplanado ``[T, 126]``."""
    normalized = normalize_keypoints(kps)
    fixed = pad_or_crop(normalized, max_frames)
    return fixed.reshape(max_frames, INPUT_DIM).astype(np.float32, copy=False)
