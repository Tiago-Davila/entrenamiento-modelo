"""Contrato único de Eva LSA64.

El modelo descarta los landmarks faciales de Pose (0..10). Conserva ambas
manos y el tronco/brazos de Pose (11..24), por lo que cada cuadro tiene 168
coordenadas. Este archivo es la única fuente de verdad para preprocesamiento,
augmentación y entrenamiento.

CONTRATO v3 (vigente)
    Misma forma 40 x 168 y mismo layout de bloques que v2. Cambian los valores:
    la entrada queda en "anchos de hombro" y deja de depender de la cámara, la
    orientación y la distancia.

    1. Muestreo entero de 40 cuadros:
         T >= 40: idx[i] = (i*(T-1)) // 39   (igual que v2)
         T <  40: idx[i] = (i*T) // 40       (estira repitiendo cuadros parejo)
       v2 con T < 40 repetía el último cuadro al final: con el teléfono a
       12-24 fps la seña quedaba comprimida al principio y el resto congelado.
    2. Landmarks presentes a píxeles: X = x*W, Y = y*H, Z = z*W.
       (MediaPipe expresa z en la misma escala que x.)
    3. Centro por cuadro: punto medio de los hombros (pose 11/12) en píxeles,
       restado a X e Y. Z no se centra. Un cuadro sin ambos hombros usa el
       centro del cuadro válido más cercano (el anterior si hay empate).
    4. Escala s: mediana, sobre los cuadros con ambos hombros, de la distancia
       euclídea X/Y entre hombros (cantidad par: promedio de los dos centrales).
       Todo landmark presente se divide por s (X, Y y Z).
    5. Sin hombros en ningún cuadro: la secuencia es inválida (ValueError).
    6. Ausente sigue siendo (0, 0, 0).

    Toda la aritmética es float32 y en este orden, para que el espejo Kotlin
    (KeypointContract.kt) produzca el mismo tensor.

CONTRATO v2 (histórico)
    Solo el paso 3 sobre coordenadas normalizadas de MediaPipe, sin escala.
    Se conserva para convertir datasets viejos y para brecha_dominio.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 3

FRAMES = 40
HAND_LANDMARKS = 21
POSE_SOURCE_INDICES = tuple(range(11, 25))
POSE_LANDMARKS = len(POSE_SOURCE_INDICES)

LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = HAND_LANDMARKS * 3
POSE_OFFSET = HAND_LANDMARKS * 2 * 3
COORDS = POSE_OFFSET + POSE_LANDMARKS * 3
POINTS = COORDS // 3

# Pose 11 y 12 pasan a ser las posiciones locales 0 y 1 del bloque reducido.
LEFT_SHOULDER_OFFSET = POSE_OFFSET
RIGHT_SHOULDER_OFFSET = POSE_OFFSET + 3
LEFT_SHOULDER_POINT = LEFT_SHOULDER_OFFSET // 3
RIGHT_SHOULDER_POINT = RIGHT_SHOULDER_OFFSET // 3

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
    """Muestreo entero exacto v3, igual al que implementa Android.

    T >= N: idx[i] = (i*(T-1)) // (N-1), submuestrea incluyendo el último.
    T <  N: idx[i] = (i*T) // N, cada cuadro se repite N//T o N//T+1 veces.
    """
    if total <= 0:
        return []
    if total >= FRAMES:
        return [(i * (total - 1)) // (FRAMES - 1) for i in range(FRAMES)]
    return [(i * total) // FRAMES for i in range(FRAMES)]


def sample_indices_v2(total: int) -> list[int]:
    """v2 (histórico): con T < N repetía el último cuadro al final."""
    if total <= 0:
        return []
    if total >= FRAMES:
        return [(i * (total - 1)) // (FRAMES - 1) for i in range(FRAMES)]
    return list(range(total)) + [total - 1] * (FRAMES - total)


def is_present(frame: np.ndarray) -> np.ndarray:
    """Máscara por landmark: (x,y,z)==0 representa ausencia."""
    return np.any(frame.reshape(-1, 3) != 0.0, axis=1)


# --------------------------------------------------------------------------
# Contrato v2 (histórico)
# --------------------------------------------------------------------------

def center_frame(frame: np.ndarray) -> np.ndarray:
    """v2: centra x/y en hombros y conserva ceros de landmarks no detectados."""
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
    """v2: centra cada cuadro de una secuencia (T, 168)."""
    if sequence.ndim != 2 or sequence.shape[1] != COORDS:
        raise ValueError(f"Se esperaba (T, {COORDS}), llegó {sequence.shape}")
    return np.stack([center_frame(frame) for frame in sequence]).astype(np.float32)


# --------------------------------------------------------------------------
# Contrato v3
# --------------------------------------------------------------------------

def _median_f32(values: np.ndarray) -> np.float32:
    """Mediana en float32 con la regla del contrato (par: promedio central)."""
    v = np.sort(values.astype(np.float32))
    n = len(v)
    if n % 2 == 1:
        return v[n // 2]
    return np.float32((v[n // 2 - 1] + v[n // 2]) / np.float32(2.0))


def _nearest_valid(valid: np.ndarray) -> np.ndarray:
    """Para cada cuadro, índice del cuadro válido más cercano (anterior si empata)."""
    idx = np.flatnonzero(valid)
    out = np.empty(len(valid), dtype=np.int64)
    for t in range(len(valid)):
        d = np.abs(idx - t)
        out[t] = idx[np.argmin(d)]      # argmin devuelve el primero: el anterior
    return out


def normalize_pixels_v3(pts_px: np.ndarray) -> np.ndarray:
    """Pasos 3..6 del contrato sobre puntos ya en píxeles.

    pts_px: (T, 56, 3) float32 con X=x*W, Y=y*H, Z=z*W y ceros en ausentes.
    Devuelve (T, 168) float32.
    """
    pts = np.asarray(pts_px, dtype=np.float32).copy()
    if pts.ndim != 3 or pts.shape[1:] != (POINTS, 3):
        raise ValueError(f"Se esperaba (T, {POINTS}, 3), llegó {pts.shape}")
    present = np.any(pts != 0.0, axis=-1)
    valid = present[:, LEFT_SHOULDER_POINT] & present[:, RIGHT_SHOULDER_POINT]
    if not valid.any():
        raise ValueError("secuencia sin hombros en ningún cuadro: no se puede normalizar")

    ls = pts[:, LEFT_SHOULDER_POINT]
    rs = pts[:, RIGHT_SHOULDER_POINT]
    dx = ls[:, 0] - rs[:, 0]
    dy = ls[:, 1] - rs[:, 1]
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    s = _median_f32(dist[valid])
    if not np.isfinite(s) or s <= np.float32(1e-6):
        raise ValueError(f"ancho de hombros inválido: {s}")

    cx = ((ls[:, 0] + rs[:, 0]) / np.float32(2.0)).astype(np.float32)
    cy = ((ls[:, 1] + rs[:, 1]) / np.float32(2.0)).astype(np.float32)
    src = _nearest_valid(valid)
    cx, cy = cx[src], cy[src]

    for t in range(pts.shape[0]):
        m = present[t]
        pts[t, m, 0] = (pts[t, m, 0] - cx[t]) / s
        pts[t, m, 1] = (pts[t, m, 1] - cy[t]) / s
        pts[t, m, 2] = pts[t, m, 2] / s
    return pts.reshape(pts.shape[0], COORDS).astype(np.float32)


def to_pixels(seq: np.ndarray, width: float, height: float) -> np.ndarray:
    """(T, 168) en coords normalizadas de MediaPipe -> (T, 56, 3) en píxeles."""
    if seq.ndim != 2 or seq.shape[1] != COORDS:
        raise ValueError(f"Se esperaba (T, {COORDS}), llegó {seq.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"dimensiones de imagen inválidas: {width}x{height}")
    pts = seq.reshape(seq.shape[0], POINTS, 3).astype(np.float32, copy=True)
    w, h = np.float32(width), np.float32(height)
    pts[..., 0] *= w
    pts[..., 1] *= h
    pts[..., 2] *= w
    return pts


def normalize_sequence_v3(seq_raw: np.ndarray, width: float, height: float) -> np.ndarray:
    """Contrato v3 completo sobre cuadros YA muestreados, crudos de MediaPipe.

    seq_raw: (T, 168) con x,y en [0,1] relativos a la imagen y ceros en ausentes.
    """
    return normalize_pixels_v3(to_pixels(seq_raw, width, height))


def build_input_v3(frames_raw: np.ndarray, width: float, height: float) -> np.ndarray:
    """Cadena completa de inferencia: muestreo entero + normalización v3."""
    frames_raw = np.asarray(frames_raw, dtype=np.float32)
    if len(frames_raw) == 0:
        raise ValueError("secuencia vacía")
    return normalize_sequence_v3(frames_raw[sample_indices(len(frames_raw))], width, height)


def v2_to_v3(x_v2: np.ndarray, width: float, height: float) -> np.ndarray:
    """Convierte una secuencia v2 (T, 168) a v3 sin volver a correr MediaPipe.

    Exacta cuando todos los cuadros tienen hombros: v2 = crudo - centro, y el
    centro en píxeles es el centro normalizado multiplicado por W/H, así que
    v2*W/H ya es el centrado en píxeles. Recentrar da cero de corrimiento.
    Si algún cuadro no tiene hombros, v2 lo dejó sin centrar y ese centro ya
    no se puede recuperar: se rechaza.
    """
    pts = to_pixels(x_v2, width, height)
    present = np.any(pts != 0.0, axis=-1)
    valid = present[:, LEFT_SHOULDER_POINT] & present[:, RIGHT_SHOULDER_POINT]
    if not valid.all():
        raise ValueError("v2 con cuadros sin hombros: la conversión no es exacta")
    return normalize_pixels_v3(pts)
