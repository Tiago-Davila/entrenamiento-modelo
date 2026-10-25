"""
augment.py — Aumentación en espacio de puntos clave para Eva (Helpi).

Opera sobre el contrato Eva v3 de 168 coordenadas:

    [  0: 63)  mano izquierda   21 landmarks x 3
    [ 63:126)  mano derecha     21 landmarks x 3
    [126:168)  pose 11..24      14 landmarks x 3 (sin cara)

en "anchos de hombro": centrado en el punto medio de los hombros y dividido
por la mediana del ancho de hombros. Internamente se trabaja como (T, 56, 3).

Reglas que el módulo respeta y que NO deben relajarse sin pensarlo:

  1. Toda transformación preserva la etiqueta. No hay reflexiones parciales,
     ni rotaciones grandes, ni inversión temporal.
  2. Se aplica SOLO al split de entrenamiento, y en cada época una variante
     nueva (ver train_keras_tflite.AugmentedSequence).
  3. Al final se vuelve a aplicar la normalización v3 del contrato
     (eva_contract.normalize_pixels_v3). Así ninguna muestra aumentada tiene
     algo que la app no pueda producir: una escala global, por ejemplo,
     dejaría hombros de ancho != 1 y el modelo aprendería de eso.
     Por lo mismo, lo que tiene sentido variar son las PROPORCIONES (largo de
     brazos, tamaño de mano, aspecto), no el tamaño total.
  4. Determinismo por semilla: la misma semilla produce la misma muestra.

Magnitudes en unidades v3. Referencias medidas en LSA64 v3: manos en x entre
-1,55 y +0,97 y en y entre -0,95 y +1,36 (percentiles 1 y 99).

Uso típico:

    aug = Augmenter(AugmentConfig())
    rng = np.random.default_rng(1234)
    x_aug = aug(x, rng)          # x: (40, 168) float32 v3
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from eva_contract import (
    COORDS,
    FRAMES,
    POSE_LANDMARKS,
    normalize_pixels_v3,
    sample_indices,
)

# ---------------------------------------------------------------------------
# Layout del contrato
# ---------------------------------------------------------------------------

N_HAND = 21
N_POSE = POSE_LANDMARKS
N_POINTS = N_HAND * 2 + N_POSE          # 56
N_COORDS = COORDS

IDX_LEFT_HAND = slice(0, N_HAND)                       # 0..20
IDX_RIGHT_HAND = slice(N_HAND, 2 * N_HAND)             # 21..41
IDX_POSE = slice(2 * N_HAND, N_POINTS)                 # 42..55

POSE_OFFSET = 2 * N_HAND                # los landmarks de pose arrancan acá

# Pose fuente 11 y 12 son los puntos locales 0 y 1 del bloque reducido.
POSE_LEFT_SHOULDER = 0
POSE_RIGHT_SHOULDER = 1
WRIST = 0                               # landmark 0 de cada mano

# Pares simétricos del bloque de pose reducido (fuente 11..24).
POSE_MIRROR_PAIRS = (
    (0, 1),    # hombro
    (2, 3),    # codo
    (4, 5),    # muñeca
    (6, 7),    # meñique
    (8, 9),    # índice
    (10, 11),  # pulgar
    (12, 13),  # cadera
)

def _build_mirror_permutation() -> np.ndarray:
    """Permutación de los 56 puntos que corresponde a espejar el cuerpo.

    Intercambia los dos bloques de mano completos e intercambia los pares
    simétricos de pose. Se combina con la negación de x.
    """
    perm = np.arange(N_POINTS)
    perm[IDX_LEFT_HAND] = np.arange(N_HAND) + N_HAND
    perm[IDX_RIGHT_HAND] = np.arange(N_HAND)
    for a, b in POSE_MIRROR_PAIRS:
        perm[POSE_OFFSET + a] = POSE_OFFSET + b
        perm[POSE_OFFSET + b] = POSE_OFFSET + a
    return perm


MIRROR_PERMUTATION = _build_mirror_permutation()


# ---------------------------------------------------------------------------
# Conversión de formato
# ---------------------------------------------------------------------------

def to_points(x: np.ndarray) -> np.ndarray:
    """(T, 168) -> (T, 56, 3)."""
    if x.ndim != 2 or x.shape[1] != N_COORDS:
        raise ValueError(f"Se esperaba (T, {N_COORDS}), se recibió {x.shape}")
    return x.reshape(x.shape[0], N_POINTS, 3)


def to_flat(p: np.ndarray) -> np.ndarray:
    """(T, 56, 3) -> (T, 168)."""
    return p.reshape(p.shape[0], N_COORDS)


def _present(p: np.ndarray) -> np.ndarray:
    return np.any(p != 0.0, axis=-1)


def _keep_absent(out: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Los landmarks ausentes vuelven a (0,0,0): una transformación afín los
    movería y dejarían de representar ausencia."""
    out[~present] = 0.0
    return out


# ---------------------------------------------------------------------------
# Transformaciones espaciales
# ---------------------------------------------------------------------------

def rotate_in_plane(p: np.ndarray, degrees: float) -> np.ndarray:
    """Rotación alrededor del eje de la cámara (roll).

    Modela: teléfono sostenido torcido, persona inclinada. En v3 x e y
    comparten unidades (píxeles / ancho de hombro), así que es exacta.
    """
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    present = _present(p)
    out = p.copy()
    x, y = p[..., 0], p[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return _keep_absent(out, present)


def rotate_yaw(p: np.ndarray, degrees: float, z_gain: float = 1.0) -> np.ndarray:
    """Rotación alrededor del eje vertical (persona no del todo de frente).

    Mezcla x con z. En v3 z está en la misma escala que x (z*W / ancho de
    hombros), pero la z de MediaPipe es relativa (a la muñeca en la mano, a
    la cadera en la pose) y ruidosa. Por eso es opcional y se evalúa en A/B.
    Al girar, los hombros se acortan en x; la renormalización final los
    devuelve a ancho 1 y agranda el resto, como pasa en la cámara real.
    """
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    present = _present(p)
    out = p.copy()
    x, z = p[..., 0], p[..., 2] * z_gain
    out[..., 0] = c * x + s * z
    out[..., 2] = (-s * x + c * z) / z_gain
    return _keep_absent(out, present)


def scale_aspect(p: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Escalado anisotrópico x/y.

    Una escala isotrópica no hace nada después de renormalizar; la diferencia
    entre ejes sí sobrevive y modela contextura y errores de relación de
    aspecto de la cámara.
    """
    present = _present(p)
    out = p.copy()
    out[..., 0] *= sx
    out[..., 1] *= sy
    return _keep_absent(out, present)


def scale_arms(p: np.ndarray, factor: float) -> np.ndarray:
    """Alarga o acorta brazos y manos respecto de los hombros.

    Modela la proporción brazo/hombros entre personas: cada punto del brazo
    se aleja del hombro de su lado en [factor]. Hombros y caderas quedan fijos.
    """
    present = _present(p)
    out = p.copy()
    sides = (
        (list(range(0, N_HAND)) + [POSE_OFFSET + i for i in (2, 4, 6, 8, 10)], POSE_LEFT_SHOULDER),
        (list(range(N_HAND, 2 * N_HAND)) + [POSE_OFFSET + i for i in (3, 5, 7, 9, 11)], POSE_RIGHT_SHOULDER),
    )
    for idx, shoulder in sides:
        anchor = p[:, POSE_OFFSET + shoulder, :2][:, None, :]
        out[:, idx, :2] = anchor + (p[:, idx, :2] - anchor) * factor
    return _keep_absent(out, present)


def scale_hands(p: np.ndarray, factor_left: float, factor_right: float) -> np.ndarray:
    """Agranda o achica cada mano alrededor de su muñeca (tamaño de mano)."""
    present = _present(p)
    out = p.copy()
    for block, f in ((IDX_LEFT_HAND, factor_left), (IDX_RIGHT_HAND, factor_right)):
        hand = p[:, block]
        wrist = hand[:, WRIST:WRIST + 1]
        out[:, block] = wrist + (hand - wrist) * f
    return _keep_absent(out, present)


# Cadenas de cada dedo dentro del bloque de mano (topología de MediaPipe):
# la base queda fija y cada articulación sigue a la anterior.
FINGER_CHAINS = (
    (1, 2, 3, 4),        # pulgar: CMC -> MCP -> IP -> punta
    (5, 6, 7, 8),        # índice: MCP -> PIP -> DIP -> punta
    (9, 10, 11, 12),     # medio
    (13, 14, 15, 16),    # anular
    (17, 18, 19, 20),    # meñique
)

# Puntos de mano del bloque de pose (fuente 17..22) y su muñeca (fuente 15/16).
POSE_HAND_POINTS = {"left": (4, (6, 8, 10)), "right": (5, (7, 9, 11))}


def scale_fingers(p: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Alarga o acorta las falanges de cada dedo sin cambiar sus ángulos.

    factors: (2, 5), un factor por mano (izquierda, derecha) y por dedo.
    Modela anatomía de mano distinta (dedos más largos o cortos, pulgar
    corto). Como los ángulos de flexión se conservan, una letra no se
    convierte en otra: es la variación de forma que sí preserva la etiqueta.
    """
    out = p.copy()
    for h, inicio in enumerate((0, N_HAND)):
        mano = p[:, inicio:inicio + N_HAND]
        nueva = out[:, inicio:inicio + N_HAND]
        presente = _present(mano).all(axis=1)          # cuadros con la mano entera
        for d, cadena in enumerate(FINGER_CHAINS):
            f = factors[h, d]
            for a, b in zip(cadena[:-1], cadena[1:]):
                seg = mano[:, b] - mano[:, a]
                nueva[presente, b] = nueva[presente, a] + seg[presente] * f
    return out


def _rotacion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Matriz 3x3: roll en el plano de la imagen (z), pitch (x), yaw (y)."""
    r, pt, yw = np.deg2rad([roll, pitch, yaw])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(pt), -np.sin(pt)], [0, np.sin(pt), np.cos(pt)]])
    ry = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
    return (rz @ rx @ ry).astype(np.float32)


def rotate_hand_local(p: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Gira cada mano alrededor de su muñeca (ángulo de muñeca, punto de vista).

    angles: (2, 3) grados (roll, pitch, yaw) por mano. Arrastra los puntos de
    mano del bloque de pose alrededor de la muñeca de pose. Los ángulos tienen
    que ser chicos: hay letras que se distinguen solo por la orientación.
    """
    present = _present(p)
    out = p.copy()
    for h, (inicio, lado) in enumerate(((0, "left"), (N_HAND, "right"))):
        R = _rotacion(*angles[h])
        mano = p[:, inicio:inicio + N_HAND]
        wrist = mano[:, WRIST:WRIST + 1]
        out[:, inicio:inicio + N_HAND] = wrist + (mano - wrist) @ R.T
        pw, puntos = POSE_HAND_POINTS[lado]
        idx = [POSE_OFFSET + i for i in puntos]
        ancla = p[:, POSE_OFFSET + pw:POSE_OFFSET + pw + 1]
        out[:, idx] = ancla + (p[:, idx] - ancla) @ R.T
    return _keep_absent(out, present)


def smooth_jitter(p: np.ndarray, sigma: np.ndarray, n_knots: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Deriva lenta sobre las manos: ruido en n_knots instantes, interpolado.

    El error de MediaPipe está correlacionado entre cuadros vecinos (la mano
    "flota" unos cuadros), cosa que el jitter independiente no modela.
    """
    T = p.shape[0]
    nudos = rng.normal(0.0, 1.0, size=(n_knots, 2 * N_HAND, 3)) * np.asarray(sigma).reshape(1, 1, 3)
    pos = np.linspace(0.0, n_knots - 1.0, T)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n_knots - 1)
    w = (pos - lo)[:, None, None]
    ruido = ((1.0 - w) * nudos[lo] + w * nudos[hi]).astype(p.dtype)
    out = p.copy()
    manos = out[:, :2 * N_HAND]
    presente = _present(manos)
    manos[presente] += ruido[presente]
    return out


def translate_z(p: np.ndarray, dz: float) -> np.ndarray:
    """Desplazamiento en profundidad.

    x,y no se trasladan porque el centrado del contrato los anula: sería
    un no-op disfrazado de aumentación.
    """
    out = p.copy()
    out[..., 2][_present(out)] += dz
    return out


def mirror(p: np.ndarray) -> np.ndarray:
    """Espejado anatómicamente correcto (diestro <-> zurdo).

    Tres operaciones simultáneas:
      1. negar x
      2. intercambiar los bloques de mano izquierda y derecha
      3. intercambiar los pares simétricos de pose

    Omitir el paso 3 no produce ningún error: solo entrena peor.
    """
    out = p.copy()
    out[..., 0] *= -1.0
    return out[:, MIRROR_PERMUTATION, :]


def jitter(p: np.ndarray, sigma: np.ndarray | float, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano que modela el temblor del estimador.

    Acepta un escalar o un array (3,) con un sigma por eje, en anchos de hombro.
    """
    sigma = np.asarray(sigma, dtype=p.dtype)
    if sigma.ndim == 0:
        noise = rng.normal(0.0, float(sigma), size=p.shape)
    else:
        noise = rng.normal(0.0, 1.0, size=p.shape) * sigma.reshape(1, 1, 3)
    out = p.copy()
    present = _present(out)
    out[present] += noise.astype(p.dtype)[present]
    return out


def drop_hand(
    p: np.ndarray,
    which: str,
    start: int,
    length: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Simula la pérdida de una mano durante una ventana de cuadros.

    En LSA64 casi nunca pasa: hay trípode, pared blanca y guantes
    fluorescentes. En uso real pasa todo el tiempo. El relleno es 0, que es
    lo que emite el extractor (Python y Android) cuando no detecta una mano.
    """
    idx = IDX_LEFT_HAND if which == "left" else IDX_RIGHT_HAND
    out = p.copy()
    end = min(start + length, p.shape[0])
    out[start:end, idx, :] = fill
    return out


def drop_hand_frames(p: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Pérdidas sueltas de una mano en cuadros aislados (parpadeo del detector)."""
    out = p.copy()
    for block in (IDX_LEFT_HAND, IDX_RIGHT_HAND):
        mask = rng.random(p.shape[0]) < prob
        out[mask, block, :] = 0.0
    return out


# ---------------------------------------------------------------------------
# Transformaciones temporales
# ---------------------------------------------------------------------------

def _resample(p: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Muestrea la secuencia en posiciones fraccionarias, con interpolación
    lineal entre cuadros vecinos. Un landmark ausente en alguno de los dos
    vecinos queda ausente (no se interpola contra el cero)."""
    T = p.shape[0]
    pos = np.clip(positions, 0.0, T - 1.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(p.dtype)[:, None, None]
    out = (1.0 - w) * p[lo] + w * p[hi]
    # Si falta en algún vecino, se toma el vecino más cercano tal cual
    # (presente o ausente) en lugar de interpolar contra (0,0,0).
    present = _present(p)
    both = present[lo] & present[hi]
    nearest = np.where(w < 0.5, p[lo], p[hi])
    out[~both] = nearest[~both]
    return out


def time_warp(
    p: np.ndarray,
    n_out: int,
    strength: float,
    n_control: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deformación temporal monótona no uniforme.

    Modela que la misma seña se ejecuta con aceleraciones desiguales: una
    parte más lenta, otra más rápida. Es la variación más grande entre
    personas reales y la que LSA64 tiene más subrepresentada (los diez
    sujetos imitaron el mismo video de referencia). La monotonía es
    obligatoria: garantiza que no se invierte el orden de los cuadros.
    """
    T = p.shape[0]
    knots = rng.uniform(1.0 - strength, 1.0 + strength, size=n_control)
    cum = np.concatenate([[0.0], np.cumsum(knots)])
    cum /= cum[-1]                                    # monótona en [0, 1]
    src = np.linspace(0.0, 1.0, len(cum))
    grid = np.linspace(0.0, 1.0, n_out)
    warped = np.interp(grid, src, cum) * (T - 1)
    return _resample(p, warped)


def temporal_crop(
    p: np.ndarray,
    n_out: int,
    max_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Recorta un tramo de los extremos y remuestrea a n_out cuadros.

    Modela que el segmentador de la app corta la seña en otro lugar que el
    editor de LSA64 (que deja reposo antes y después).
    """
    T = p.shape[0]
    a = rng.uniform(0.0, max_frac) * (T - 1)
    b = (1.0 - rng.uniform(0.0, max_frac)) * (T - 1)
    if b - a < 2.0:
        a, b = 0.0, float(T - 1)
    return _resample(p, np.linspace(a, b, n_out))


def simulate_fps(p: np.ndarray, captured: int) -> np.ndarray:
    """Simula una captura con menos cuadros y el muestreo del contrato.

    LSA64 es 60 fps (40 cuadros de ~128); el teléfono captura a 12-24 fps y un
    segmento trae ~20-40 cuadros. Se toman [captured] cuadros de los 40 y se
    vuelven a llevar a 40 con la regla entera del contrato para T < 40
    (repetición pareja, sin interpolar), exactamente como en Android.
    """
    T = p.shape[0]
    if captured >= T:
        return p.copy()
    keep = [(i * (T - 1)) // (captured - 1) for i in range(captured)] if captured > 1 else [0]
    short = p[keep]
    return short[sample_indices(len(short))]


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """Magnitudes en unidades v3 (ancho de hombros = 1)."""

    n_frames: int = FRAMES              # largo de salida; contrato Eva

    # Temporales
    p_temporal_crop: float = 0.5
    crop_max_frac: float = 0.2

    p_time_warp: float = 0.7
    warp_strength: float = 0.3
    warp_control_points: int = 5

    p_fps: float = 0.5
    fps_min_frames: int = 18            # ~12 fps en una seña de 1,5 s

    # Espejado (diestro <-> zurdo)
    p_mirror: float = 0.5

    # Espaciales
    p_rotate: float = 0.8
    rotate_deg: float = 15.0

    p_aspect: float = 0.6
    aspect_range: float = 0.08          # sx, sy independientes ±8%

    p_arm_scale: float = 0.6
    arm_scale_range: float = 0.12       # largo de brazo relativo ±12%

    p_hand_scale: float = 0.6
    hand_scale_range: float = 0.15      # tamaño de mano ±15%

    # Forma de la mano (lo que separa letras y señas parecidas)
    p_finger_scale: float = 0.5
    finger_scale_range: float = 0.08    # largo de falanges ±8% por dedo

    p_hand_rotate: float = 0.5
    hand_roll_deg: float = 10.0         # giro de muñeca en el plano de la imagen
    hand_tilt_deg: float = 8.0          # pitch/yaw: mezclan z, más chicos

    p_yaw: float = 0.0                  # opcional: evaluar en A/B (ver rotate_yaw)
    yaw_deg: float = 25.0
    yaw_z_gain: float = 1.0

    p_translate_z: float = 0.3
    translate_z_range: float = 0.1      # 0,02 de v2 * 1920/382

    # Degradaciones del extractor
    p_drop_hand: float = 0.3
    drop_len_frac: tuple = (0.1, 0.4)
    drop_fill: float = 0.0

    p_drop_frames: float = 0.3
    drop_frame_prob: float = 0.06

    p_jitter: float = 0.7
    # ~2-4 px de ruido del extractor sobre hombros de ~180 px en 960x540.
    jitter_sigma: tuple = (0.01, 0.01, 0.02)

    p_smooth_jitter: float = 0.5
    smooth_jitter_sigma: tuple = (0.015, 0.015, 0.02)
    smooth_jitter_knots: int = 5

    # Seguridad
    enforce_contract: bool = True


def perfil_un_senante(cfg: AugmentConfig) -> AugmentConfig:
    """Config más fuerte en proporciones para clases con un único señante.

    Hoy/Ayer/Mañana y el abecedario solo los grabó el sujeto 012: sin otras
    personas, el modelo aprendería su anatomía. Ampliar brazos, manos, dedos y
    aspecto simula otros cuerpos; lo temporal y la forma de la mano (ángulos)
    no se tocan, porque ahí está la etiqueta.
    """
    return replace(
        cfg,
        p_arm_scale=0.8, arm_scale_range=max(cfg.arm_scale_range, 0.18),
        p_hand_scale=0.8, hand_scale_range=max(cfg.hand_scale_range, 0.20),
        p_aspect=0.8, aspect_range=max(cfg.aspect_range, 0.12),
        p_finger_scale=0.8, finger_scale_range=max(cfg.finger_scale_range, 0.12),
    )


# ---------------------------------------------------------------------------
# Aumentador
# ---------------------------------------------------------------------------

class Augmenter:
    """Compone las transformaciones. Se aplica SOLO al split de entrenamiento.

    Aplicarlo a validación o test invalida la métrica: se estaría midiendo
    sobre datos que ninguna cámara podría producir.
    """

    def __init__(self, config: AugmentConfig | None = None):
        self.cfg = config or AugmentConfig()

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        p = to_points(np.asarray(x, dtype=np.float32))

        # --- Temporales con interpolación: cambian el largo ---
        if rng.random() < cfg.p_temporal_crop:
            p = temporal_crop(p, cfg.n_frames, cfg.crop_max_frac, rng)
        if rng.random() < cfg.p_time_warp:
            p = time_warp(p, cfg.n_frames, cfg.warp_strength,
                          cfg.warp_control_points, rng)
        if p.shape[0] != cfg.n_frames:
            p = _resample(p, np.linspace(0.0, p.shape[0] - 1.0, cfg.n_frames))
        # --- Frecuencia de captura: al final, porque duplica cuadros ---
        if rng.random() < cfg.p_fps:
            p = simulate_fps(p, int(rng.integers(cfg.fps_min_frames, cfg.n_frames)))

        # --- Espejado: antes de las rotaciones, para no componer signos ---
        if rng.random() < cfg.p_mirror:
            p = mirror(p)

        # --- Proporciones del cuerpo ---
        if rng.random() < cfg.p_arm_scale:
            p = scale_arms(p, 1.0 + rng.uniform(-cfg.arm_scale_range, cfg.arm_scale_range))
        if rng.random() < cfg.p_hand_scale:
            r = cfg.hand_scale_range
            p = scale_hands(p, 1.0 + rng.uniform(-r, r), 1.0 + rng.uniform(-r, r))
        if rng.random() < cfg.p_finger_scale:
            r = cfg.finger_scale_range
            p = scale_fingers(p, 1.0 + rng.uniform(-r, r, size=(2, 5)))
        if rng.random() < cfg.p_hand_rotate:
            lim = np.array([cfg.hand_roll_deg, cfg.hand_tilt_deg, cfg.hand_tilt_deg])
            p = rotate_hand_local(p, rng.uniform(-lim, lim, size=(2, 3)))
        if rng.random() < cfg.p_aspect:
            r = cfg.aspect_range
            p = scale_aspect(p, 1.0 + rng.uniform(-r, r), 1.0 + rng.uniform(-r, r))

        # --- Pose respecto de la cámara ---
        if cfg.p_yaw > 0.0 and rng.random() < cfg.p_yaw:
            p = rotate_yaw(p, rng.uniform(-cfg.yaw_deg, cfg.yaw_deg), cfg.yaw_z_gain)
        if rng.random() < cfg.p_rotate:
            p = rotate_in_plane(p, rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
        if rng.random() < cfg.p_translate_z:
            p = translate_z(p, rng.uniform(-cfg.translate_z_range, cfg.translate_z_range))

        # --- Degradaciones del extractor ---
        if rng.random() < cfg.p_drop_hand:
            which = "left" if rng.random() < 0.5 else "right"
            lo, hi = cfg.drop_len_frac
            length = max(1, int(rng.uniform(lo, hi) * p.shape[0]))
            start = int(rng.integers(0, max(1, p.shape[0] - length + 1)))
            p = drop_hand(p, which, start, length, cfg.drop_fill)
        if rng.random() < cfg.p_drop_frames:
            p = drop_hand_frames(p, cfg.drop_frame_prob, rng)
        if rng.random() < cfg.p_jitter:
            p = jitter(p, np.asarray(cfg.jitter_sigma, dtype=np.float32), rng)
        if rng.random() < cfg.p_smooth_jitter:
            p = smooth_jitter(p, np.asarray(cfg.smooth_jitter_sigma, dtype=np.float32),
                              cfg.smooth_jitter_knots, rng)

        # --- Restablecer el contrato v3 (centro y ancho de hombros = 1) ---
        out = normalize_pixels_v3(p.astype(np.float32))
        if cfg.enforce_contract:
            assert_contract(out)
        return out


# ---------------------------------------------------------------------------
# Workers de multiprocessing (entrenamiento)
# ---------------------------------------------------------------------------
# Van a nivel de módulo porque en Windows los procesos se crean con spawn y
# solo pueden recibir funciones importables. Cada worker recibe X una vez.

_WORKER_X: np.ndarray | None = None
_WORKER_AUGS: tuple[Augmenter, ...] = ()
_WORKER_PERFIL: np.ndarray | None = None


def worker_init(X: np.ndarray, config: AugmentConfig,
                config_fuerte: AugmentConfig | None = None,
                perfil: np.ndarray | None = None) -> None:
    """perfil[i] = 1 elige config_fuerte para la muestra i (clases de un solo
    señante); sin perfil todas usan config."""
    global _WORKER_X, _WORKER_AUGS, _WORKER_PERFIL
    _WORKER_X = X
    _WORKER_AUGS = (Augmenter(config), Augmenter(config_fuerte or config))
    _WORKER_PERFIL = perfil


def aumentar_lote(X: np.ndarray, idx: np.ndarray, augs: tuple[Augmenter, ...],
                  perfil: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Aumenta las muestras idx con el aumentador de su perfil. Lo usan el
    worker y el camino sin pool, así ambos dan exactamente lo mismo."""
    return np.stack([augs[0 if perfil is None else int(perfil[i])](X[i], rng)
                     for i in idx]).astype(np.float32)


def worker_batch(tarea: tuple[np.ndarray, tuple[int, ...]]) -> np.ndarray:
    """Aumenta un lote. La semilla (seed, época, lote) hace que el resultado no
    dependa de qué worker lo procese."""
    idx, semilla = tarea
    rng = np.random.default_rng(list(semilla))
    return aumentar_lote(_WORKER_X, idx, _WORKER_AUGS, _WORKER_PERFIL, rng)


# ---------------------------------------------------------------------------
# Verificación del contrato
# ---------------------------------------------------------------------------

def assert_contract(x: np.ndarray, tol: float = 1e-4) -> None:
    """Verifica que la salida sigue cumpliendo el contrato Eva v3."""
    if x.ndim != 2 or x.shape[1] != N_COORDS:
        raise AssertionError(f"Forma inválida: {x.shape}, se esperaba (T, {N_COORDS})")
    if not np.isfinite(x).all():
        raise AssertionError("La salida contiene NaN o infinitos")
    p = to_points(x)
    ls = p[:, POSE_OFFSET + POSE_LEFT_SHOULDER, :2]
    rs = p[:, POSE_OFFSET + POSE_RIGHT_SHOULDER, :2]
    mid = np.abs((ls + rs) * 0.5)
    if mid.max() > tol:
        raise AssertionError(
            f"Centrado roto: |punto medio de hombros| = {mid.max():.6f} > {tol}")
    width = np.median(np.linalg.norm(ls - rs, axis=-1))
    if abs(width - 1.0) > 1e-3:
        raise AssertionError(f"Escala rota: mediana del ancho de hombros = {width:.5f}")


def _self_test() -> None:
    """Pruebas mínimas. Ejecutar con: python augment.py"""
    rng = np.random.default_rng(0)

    # La permutación de espejado debe ser una involución.
    assert np.array_equal(MIRROR_PERMUTATION[MIRROR_PERMUTATION],
                          np.arange(N_POINTS)), "El espejado no es involutivo"
    assert np.array_equal(MIRROR_PERMUTATION[IDX_LEFT_HAND],
                          np.arange(N_HAND) + N_HAND)

    # Muestra sintética v3: hombros en ±0,5, resto al azar.
    T = FRAMES
    p = rng.normal(0.0, 0.5, size=(T, N_POINTS, 3)).astype(np.float32)
    p[:, POSE_OFFSET + POSE_LEFT_SHOULDER, :2] = [0.5, 0.0]
    p[:, POSE_OFFSET + POSE_RIGHT_SHOULDER, :2] = [-0.5, 0.0]
    x = normalize_pixels_v3(p)
    assert_contract(x)
    p = to_points(x)

    # Espejar dos veces devuelve el original.
    assert np.allclose(to_flat(mirror(mirror(p))), x, atol=1e-6), \
        "Doble espejado no es identidad"

    # Rotar 0 grados es identidad.
    assert np.allclose(rotate_in_plane(p, 0.0), p)

    # Brazos: hombros fijos, mano se aleja del hombro.
    q = scale_arms(p, 1.2)
    assert np.allclose(q[:, IDX_POSE][:, :2], p[:, IDX_POSE][:, :2])
    assert not np.allclose(q[:, IDX_LEFT_HAND], p[:, IDX_LEFT_HAND])

    # Tamaño de mano: la muñeca no se mueve.
    q = scale_hands(p, 1.2, 0.8)
    assert np.allclose(q[:, WRIST], p[:, WRIST])

    # Frecuencia de captura: 40 cuadros, solo cuadros originales, en orden.
    ramp = np.zeros((T, N_POINTS, 3), dtype=np.float32)
    ramp[:, :, 0] = np.arange(1, T + 1)[:, None]
    f = simulate_fps(ramp, 20)[:, 0, 0]
    assert len(f) == T and np.all(np.diff(f) >= 0) and len(set(f.tolist())) == 20

    # Ausentes siguen ausentes tras una rotación.
    q = p.copy()
    q[:10, IDX_LEFT_HAND] = 0.0
    assert np.all(rotate_in_plane(q, 10.0)[:10, IDX_LEFT_HAND] == 0.0)

    # El aumentador respeta forma y contrato, y es determinista por semilla.
    aug = Augmenter(AugmentConfig())
    a = aug(x, np.random.default_rng(7))
    b = aug(x, np.random.default_rng(7))
    assert a.shape == (T, N_COORDS), f"Forma inesperada: {a.shape}"
    assert np.array_equal(a, b), "No es determinista con la misma semilla"
    c = aug(x, np.random.default_rng(8))
    assert not np.array_equal(a, c), "Semillas distintas dan el mismo resultado"

    # Lote largo, con yaw y el perfil fuerte: ninguna combinación rompe el contrato.
    for cfg in (AugmentConfig(p_yaw=0.5), perfil_un_senante(AugmentConfig())):
        aug = Augmenter(cfg)
        r = np.random.default_rng(123)
        for _ in range(500):
            assert_contract(aug(x, r))

    # Dedos: se conservan los ángulos de flexión y las bases quedan fijas.
    def angulos(q):
        out = []
        for inicio in (0, N_HAND):
            for cadena in FINGER_CHAINS:
                for a, b, c in zip(cadena, cadena[1:], cadena[2:]):
                    u = q[:, inicio + b] - q[:, inicio + a]
                    v = q[:, inicio + c] - q[:, inicio + b]
                    out.append(np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) *
                                                     np.linalg.norm(v, axis=-1)))
        return np.array(out)
    q = scale_fingers(p, 1.0 + rng.uniform(-0.15, 0.15, size=(2, 5)))
    assert np.allclose(angulos(q), angulos(p), atol=1e-4), "scale_fingers cambió ángulos"
    assert np.allclose(q[:, [0, 1, 5, 9, 13, 17]], p[:, [0, 1, 5, 9, 13, 17]])
    assert not np.allclose(q[:, 8], p[:, 8])

    # Mano local: la muñeca queda fija y las distancias a la muñeca también.
    q = rotate_hand_local(p, np.array([[10.0, 8.0, -8.0], [-10.0, 5.0, 3.0]]))
    for inicio in (0, N_HAND):
        assert np.allclose(q[:, inicio + WRIST], p[:, inicio + WRIST], atol=1e-6)
        d0 = np.linalg.norm(p[:, inicio:inicio + N_HAND] - p[:, inicio:inicio + 1], axis=-1)
        d1 = np.linalg.norm(q[:, inicio:inicio + N_HAND] - q[:, inicio:inicio + 1], axis=-1)
        assert np.allclose(d0, d1, atol=1e-5), "rotate_hand_local deformó la mano"
    assert np.allclose(rotate_hand_local(p, np.zeros((2, 3))), p, atol=1e-6)

    # Las transformaciones de mano no inventan una mano ausente.
    q = p.copy()
    q[:12, IDX_RIGHT_HAND] = 0.0
    for t in (scale_fingers(q, np.full((2, 5), 1.1)),
              rotate_hand_local(q, np.full((2, 3), 7.0)),
              smooth_jitter(q, np.array([0.02, 0.02, 0.02]), 5, np.random.default_rng(1))):
        assert np.all(t[:12, IDX_RIGHT_HAND] == 0.0), "una mano ausente dejó de estar en cero"

    # Deriva lenta: suave en el tiempo (diferencias chicas entre cuadros).
    d = smooth_jitter(p, np.array([0.05, 0.05, 0.05]), 5, np.random.default_rng(2)) - p
    assert np.abs(np.diff(d[:, :2 * N_HAND], axis=0)).max() < 0.05

    # Workers: el perfil elige el aumentador y el resultado es determinista.
    worker_init(np.stack([x, x]), AugmentConfig(), perfil_un_senante(AugmentConfig()),
                np.array([0, 1]))
    l1 = worker_batch((np.array([0, 1]), (1, 2, 3)))
    l2 = worker_batch((np.array([0, 1]), (1, 2, 3)))
    assert l1.shape == (2, T, N_COORDS) and np.array_equal(l1, l2)

    # La deformación temporal debe ser monótona (no invierte cuadros).
    w = time_warp(ramp, 32, 0.4, 5, np.random.default_rng(3))[:, 0, 0]
    assert np.all(np.diff(w) >= -1e-6), "La deformación temporal no es monótona"

    print("Todas las pruebas pasaron.")


if __name__ == "__main__":
    _self_test()
