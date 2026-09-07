"""
augment.py — Aumentación en espacio de puntos clave para Eva (Helpi).

Opera sobre el contrato de 201 coordenadas del proyecto:

    [  0: 63)  mano izquierda   21 landmarks x 3
    [ 63:126)  mano derecha     21 landmarks x 3
    [126:201)  pose 0..24       25 landmarks x 3

Internamente todo se trabaja como (T, 67, 3), donde 67 = 21 + 21 + 25.

Reglas que el módulo respeta y que NO deben relajarse sin pensarlo:

  1. Toda transformación preserva la etiqueta. No hay reflexiones parciales,
     ni rotaciones grandes, ni inversión temporal.
  2. Se aplica SOLO al split de entrenamiento.
  3. Después de cada transformación se restablece el centrado en el punto
     medio de los hombros en x,y (z NO se centra), que es el contrato.
  4. Determinismo por semilla: la misma semilla produce la misma muestra.

Uso típico:

    aug = Augmenter(AugmentConfig())
    rng = np.random.default_rng(1234)
    x_aug = aug(x, rng)          # x: (T, 201) float32

Nota sobre aritmética entera: el contrato de submuestreo con división entera
rige el camino de INFERENCIA, donde hace falta reproducibilidad bit a bit
entre plataformas. Acá estamos en entrenamiento, en Python, y la interpolación
en punto flotante es correcta y deseable. El submuestreo entero se aplica
después, sobre el resultado, igual que sobre una muestra sin aumentar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Layout del contrato
# ---------------------------------------------------------------------------

N_HAND = 21
N_POSE = 25
N_POINTS = N_HAND * 2 + N_POSE          # 67
N_COORDS = N_POINTS * 3                 # 201

IDX_LEFT_HAND = slice(0, N_HAND)                       # 0..20
IDX_RIGHT_HAND = slice(N_HAND, 2 * N_HAND)             # 21..41
IDX_POSE = slice(2 * N_HAND, N_POINTS)                 # 42..66

POSE_OFFSET = 2 * N_HAND                # los landmarks de pose arrancan acá

# Índices de pose de MediaPipe usados para el centrado del contrato.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# Pares simétricos de MediaPipe Pose (landmarks 0..24). El 0 (nariz) es su
# propio espejo. Si este mapa está mal, el espejado produce datos inválidos
# sin ningún error visible.
POSE_MIRROR_PAIRS = (
    (1, 4),    # ojo interno izq / der
    (2, 5),    # ojo izq / der
    (3, 6),    # ojo externo izq / der
    (7, 8),    # oreja
    (9, 10),   # comisura de la boca
    (11, 12),  # hombro
    (13, 14),  # codo
    (15, 16),  # muñeca
    (17, 18),  # meñique
    (19, 20),  # índice
    (21, 22),  # pulgar
    (23, 24),  # cadera
)


def _build_mirror_permutation() -> np.ndarray:
    """Permutación de los 67 puntos que corresponde a espejar el cuerpo.

    Intercambia los dos bloques de mano completos e intercambia los pares
    simétricos de pose. Se combina con la negación de x.
    """
    perm = np.arange(N_POINTS)
    # Bloques de mano: izquierda <-> derecha, manteniendo el orden interno
    # de los 21 landmarks (la topología de la mano es la misma en ambas).
    perm[IDX_LEFT_HAND] = np.arange(N_HAND) + N_HAND
    perm[IDX_RIGHT_HAND] = np.arange(N_HAND)
    # Pares simétricos de pose.
    for a, b in POSE_MIRROR_PAIRS:
        perm[POSE_OFFSET + a] = POSE_OFFSET + b
        perm[POSE_OFFSET + b] = POSE_OFFSET + a
    return perm


MIRROR_PERMUTATION = _build_mirror_permutation()


# ---------------------------------------------------------------------------
# Conversión de formato
# ---------------------------------------------------------------------------

def to_points(x: np.ndarray) -> np.ndarray:
    """(T, 201) -> (T, 67, 3)."""
    if x.ndim != 2 or x.shape[1] != N_COORDS:
        raise ValueError(f"Se esperaba (T, {N_COORDS}), se recibió {x.shape}")
    return x.reshape(x.shape[0], N_POINTS, 3)


def to_flat(p: np.ndarray) -> np.ndarray:
    """(T, 67, 3) -> (T, 201)."""
    return p.reshape(p.shape[0], N_COORDS)


def recenter(p: np.ndarray) -> np.ndarray:
    """Restablece el contrato: origen en el punto medio de los hombros.

    Solo x,y. El eje z NO se centra, por definición del contrato.
    """
    ls = p[:, POSE_OFFSET + POSE_LEFT_SHOULDER, :2]
    rs = p[:, POSE_OFFSET + POSE_RIGHT_SHOULDER, :2]
    mid = ((ls + rs) * 0.5)[:, None, :]        # (T, 1, 2)
    p = p.copy()
    p[:, :, :2] -= mid
    return p


# ---------------------------------------------------------------------------
# Transformaciones espaciales
# ---------------------------------------------------------------------------

def rotate_in_plane(p: np.ndarray, degrees: float) -> np.ndarray:
    """Rotación alrededor del eje de la cámara (roll).

    Modela: teléfono sostenido torcido, cámara inclinada.
    Es la rotación segura: x e y comparten unidades, así que es exacta.
    """
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    out = p.copy()
    x, y = p[..., 0], p[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def rotate_yaw(p: np.ndarray, degrees: float, z_gain: float = 1.0) -> np.ndarray:
    """Rotación alrededor del eje vertical. USAR CON CUIDADO.

    Modela: la persona no está exactamente de frente a la cámara. Es la
    variación más frecuente en uso real, pero mezcla x con z, y en MediaPipe
    esos ejes NO comparten escala: z de mano es relativo a la muñeca y z de
    pose es relativo a la cadera, ambos en unidades distintas de x.

    z_gain intenta compensar esa diferencia de escala. No hay un valor
    correcto universal: hay que estimarlo empíricamente antes de usar esto
    en serio. Por eso viene desactivada por defecto en la configuración.
    """
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    out = p.copy()
    x, z = p[..., 0], p[..., 2] * z_gain
    out[..., 0] = c * x + s * z
    out[..., 2] = (-s * x + c * z) / z_gain
    return out


def scale(p: np.ndarray, sx: float, sy: float, sz: float) -> np.ndarray:
    """Escalado por eje.

    Isotrópico (sx == sy) modela distancia a la cámara o tamaño corporal.
    Anisotrópico modela, de forma cruda, diferencias de contextura.
    """
    out = p.copy()
    out[..., 0] *= sx
    out[..., 1] *= sy
    out[..., 2] *= sz
    return out


def translate_z(p: np.ndarray, dz: float) -> np.ndarray:
    """Desplazamiento en profundidad.

    x,y no se trasladan porque el centrado del contrato los anula: sería
    un no-op disfrazado de aumentación.
    """
    out = p.copy()
    out[..., 2] += dz
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

    sigma debe MEDIRSE, no inventarse: filmar una mano quieta, extraer
    landmarks y calcular la desviación estándar por coordenada. Un sigma
    arbitrario o no hace nada o destruye la señal.

    Acepta un escalar o un array (3,) con un sigma por eje.
    """
    sigma = np.asarray(sigma, dtype=p.dtype)
    if sigma.ndim == 0:
        noise = rng.normal(0.0, float(sigma), size=p.shape)
    else:
        noise = rng.normal(0.0, 1.0, size=p.shape) * sigma.reshape(1, 1, 3)
    return p + noise.astype(p.dtype)


def drop_hand(
    p: np.ndarray,
    which: str,
    start: int,
    length: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Simula la pérdida de una mano durante una ventana de cuadros.

    En LSA64 casi nunca pasa: hay trípode, pared blanca y guantes
    fluorescentes. En uso real pasa todo el tiempo.

    IMPORTANTE: `fill` debe coincidir con lo que emite tu extractor cuando
    no detecta una mano. Si tu pipeline usa NaN, o repite el último cuadro
    válido, o interpola, hay que replicar ESE comportamiento. Rellenar con
    ceros cuando el extractor hace otra cosa le enseña al modelo una señal
    de ausencia que nunca va a ver en producción.
    """
    idx = IDX_LEFT_HAND if which == "left" else IDX_RIGHT_HAND
    out = p.copy()
    end = min(start + length, p.shape[0])
    out[start:end, idx, :] = fill
    return out


# ---------------------------------------------------------------------------
# Transformaciones temporales
# ---------------------------------------------------------------------------

def _resample(p: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Muestrea la secuencia en posiciones fraccionarias, con interpolación
    lineal entre cuadros vecinos."""
    T = p.shape[0]
    pos = np.clip(positions, 0.0, T - 1.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(p.dtype)[:, None, None]
    return (1.0 - w) * p[lo] + w * p[hi]


def time_warp(
    p: np.ndarray,
    n_out: int,
    strength: float,
    n_control: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deformación temporal monótona no uniforme.

    Modela que la misma seña se ejecuta con aceleraciones desiguales: una
    parte más lenta, otra más rápida. Probablemente la transformación de
    mayor valor, porque es la variación más grande entre personas reales y
    la que LSA64 tiene más subrepresentada (los diez sujetos imitaron el
    mismo video de referencia).

    Se construye una función de reparametrización monótona a partir de
    incrementos aleatorios positivos, se normaliza a [0, T-1] y se remuestrea.
    La monotonía es obligatoria: garantiza que no se invierte el orden de los
    cuadros, que cambiaría la seña.
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

    Modela que en producción no hay un video segmentado prolijamente: hay
    una ventana deslizante que cae donde cae.
    """
    T = p.shape[0]
    a = rng.uniform(0.0, max_frac) * (T - 1)
    b = (1.0 - rng.uniform(0.0, max_frac)) * (T - 1)
    if b - a < 2.0:
        a, b = 0.0, float(T - 1)
    return _resample(p, np.linspace(a, b, n_out))


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """Magnitudes conservadoras por defecto.

    Los valores de rotación y escala siguen el orden de magnitud reportado
    en la literatura de referencia del LIDI (±5°, ±5%). Subirlos sin medir
    es la forma más común de degradar el entrenamiento en silencio.
    """

    n_frames: int = 32                  # largo de salida; alinear con el pipeline

    # Espaciales
    p_rotate: float = 0.8
    rotate_deg: float = 5.0

    p_scale: float = 0.8
    scale_range: float = 0.05           # ±5%
    anisotropy: float = 0.02            # desviación extra por eje

    p_translate_z: float = 0.3
    translate_z_range: float = 0.02

    p_yaw: float = 0.0                  # DESACTIVADO: ver rotate_yaw()
    yaw_deg: float = 8.0
    yaw_z_gain: float = 1.0

    p_mirror: float = 0.5

    p_jitter: float = 0.7
    # Placeholder: MEDIR sobre una mano quieta antes de confiar en esto.
    jitter_sigma: tuple = (0.0015, 0.0015, 0.003)

    p_drop_hand: float = 0.15
    drop_len_frac: tuple = (0.1, 0.3)
    drop_fill: float = 0.0

    # Temporales
    p_time_warp: float = 0.7
    warp_strength: float = 0.25
    warp_control_points: int = 5

    p_temporal_crop: float = 0.5
    crop_max_frac: float = 0.1

    # Seguridad
    enforce_contract: bool = True


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

        # --- Temporales primero: cambian el largo de la secuencia ---
        if rng.random() < cfg.p_temporal_crop:
            p = temporal_crop(p, cfg.n_frames, cfg.crop_max_frac, rng)
        if rng.random() < cfg.p_time_warp:
            p = time_warp(p, cfg.n_frames, cfg.warp_strength,
                          cfg.warp_control_points, rng)
        if p.shape[0] != cfg.n_frames:
            p = _resample(p, np.linspace(0.0, p.shape[0] - 1.0, cfg.n_frames))

        # --- Espejado: antes de las rotaciones, para no componer signos ---
        if rng.random() < cfg.p_mirror:
            p = mirror(p)

        # --- Espaciales ---
        if rng.random() < cfg.p_scale:
            base = 1.0 + rng.uniform(-cfg.scale_range, cfg.scale_range)
            sx = base + rng.uniform(-cfg.anisotropy, cfg.anisotropy)
            sy = base + rng.uniform(-cfg.anisotropy, cfg.anisotropy)
            sz = base
            p = scale(p, sx, sy, sz)

        if cfg.p_yaw > 0.0 and rng.random() < cfg.p_yaw:
            p = rotate_yaw(p, rng.uniform(-cfg.yaw_deg, cfg.yaw_deg),
                           cfg.yaw_z_gain)

        if rng.random() < cfg.p_rotate:
            p = rotate_in_plane(p, rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))

        if rng.random() < cfg.p_translate_z:
            p = translate_z(p, rng.uniform(-cfg.translate_z_range,
                                           cfg.translate_z_range))

        # --- Degradaciones del extractor ---
        if rng.random() < cfg.p_drop_hand:
            which = "left" if rng.random() < 0.5 else "right"
            lo, hi = cfg.drop_len_frac
            length = max(1, int(rng.uniform(lo, hi) * p.shape[0]))
            start = int(rng.integers(0, max(1, p.shape[0] - length + 1)))
            p = drop_hand(p, which, start, length, cfg.drop_fill)

        if rng.random() < cfg.p_jitter:
            p = jitter(p, np.asarray(cfg.jitter_sigma, dtype=np.float32), rng)

        # --- Restablecer el contrato ---
        p = recenter(p)

        out = to_flat(p).astype(np.float32)
        if cfg.enforce_contract:
            assert_contract(out)
        return out


# ---------------------------------------------------------------------------
# Verificación del contrato
# ---------------------------------------------------------------------------

def assert_contract(x: np.ndarray, tol: float = 1e-4) -> None:
    """Verifica que la salida sigue cumpliendo el contrato de 201 coordenadas.

    Pensada para correr en CI junto con las pruebas existentes del contrato.
    """
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
            f"Centrado roto: |punto medio de hombros| = {mid.max():.6f} > {tol}"
        )


def _self_test() -> None:
    """Pruebas mínimas. Ejecutar con: python augment.py"""
    rng = np.random.default_rng(0)

    # La permutación de espejado debe ser una involución.
    assert np.array_equal(MIRROR_PERMUTATION[MIRROR_PERMUTATION],
                          np.arange(N_POINTS)), "El espejado no es involutivo"

    # Los bloques de mano deben intercambiarse completos.
    assert np.array_equal(MIRROR_PERMUTATION[IDX_LEFT_HAND],
                          np.arange(N_HAND) + N_HAND)

    # Muestra sintética con hombros separados y centrados.
    T = 48
    p = rng.normal(0.0, 0.1, size=(T, N_POINTS, 3)).astype(np.float32)
    p[:, POSE_OFFSET + POSE_LEFT_SHOULDER, :2] = [0.2, 0.0]
    p[:, POSE_OFFSET + POSE_RIGHT_SHOULDER, :2] = [-0.2, 0.0]
    p = recenter(p)
    x = to_flat(p)
    assert_contract(x)

    # Espejar dos veces devuelve el original.
    assert np.allclose(to_flat(mirror(mirror(p))), x, atol=1e-6), \
        "Doble espejado no es identidad"

    # Rotar 0 grados es identidad.
    assert np.allclose(rotate_in_plane(p, 0.0), p)

    # El aumentador respeta forma y contrato, y es determinista por semilla.
    aug = Augmenter(AugmentConfig(n_frames=32))
    a = aug(x, np.random.default_rng(7))
    b = aug(x, np.random.default_rng(7))
    assert a.shape == (32, N_COORDS), f"Forma inesperada: {a.shape}"
    assert np.array_equal(a, b), "No es determinista con la misma semilla"

    c = aug(x, np.random.default_rng(8))
    assert not np.array_equal(a, c), "Semillas distintas dan el mismo resultado"

    # Lote largo: ninguna combinación de transformaciones rompe el contrato.
    r = np.random.default_rng(123)
    for _ in range(500):
        assert_contract(aug(x, r))

    # La deformación temporal debe ser monótona (no invierte cuadros).
    ramp = np.zeros((T, N_POINTS, 3), dtype=np.float32)
    ramp[:, :, 0] = np.linspace(0.0, 1.0, T)[:, None]
    w = time_warp(ramp, 32, 0.4, 5, np.random.default_rng(3))[:, 0, 0]
    assert np.all(np.diff(w) >= -1e-6), "La deformación temporal no es monótona"

    print("Todas las pruebas pasaron.")


if __name__ == "__main__":
    _self_test()