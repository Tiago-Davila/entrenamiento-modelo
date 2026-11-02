"""Verifica si SWL-LSE y Google ISLR se pueden llevar al contrato v3 de Eva.

Por dataset estima la relación de aspecto H/W de cada secuencia, detecta
espejado y lateralidad de manos, cuenta faltantes y escribe un resumen JSON.
Lo que no sale de los datos no se afirma: ver VERIFICACION_DATASETS.md.

Los .pkl de SWL-LSE se leen con un Unpickler restringido: solo se resuelven
las clases de la lista blanca, y las de mediapipe se reemplazan por un stub
inerte que guarda el estado sin importar la librería.

Uso (desde la raíz del repo):
    venv/Scripts/python.exe -I multilengua/verificar_datasets.py swl \
        --pkl data/multilengua/swl_lse/pkl \
        --metadatos data/multilengua/swl_lse \
        --salida data/multilengua/swl_lse/resumen_swl.json \
        --revision data/multilengua/swl_lse/revision
    venv/Scripts/python.exe -I multilengua/verificar_datasets.py auditar \
        --pkl data/multilengua/swl_lse/pkl
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "lsa-94"))
from eva_contract import HAND_LANDMARKS, POSE_SOURCE_INDICES  # noqa: E402

POSE_PUNTOS = 33
HOMBRO_IZQ, HOMBRO_DER = 11, 12
MUNECA_IZQ, MUNECA_DER = 15, 16
MIN_CUADROS = 10
GRILLA_R = np.linspace(0.3, 3.5, 3201)  # paso 0.001

CANDIDATAS = {
    "16:9 horizontal (H/W=0.5625)": 9 / 16,
    "9:16 vertical (H/W=1.7778)": 16 / 9,
    "4:3 horizontal (H/W=0.75)": 3 / 4,
    "3:4 vertical (H/W=1.3333)": 4 / 3,
    "1:1 (H/W=1)": 1.0,
}

# Las 21 aristas de la mano de MediaPipe más la palma 0-17: huesos rígidos.
ARISTAS_MANO = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
# Huesos rígidos del tronco y brazos de Pose (control cruzado del estimador).
ARISTAS_TRONCO = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
)

# Clases que el relevamiento de los .pkl de SWL-LSE mostró que se piden.
# Ver VERIFICACION_DATASETS.md, sección "Auditoría de clases en los .pkl".
MEDIAPIPE_PERMITIDAS = frozenset({
    ("mediapipe.tasks.python.components.containers.landmark", "NormalizedLandmark"),
    ("mediapipe.tasks.python.components.containers.landmark", "Landmark"),
    ("mediapipe.tasks.python.components.containers.category", "Category"),
    ("mediapipe.tasks.python.vision.hand_landmarker", "HandLandmarkerResult"),
    ("mediapipe.tasks.python.vision.pose_landmarker", "PoseLandmarkerResult"),
    ("mediapipe.framework.formats.landmark_pb2", "NormalizedLandmarkList"),
    ("mediapipe.framework.formats.landmark_pb2", "LandmarkList"),
})
# Lista blanca de la especificación para numpy y OrderedDict (no aparecen en la muestra).
NUMPY_Y_BASE_PERMITIDAS = frozenset({
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("collections", "OrderedDict"),
})


# --------------------------------------------------------------------------
# Carga segura de .pkl
# --------------------------------------------------------------------------

class _Objeto:
    """Stub inerte de una clase de mediapipe: guarda el estado, no ejecuta nada."""

    args = ()
    estado = None

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, estado):
        self.estado = estado


class _DesempaquetadorSeguro(pickle.Unpickler):
    """Unpickler que solo resuelve clases de la lista blanca."""

    def find_class(self, module, name):
        if (module, name) in MEDIAPIPE_PERMITIDAS:
            return _Objeto
        if (module, name) in NUMPY_Y_BASE_PERMITIDAS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"clase no permitida en el pickle: {module}.{name}")


class _AuditorDeClases(pickle.Unpickler):
    """Registra las clases que pide el pickle y responde con stubs inertes."""

    def __init__(self, archivo):
        super().__init__(archivo)
        self.pedidas: set[tuple[str, str]] = set()

    def find_class(self, module, name):
        self.pedidas.add((module, name))
        return _Objeto


def cargar_pkl(origen):
    """Carga un .pkl con la lista blanca. origen: ruta o archivo binario abierto (p. ej. de un zip).

    UnpicklingError si el pickle pide una clase fuera de la lista blanca.
    """
    if isinstance(origen, (str, Path)):
        with open(origen, "rb") as f:
            return _DesempaquetadorSeguro(f).load()
    return _DesempaquetadorSeguro(origen).load()


def clases_pedidas(ruta: Path) -> set[tuple[str, str]]:
    """Conjunto de (módulo, clase) que el pickle pide, sin ejecutar nada."""
    with open(ruta, "rb") as f:
        auditor = _AuditorDeClases(f)
        auditor.load()
    return auditor.pedidas


# --------------------------------------------------------------------------
# Protobuf mínimo (NormalizedLandmarkList serializada en holistic_legacy)
# --------------------------------------------------------------------------

def _varint(datos: bytes, i: int) -> tuple[int, int]:
    valor, desplazamiento = 0, 0
    while True:
        if i >= len(datos):
            raise ValueError("varint truncado")
        byte = datos[i]
        i += 1
        valor |= (byte & 0x7F) << desplazamiento
        if not byte & 0x80:
            return valor, i
        desplazamiento += 7


def _campos_proto(datos: bytes):
    """Recorre los campos de un mensaje protobuf: (número, tipo de cable, valor)."""
    i = 0
    while i < len(datos):
        clave, i = _varint(datos, i)
        numero, tipo = clave >> 3, clave & 7
        if tipo == 0:
            valor, i = _varint(datos, i)
        elif tipo == 1:
            valor, i = datos[i:i + 8], i + 8
        elif tipo == 2:
            longitud, i = _varint(datos, i)
            valor, i = datos[i:i + longitud], i + longitud
        elif tipo == 5:
            valor, i = datos[i:i + 4], i + 4
        else:
            raise ValueError(f"tipo de cable protobuf no soportado: {tipo}")
        if i > len(datos):
            raise ValueError("mensaje protobuf truncado")
        yield numero, tipo, valor


def decodificar_landmarks(datos: bytes) -> np.ndarray:
    """NormalizedLandmarkList serializada -> (n, 3) float32 con x, y, z.

    En proto3 un campo en cero no se escribe: los ausentes valen 0.
    """
    puntos = []
    for numero, tipo, valor in _campos_proto(datos):
        if numero != 1 or tipo != 2:
            continue
        xyz = [0.0, 0.0, 0.0]
        for n2, t2, v2 in _campos_proto(valor):
            if t2 == 5 and n2 in (1, 2, 3):
                xyz[n2 - 1] = struct.unpack("<f", v2)[0]
        puntos.append(xyz)
    return np.array(puntos, dtype=np.float32).reshape(-1, 3)


# --------------------------------------------------------------------------
# Estimador de relación de aspecto H/W
# --------------------------------------------------------------------------

def curva_cv(listas: list[np.ndarray], aristas: tuple, con_z: bool = False) -> np.ndarray | None:
    """CV medio de los huesos rígidos para cada r candidata de GRILLA_R.

    Cada entrada es (T, n, 2) con x, y normalizados por el cuadro (o (T, n, 3)
    con z en unidades de W si con_z); NaN = ausente. Para r, la longitud de
    cada hueso es sqrt(dx^2 + (r*dy)^2 [+ dz^2]), en unidades de W. Si r es
    correcta, las longitudes no varían en el tiempo.
    """
    usados = sorted({k for par in aristas for k in par})
    a = np.array([p for p, _ in aristas])
    b = np.array([q for _, q in aristas])
    cvs = []
    for pts in listas:
        valido = np.all(np.isfinite(pts[:, usados, :]), axis=(1, 2))
        if valido.sum() < MIN_CUADROS:
            continue
        p = pts[valido].astype(np.float64)
        d = p[:, a, :] - p[:, b, :]                       # (T, E, 2 o 3)
        cuadrado = d[None, :, :, 0] ** 2 + (GRILLA_R[:, None, None] * d[None, :, :, 1]) ** 2
        if con_z:
            cuadrado = cuadrado + d[None, :, :, 2] ** 2
        longitudes = np.sqrt(cuadrado)                    # (G, T, E)
        cv = longitudes.std(axis=1) / np.maximum(longitudes.mean(axis=1), 1e-12)
        cvs.append(cv)                                    # (G, E)
    if not cvs:
        return None
    return np.concatenate(cvs, axis=1).mean(axis=1)


def estimar_r(listas: list[np.ndarray], aristas: tuple,
              con_z: bool = False) -> tuple[float | None, np.ndarray | None]:
    """r = H/W que minimiza el CV de los huesos. Devuelve (r, curva) o (None, None)."""
    curva = curva_cv(listas, aristas, con_z=con_z)
    if curva is None or np.ptp(curva) < 1e-12:   # huesos sin movimiento: r no se identifica
        return None, None
    return float(GRILLA_R[int(np.argmin(curva))]), curva


LONGITUD_HUESO_REAL = 0.0433  # media de los huesos de la mano en SWL-LSE, en unidades de W


def simular_ruido(sd_xy: float, sd_z: float, r_real: float, repeticiones: int = 12,
                  amplitud: float = 0.6, seed: int = 7,
                  longitud_hueso: float = LONGITUD_HUESO_REAL) -> dict:
    """Error de r (%) en una mano rígida sintética con ruido de landmarks.

    La mano es una nube de 21 puntos rígida que rota; se proyecta isotrópicamente
    (W=1, H=r_real) y se le suma ruido gaussiano en x/y (sd_xy) y z (sd_z).
    El tamaño del molde se calibra para que el hueso medio mida longitud_hueso,
    como en los datos reales: el ruido absoluto pesa según el tamaño de la mano.
    Devuelve el error medio y desvío para el estimador 2D y el 3D.
    """
    rng = np.random.default_rng(seed)
    # Gaussiana 3D: el hueso medio mide sqrt(2)*E[chi_3]*sigma = 2.2566*sigma.
    sigma = longitud_hueso / 2.2566
    molde = rng.normal(0.0, sigma, (HAND_LANDMARKS, 3))
    e2, e3 = [], []
    for rep in range(repeticiones):
        seq = []
        for t in range(45):
            rot = _rotacion(amplitud * np.sin(t / 7 + rep), amplitud * np.cos(t / 5),
                            amplitud * np.sin(t / 3 + 1))
            p = molde @ rot.T
            obs = np.stack([p[:, 0], p[:, 1] / r_real, p[:, 2]], axis=1)
            obs[:, :2] += rng.normal(0.0, sd_xy, (HAND_LANDMARKS, 2))
            obs[:, 2] += rng.normal(0.0, sd_z, HAND_LANDMARKS)
            seq.append(obs)
        seq = np.array(seq, dtype=np.float32)
        r2, _ = estimar_r([seq[..., :2]], ARISTAS_MANO)
        r3, _ = estimar_r([seq], ARISTAS_MANO, con_z=True)
        e2.append((r2 - r_real) / r_real * 100)
        e3.append((r3 - r_real) / r_real * 100)
    return {"err2d_media": float(np.mean(e2)), "err2d_sd": float(np.std(e2)),
            "err3d_media": float(np.mean(e3)), "err3d_sd": float(np.std(e3))}


def _rotacion(a: float, b: float, c: float) -> np.ndarray:
    """Matriz de rotación Rz·Ry·Rx con ángulos a, b, c en radianes."""
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _cv_en(curva: np.ndarray, r: float) -> float:
    return float(curva[int(np.argmin(np.abs(GRILLA_R - r)))])


# --------------------------------------------------------------------------
# SWL-LSE: extracción desde holistic_legacy y pose/hands
# --------------------------------------------------------------------------

def _serializado(valor):
    """Bytes protobuf de un campo de holistic_legacy, o None si no hay detección."""
    if valor is None:
        return None
    return valor.estado["serialized"]


def secuencia_holistic(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pose 33, mano izquierda 21, mano derecha 21) por cuadro, NaN si no hay detección."""
    T = len(frames)
    pose = np.full((T, POSE_PUNTOS, 3), np.nan, dtype=np.float32)
    izq = np.full((T, HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
    der = np.full((T, HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
    for t, fr in enumerate(frames):
        h = fr["holistic_legacy"]
        for destino, clave, n in ((pose, "pose_landmarks", POSE_PUNTOS),
                                  (izq, "left_hand_landmarks", HAND_LANDMARKS),
                                  (der, "right_hand_landmarks", HAND_LANDMARKS)):
            datos = _serializado(h.get(clave))
            if datos is None:
                continue
            pts = decodificar_landmarks(datos)
            if len(pts) == 0:
                continue
            if len(pts) != n:
                raise ValueError(f"{clave} con {len(pts)} puntos, se esperaban {n}")
            destino[t] = pts
    return pose, izq, der


def pose_de_clave_pose(frames: list[dict]) -> np.ndarray:
    """Pose de la clave 'pose' (PoseLandmarkerResult), (T, 33, 3), NaN si falta."""
    T = len(frames)
    out = np.full((T, POSE_PUNTOS, 3), np.nan, dtype=np.float32)
    for t, fr in enumerate(frames):
        personas = fr["pose"].estado.get("pose_landmarks") or []
        if not personas:
            continue
        pts = personas[0]
        if len(pts) == POSE_PUNTOS:
            out[t] = [[p.estado.get("x", 0.0), p.estado.get("y", 0.0), p.estado.get("z", 0.0)]
                      for p in pts]
    return out


def etiquetas_por_cuadro(frames: list[dict]) -> list[list[tuple[str, np.ndarray]]]:
    """Por cuadro, de la clave 'hands': (etiqueta de handedness, muñeca x,y) de cada mano."""
    salida = []
    for fr in frames:
        est = fr["hands"].estado or {}
        manos = est.get("hand_landmarks") or []
        lados = est.get("handedness") or []
        del_cuadro = []
        for i, mano in enumerate(manos):
            if not mano or i >= len(lados) or not lados[i]:
                continue
            etiqueta = lados[i][0].estado.get("category_name")
            muneca = np.array([mano[0].estado.get("x", np.nan), mano[0].estado.get("y", np.nan)])
            del_cuadro.append((etiqueta, muneca))
        salida.append(del_cuadro)
    return salida


def _normas(vec: np.ndarray) -> np.ndarray:
    return np.sqrt((vec ** 2).sum(axis=-1))


def analizar_secuencia(frames: list[dict]) -> dict:
    """Análisis de una secuencia del .pkl de SWL-LSE (ver analizar_arrays)."""
    pose, izq, der = secuencia_holistic(frames)
    return analizar_arrays(pose, izq, der,
                           pose_clave=pose_de_clave_pose(frames),
                           etiquetas=etiquetas_por_cuadro(frames))


def analizar_arrays(pose: np.ndarray, izq: np.ndarray, der: np.ndarray,
                    pose_clave: np.ndarray | None = None,
                    etiquetas: list | None = None) -> dict:
    """Relación de aspecto, espejado, lateralidad, movimiento y faltantes de una secuencia.

    pose (T,33,3), izq y der (T,21,3) en coordenadas normalizadas, NaN si ausente.
    pose_clave y etiquetas son opcionales (fuentes de control del .pkl).
    """
    T = len(pose)

    r_manos, curva_manos = estimar_r([izq[..., :2], der[..., :2]], ARISTAS_MANO)
    r_tronco, curva_tronco = estimar_r([pose[..., :2]], ARISTAS_TRONCO)
    if r_manos is not None:
        r, fuente = r_manos, "manos"
    elif r_tronco is not None:
        r, fuente = r_tronco, "tronco"
    else:
        r, fuente = None, "sin estimación"

    res = {
        "cuadros": T,
        "r_manos": r_manos,
        "r_tronco": r_tronco,
        "r_usada": r,
        "fuente_r": fuente,
    }
    if curva_manos is not None:
        res["cv_manos_min"] = float(curva_manos.min())
        res["cv_manos_en_candidatas"] = {k: _cv_en(curva_manos, v) for k, v in CANDIDATAS.items()}
    if curva_tronco is not None:
        res["cv_tronco_min"] = float(curva_tronco.min())

    # Espejado: en un cuadro no espejado el hombro izquierdo (11) queda a la derecha de la imagen.
    ambos_hombros = np.isfinite(pose[:, HOMBRO_IZQ, 0]) & np.isfinite(pose[:, HOMBRO_DER, 0])
    res["espejado"] = {
        "cuadros_con_hombros": int(ambos_hombros.sum()),
        "x11_mayor_que_x12": int((pose[ambos_hombros, HOMBRO_IZQ, 0]
                                  > pose[ambos_hombros, HOMBRO_DER, 0]).sum()),
    }

    if r is not None:
        res["lateralidad"] = _lateralidad(izq, der, pose, r)
        if etiquetas is not None:
            res["lateralidad_hands"] = _lateralidad_hands(etiquetas, pose, r)
        res["movimiento_medio"] = {
            "izquierda": _movimiento(izq, r),
            "derecha": _movimiento(der, r),
        }

    pres_izq = np.all(np.isfinite(izq), axis=(1, 2))
    pres_der = np.all(np.isfinite(der), axis=(1, 2))
    pres_pose = np.all(np.isfinite(pose[:, :, :2]), axis=(1, 2))
    res["faltantes"] = {
        "sin_mano_izquierda": int((~pres_izq).sum()),
        "sin_mano_derecha": int((~pres_der).sum()),
        "sin_ninguna_mano": int((~pres_izq & ~pres_der).sum()),
        "sin_pose": int((~pres_pose).sum()),
        "sin_hombros": int((~ambos_hombros).sum()),
    }
    # Control: las dos fuentes de pose (holistic_legacy y PoseLandmarker) deben coincidir.
    if pose_clave is not None:
        comunes = np.all(np.isfinite(pose[:, POSE_SOURCE_INDICES, :2]), axis=(1, 2)) & \
            np.all(np.isfinite(pose_clave[:, POSE_SOURCE_INDICES, :2]), axis=(1, 2))
        if comunes.any():
            dif = np.abs(pose[comunes][:, POSE_SOURCE_INDICES, :2]
                         - pose_clave[comunes][:, POSE_SOURCE_INDICES, :2])
            res["pose_dos_fuentes_dif_media"] = float(dif.mean())
    return res


def _lateralidad(izq: np.ndarray, der: np.ndarray, pose: np.ndarray, r: float) -> dict:
    """Muñeca de cada mano vs muñecas de Pose: la izquierda debe estar más cerca de 15."""
    p15 = pose[:, MUNECA_IZQ, :2] * [1.0, r]
    p16 = pose[:, MUNECA_DER, :2] * [1.0, r]
    base = np.all(np.isfinite(p15), axis=1) & np.all(np.isfinite(p16), axis=1)
    out = {}
    for nombre, mano in (("izquierda", izq), ("derecha", der)):
        m0 = mano[:, 0, :2] * [1.0, r]
        ok = base & np.all(np.isfinite(m0), axis=1)
        d15 = _normas(m0 - p15)
        d16 = _normas(m0 - p16)
        coincide = (d15 < d16) if nombre == "izquierda" else (d16 < d15)
        out[nombre] = {"cuadros": int(ok.sum()), "coinciden": int(coincide[ok].sum())}
    return out


def _lateralidad_hands(etiquetas: list, pose: np.ndarray, r: float) -> dict:
    """Mismo control con la etiqueta de handedness de la clave 'hands'."""
    conteo = {"Left": [0, 0], "Right": [0, 0]}
    for t, del_cuadro in enumerate(etiquetas):
        if not np.all(np.isfinite(pose[t, [MUNECA_IZQ, MUNECA_DER], :2])):
            continue
        q15 = pose[t, MUNECA_IZQ, :2] * np.array([1.0, r])
        q16 = pose[t, MUNECA_DER, :2] * np.array([1.0, r])
        for etiqueta, muneca in del_cuadro:
            if etiqueta not in conteo or not np.all(np.isfinite(muneca)):
                continue
            m = muneca * np.array([1.0, r])
            d15 = float(_normas(m - q15))
            d16 = float(_normas(m - q16))
            cercano = 15 if d15 < d16 else 16
            conteo[etiqueta][0] += 1
            conteo[etiqueta][1] += int(cercano == (MUNECA_IZQ if etiqueta == "Left" else MUNECA_DER))
    return {
        "Left": {"manos": conteo["Left"][0], "cerca_de_pose15": conteo["Left"][1]},
        "Right": {"manos": conteo["Right"][0], "cerca_de_pose16": conteo["Right"][1]},
    }


def _movimiento(mano: np.ndarray, r: float) -> float | None:
    """Desplazamiento medio por cuadro de los 21 puntos, en unidades de W."""
    pts = mano[..., :2] * np.array([1.0, r])
    ok = np.all(np.isfinite(pts), axis=(1, 2))
    par = ok[1:] & ok[:-1]
    if not par.any():
        return None
    d = _normas(pts[1:][par] - pts[:-1][par])
    return float(d.mean())


def _nombres_clase(carpeta_meta: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Mapa id de secuencia -> clase (de ANNOTATIONS.zip) y clase -> glosa."""
    ids: dict[str, int] = {}
    with zipfile.ZipFile(carpeta_meta / "ANNOTATIONS.zip") as z:
        for nombre in ("ANNOTATIONS/train_labels.csv", "ANNOTATIONS/val_labels.csv",
                       "ANNOTATIONS/test_labels.csv"):
            for fila in csv.reader(z.read(nombre).decode("utf-8").splitlines()):
                if len(fila) == 2:
                    ids[fila[0]] = int(fila[1])
    glosas: dict[int, str] = {}
    with open(carpeta_meta / "videos_ref_annotations.csv", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            glosas.setdefault(int(fila["CLASS_ID"]), fila["LABEL"])
    return ids, glosas


def analizar_swl(carpeta_pkl: Path, carpeta_meta: Path, muestra: list[str] | None = None) -> dict:
    """Analiza todos los .pkl de carpeta_pkl (o la lista de ids dada)."""
    ids, glosas = _nombres_clase(carpeta_meta)
    archivos = sorted(carpeta_pkl.glob("*.pkl"))
    if muestra is not None:
        archivos = [carpeta_pkl / f"{i}.pkl" for i in muestra]
    secuencias = []
    for ruta in archivos:
        frames = cargar_pkl(ruta)
        info = analizar_secuencia(frames)
        sid = ruta.stem
        info["id"] = sid
        info["clase"] = ids.get(sid)
        info["glosa"] = glosas.get(ids.get(sid), "?")
        secuencias.append(info)
    return {"dataset": "SWL-LSE", "secuencias": secuencias, "resumen": _resumir(secuencias)}


def _resumir(secs: list[dict]) -> dict:
    rs = [s["r_usada"] for s in secs if s.get("r_usada") is not None]
    rs_manos = [s["r_manos"] for s in secs if s.get("r_manos") is not None]
    rs_tronco = [s["r_tronco"] for s in secs if s.get("r_tronco") is not None]
    esp_tot = sum(s["espejado"]["cuadros_con_hombros"] for s in secs)
    esp_x = sum(s["espejado"]["x11_mayor_que_x12"] for s in secs)
    lat = {"izquierda": [0, 0], "derecha": [0, 0]}
    lat_hands = {"Left": [0, 0], "Right": [0, 0]}
    mov_mas_izq = mov_mas_der = 0
    faltantes = {k: 0 for k in ("sin_mano_izquierda", "sin_mano_derecha", "sin_ninguna_mano",
                                "sin_pose", "sin_hombros")}
    cuadros = 0
    sin_hombros_en_secuencia = 0
    for s in secs:
        cuadros += s["cuadros"]
        for k in faltantes:
            faltantes[k] += s["faltantes"][k]
        if s["faltantes"]["sin_hombros"] == s["cuadros"]:
            sin_hombros_en_secuencia += 1
        if "lateralidad" in s:
            for mano in lat:
                lat[mano][0] += s["lateralidad"][mano]["cuadros"]
                lat[mano][1] += s["lateralidad"][mano]["coinciden"]
        if "lateralidad_hands" in s:
            for mano in lat_hands:
                lat_hands[mano][0] += s["lateralidad_hands"][mano]["manos"]
                key = "cerca_de_pose15" if mano == "Left" else "cerca_de_pose16"
                lat_hands[mano][1] += s["lateralidad_hands"][mano][key]
        if "movimiento_medio" in s:
            mi, md = s["movimiento_medio"]["izquierda"], s["movimiento_medio"]["derecha"]
            if mi is not None and md is not None:
                mov_mas_izq += mi > md
                mov_mas_der += md > mi
    cv_candidatas = {k: float(np.median([s["cv_manos_en_candidatas"][k] for s in secs
                                         if "cv_manos_en_candidatas" in s]))
                     for k in CANDIDATAS} if any("cv_manos_en_candidatas" in s for s in secs) else {}
    return {
        "secuencias": len(secs),
        "cuadros": cuadros,
        "r_manos": _distribucion(rs_manos),
        "r_tronco": _distribucion(rs_tronco),
        "r_usada": _distribucion(rs),
        "cv_mediana_en_candidatas": cv_candidatas,
        "espejado": {"cuadros_con_hombros": esp_tot, "x11_mayor_que_x12": esp_x,
                     "proporcion": esp_x / esp_tot if esp_tot else None},
        "lateralidad_pose": {m: {"cuadros": v[0], "coinciden": v[1],
                                 "proporcion": v[1] / v[0] if v[0] else None}
                             for m, v in lat.items()},
        "lateralidad_handedness": {m: {"manos": v[0], "cerca_de_pose": v[1],
                                       "proporcion": v[1] / v[0] if v[0] else None}
                                   for m, v in lat_hands.items()},
        "movimiento": {"secuencias_mas_mueve_izquierda": mov_mas_izq,
                       "secuencias_mas_mueve_derecha": mov_mas_der},
        "faltantes": {**faltantes, "cuadros": cuadros,
                      "secuencias_sin_hombros_en_ningun_cuadro": sin_hombros_en_secuencia},
    }


def _distribucion(v: list[float]) -> dict | None:
    if not v:
        return None
    a = np.array(v)
    return {"n": len(v), "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "mediana": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "max": float(a.max())}


def mas_cercano_candidata(r: float | None) -> str | None:
    """Nombre de la relación estándar más próxima en escala logarítmica."""
    if r is None:
        return None
    return min(CANDIDATAS, key=lambda k: abs(np.log(r / CANDIDATAS[k])))


def ejemplos(secs: list[dict], n: int = 5) -> list[dict]:
    """n secuencias repartidas en el orden de la muestra, para revisión a mano."""
    if not secs:
        return []
    idx = np.linspace(0, len(secs) - 1, n).round().astype(int)
    return [secs[i] for i in idx]


def dibujar_revision(carpeta_pkl: Path, secs: list[dict], carpeta_salida: Path) -> list[Path]:
    """Dibuja el cuadro central de cada ejemplo con la mano en (x, y*r) para revisarlo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    rutas = []
    for s in ejemplos(secs):
        frames = cargar_pkl(carpeta_pkl / f"{s['id']}.pkl")
        pose, izq, der = secuencia_holistic(frames)
        r = s.get("r_usada") or 1.0
        t = len(frames) // 2
        fig, ax = plt.subplots(figsize=(5, 6))
        p = pose[t, :, :2] * [1.0, r]
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24)):
            if np.all(np.isfinite(p[[a, b]])):
                ax.plot(p[[a, b], 0], p[[a, b], 1], color="gray", lw=1)
        for nombre, mano, color in (("izq", izq[t], "red"), ("der", der[t], "blue")):
            if np.all(np.isfinite(mano)):
                m = mano[:, :2] * [1.0, r]
                for a, b in ARISTAS_MANO:
                    ax.plot(m[[a, b], 0], m[[a, b], 1], color=color, lw=1.5)
                ax.scatter(m[0, 0], m[0, 1], color=color, s=25, label=f"mano {nombre} (muñeca)")
        for i, nombre in ((11, "hombro izq (11)"), (12, "hombro der (12)"),
                          (15, "muñeca izq (15)"), (16, "muñeca der (16)")):
            if np.all(np.isfinite(p[i])):
                ax.scatter(p[i, 0], p[i, 1], color="black", s=20)
                ax.annotate(nombre, p[i], fontsize=7)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"{s['id']} · {s['glosa']} · r={r:.3f} · cuadro {t}", fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        ruta = carpeta_salida / f"{s['id']}.png"
        fig.savefig(ruta, dpi=110)
        plt.close(fig)
        rutas.append(ruta)
    return rutas


def extraer_mediapipe(ruta_video: Path, modelo: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Corre HolisticLandmarker (modo imagen) sobre un video.

    Devuelve pose, mano izquierda, mano derecha (NaN si no hay detección) y (W, H)
    medidos con OpenCV. Es el mismo task que usa la app.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opciones = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(modelo)),
        running_mode=vision.RunningMode.IMAGE,
    )
    cuadros = []
    cap = cv2.VideoCapture(str(ruta_video))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with vision.HolisticLandmarker.create_from_options(opciones) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            cuadros.append(lm.detect(img))
    cap.release()

    def arr(lista, n):
        if not lista:
            return np.full((n, 3), np.nan, dtype=np.float32)
        return np.array([[p.x, p.y, p.z] for p in lista], dtype=np.float32)

    pose = np.stack([arr(r.pose_landmarks, POSE_PUNTOS) for r in cuadros])
    izq = np.stack([arr(r.left_hand_landmarks, HAND_LANDMARKS) for r in cuadros])
    der = np.stack([arr(r.right_hand_landmarks, HAND_LANDMARKS) for r in cuadros])
    return pose, izq, der, (ancho, alto)


def validar_videos(carpeta_videos: Path, modelo: Path) -> dict:
    """Corre el extractor sobre videos de resolución conocida y compara r estimada vs real."""
    secuencias = []
    for ruta in sorted(carpeta_videos.glob("*.mp4")):
        pose, izq, der, (w, h) = extraer_mediapipe(ruta, modelo)
        info = analizar_arrays(pose, izq, der)
        info["id"] = ruta.stem
        info["resolucion_medida"] = [w, h]
        info["r_real"] = h / w
        if info.get("r_usada") is not None:
            info["error_r_pct"] = (info["r_usada"] - info["r_real"]) / info["r_real"] * 100
        secuencias.append(info)
    return {"dataset": "videos de referencia SWL-LSE", "secuencias": secuencias,
            "resumen": _resumir(secuencias)}


def dispersion_xy(pose: np.ndarray) -> tuple[float, float]:
    """Desvío estándar de x e y de todos los puntos de pose válidos."""
    q = pose[..., :2].reshape(-1, 2)
    q = q[np.all(np.isfinite(q), axis=1)]
    return float(q[:, 0].std()), float(q[:, 1].std())


def cociente_ky_kx(pose_pkl: np.ndarray, pose_video: np.ndarray) -> float:
    """(sy_pkl/sy_video) / (sx_pkl/sx_video): 1 si ambas coordenadas están en el mismo cuadro.

    No depende de alinear cuadros en el tiempo. Un pkl hecho sobre un cuadro de
    relación H/W distinta, frente al mismo video, daría el cociente H/W distinto de 1.
    """
    sxp, syp = dispersion_xy(pose_pkl)
    sxv, syv = dispersion_xy(pose_video)
    return (syp / syv) / (sxp / sxv)


def comparar_refs(carpeta_pkl: Path, carpeta_videos: Path, modelo: Path) -> dict:
    """Compara cada pkl de referencia con MediaPipe sobre su mp4 de resolución conocida."""
    secuencias = []
    for ruta in sorted(carpeta_pkl.glob("*.pkl")):
        video = carpeta_videos / f"{ruta.stem}.mp4"
        if not video.exists():
            continue
        frames = cargar_pkl(ruta)
        pose_pkl, _, _ = secuencia_holistic(frames)
        pose_vid, _, _, (w, h) = extraer_mediapipe(video, modelo)
        secuencias.append({
            "id": ruta.stem,
            "T_pkl": len(frames),
            "T_mp4": len(pose_vid),
            "resolucion_mp4": [w, h],
            "cociente_ky_kx": cociente_ky_kx(pose_pkl, pose_vid),
        })
    cocientes = np.array([s["cociente_ky_kx"] for s in secuencias])
    resumen = {
        "n": len(secuencias),
        "mediana": float(np.median(cocientes)),
        "p25": float(np.percentile(cocientes, 25)),
        "p75": float(np.percentile(cocientes, 75)),
        "min": float(cocientes.min()),
        "max": float(cocientes.max()),
        "T_iguales": int(sum(s["T_pkl"] == s["T_mp4"] for s in secuencias)),
    }
    return {"secuencias": secuencias, "resumen": resumen}


def leer_islr(ruta: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parquet de Google ISLR -> pose 33, mano izquierda 21, mano derecha 21 por cuadro.

    Columnas: frame, row_id, type (face, left_hand, pose, right_hand), landmark_index,
    x, y, z. Los landmarks sin fila quedan en NaN, igual que lo no detectado.
    """
    import pandas as pd

    df = pd.read_parquet(ruta, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    cuadros = np.sort(df["frame"].unique())
    idx_cuadro = np.searchsorted(cuadros, df["frame"].to_numpy())
    tamanos = {"pose": POSE_PUNTOS, "left_hand": HAND_LANDMARKS, "right_hand": HAND_LANDMARKS}
    salida = {}
    for tipo, n in tamanos.items():
        arr = np.full((len(cuadros), n, 3), np.nan, dtype=np.float32)
        filas = df["type"].to_numpy() == tipo
        li = df["landmark_index"].to_numpy()[filas].astype(int)
        arr[idx_cuadro[filas], li, 0] = df["x"].to_numpy()[filas]
        arr[idx_cuadro[filas], li, 1] = df["y"].to_numpy()[filas]
        arr[idx_cuadro[filas], li, 2] = df["z"].to_numpy()[filas]
        salida[tipo] = arr
    return salida["pose"], salida["left_hand"], salida["right_hand"]


def ratio_torso(pose: np.ndarray, r: float) -> np.ndarray:
    """Ancho de hombros / longitud media del torso (11-23, 12-24), por cuadro, en (x, y*r).

    NaN donde falte algún punto. Depende de la relación de aspecto: es la evidencia
    independiente de que no hay video de referencia.
    """
    def dist(a, b):
        return _normas((pose[:, a, :2] - pose[:, b, :2]) * [1.0, r])

    ancho = dist(HOMBRO_IZQ, HOMBRO_DER)
    torso = 0.5 * (dist(HOMBRO_IZQ, 23) + dist(HOMBRO_DER, 24))
    with np.errstate(divide="ignore", invalid="ignore"):
        return ancho / torso


def calibrar_r_torso(pose_ref: np.ndarray, pose_obj: np.ndarray) -> tuple[float, float]:
    """r que iguala la mediana de ancho/torso de pose_obj con la de pose_ref (con r = 1).

    Devuelve (r, mediana de referencia). Supone que la proporción de cuerpo es la
    misma en ambos grupos: es una estimación poblacional, no por secuencia.
    """
    ref = ratio_torso(pose_ref, 1.0)
    objetivo = float(np.nanmedian(ref))
    mejor_r, mejor_err = None, np.inf
    for r in GRILLA_R[::5]:
        med = float(np.nanmedian(ratio_torso(pose_obj, r)))
        err = abs(med - objetivo)
        if err < mejor_err:
            mejor_r, mejor_err = float(r), err
    return mejor_r, objetivo


def analizar_islr(carpeta_parquet: Path, train_csv: Path, carpeta_pkl_ref: Path | None = None) -> dict:
    """Analiza los parquet de ISLR descargados; cada uno se asocia a su señante en train.csv.

    Si se pasa carpeta_pkl_ref, calibra la relación de aspecto poblacional de ISLR
    contra la proporción ancho/torso de los pkl de SWL-LSE de referencia (H/W = 1).
    """
    import csv as _csv

    meta = {}
    with open(train_csv, encoding="utf-8") as f:
        for fila in _csv.DictReader(f):
            meta[fila["sequence_id"]] = fila
    datos = []
    for ruta in sorted(carpeta_parquet.rglob("*.parquet")):
        fila = meta.get(ruta.stem, {})
        datos.append((ruta.stem, fila.get("participant_id"), fila.get("sign"), leer_islr(ruta)))

    secuencias = []
    for sid, part, sena, (pose, izq, der) in datos:
        info = analizar_arrays(pose, izq, der)
        info.update({"id": sid, "participante": part, "sena": sena})
        secuencias.append(info)

    resultado = {"dataset": "Google ISLR", "secuencias": secuencias}
    if carpeta_pkl_ref is not None:
        pose_ref = np.concatenate([secuencia_holistic(cargar_pkl(p))[0]
                                   for p in sorted(carpeta_pkl_ref.glob("*.pkl"))])
        pose_todo = np.concatenate([d[3][0] for d in datos])
        r_pob, mediana_ref = calibrar_r_torso(pose_ref, pose_todo)
        resultado["torso_calibracion"] = {"r_poblacional": r_pob,
                                          "mediana_ancho_torso_swl_ref_r1": mediana_ref}
        por_senante = {}
        for sid, part, sena, (pose, _, _) in datos:
            por_senante.setdefault(part, []).append(pose)
        resultado["torso_por_senante"] = {
            str(p): calibrar_r_torso(pose_ref, np.concatenate(poses))[0]
            for p, poses in por_senante.items()
        }
    resultado["resumen_global"] = _resumir(secuencias)
    resultado["resumen_por_senante"] = {
        str(p): _resumir([s for s in secuencias if s["participante"] == p])
        for p in sorted({s["participante"] for s in secuencias})
    }
    return resultado


def revisar_islr(carpeta_parquet: Path, train_csv: Path, carpeta_salida: Path,
                 n: int = 5) -> list[Path]:
    """Dibuja el cuadro central de n secuencias ISLR repartidas, para revisión a mano."""
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = {}
    with open(train_csv, encoding="utf-8") as f:
        for fila in _csv.DictReader(f):
            meta[fila["sequence_id"]] = fila
    archivos = sorted(carpeta_parquet.rglob("*.parquet"))
    idx = np.linspace(0, len(archivos) - 1, n).round().astype(int)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    rutas = []
    for i in idx:
        ruta = archivos[i]
        pose, izq, der = leer_islr(ruta)
        r_mano, _ = estimar_r([izq[..., :2], der[..., :2]], ARISTAS_MANO)
        r = r_mano or 1.0
        t = len(pose) // 2
        fig, ax = plt.subplots(figsize=(5, 6))
        p = pose[t, :, :2] * [1.0, r]
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24)):
            if np.all(np.isfinite(p[[a, b]])):
                ax.plot(p[[a, b], 0], p[[a, b], 1], color="gray", lw=1)
        for nombre, mano, color in (("izq", izq[t], "red"), ("der", der[t], "blue")):
            if np.all(np.isfinite(mano)):
                m = mano[:, :2] * [1.0, r]
                for a, b in ARISTAS_MANO:
                    ax.plot(m[[a, b], 0], m[[a, b], 1], color=color, lw=1.5)
                ax.scatter(m[0, 0], m[0, 1], color=color, s=25, label=f"mano {nombre} (muñeca)")
        for j, nombre in ((11, "hombro izq (11)"), (12, "hombro der (12)"),
                          (15, "muñeca izq (15)"), (16, "muñeca der (16)")):
            if np.all(np.isfinite(p[j])):
                ax.scatter(p[j, 0], p[j, 1], color="black", s=20)
                ax.annotate(nombre, p[j], fontsize=7)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        fila = meta.get(ruta.stem, {})
        ax.set_title(f"{ruta.stem} · {fila.get('sign', '?')} · r={r:.3f} · cuadro {t}", fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        salida = carpeta_salida / f"{ruta.stem}.png"
        fig.savefig(salida, dpi=110)
        plt.close(fig)
        rutas.append(salida)
    return rutas


def auditar(carpeta_pkl: Path) -> dict:
    """Clases pedidas por todos los .pkl de la carpeta, con cuántos archivos las piden."""
    conteo: dict[tuple[str, str], int] = {}
    for ruta in sorted(carpeta_pkl.glob("*.pkl")):
        for clase in clases_pedidas(ruta):
            conteo[clase] = conteo.get(clase, 0) + 1
    return {f"{m}.{n}": c for (m, n), c in sorted(conteo.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_swl = sub.add_parser("swl", help="Analiza los .pkl de SWL-LSE")
    p_swl.add_argument("--pkl", type=Path, required=True)
    p_swl.add_argument("--metadatos", type=Path, required=True)
    p_swl.add_argument("--salida", type=Path, required=True)
    p_swl.add_argument("--revision", type=Path, default=None)
    p_aud = sub.add_parser("auditar", help="Lista las clases que piden los .pkl")
    p_aud.add_argument("--pkl", type=Path, required=True)
    sub.add_parser("simular-ruido", help="Error del estimador en manos sintéticas con ruido")
    p_isl = sub.add_parser("islr", help="Analiza los parquet de Google ISLR")
    p_isl.add_argument("--parquet", type=Path, required=True)
    p_isl.add_argument("--train-csv", type=Path, required=True)
    p_isl.add_argument("--pkl-ref", type=Path, default=None)
    p_isl.add_argument("--salida", type=Path, required=True)
    p_isl.add_argument("--revision", type=Path, default=None)
    p_cmp = sub.add_parser("comparar-ref", help="Compara pkl de referencia con MediaPipe sobre su video")
    p_cmp.add_argument("--pkl", type=Path, required=True)
    p_cmp.add_argument("--videos", type=Path, required=True)
    p_cmp.add_argument("--modelo", type=Path, required=True)
    p_cmp.add_argument("--salida", type=Path, required=True)
    p_vid = sub.add_parser("video-ref", help="Valida el estimador con videos de resolución conocida")
    p_vid.add_argument("--videos", type=Path, required=True)
    p_vid.add_argument("--modelo", type=Path, required=True)
    p_vid.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "auditar":
        print(json.dumps(auditar(args.pkl), indent=1, ensure_ascii=False))
        return 0
    if args.cmd == "simular-ruido":
        for sd_xy, sd_z in ((0.0, 0.0), (0.005, 0.004), (0.01, 0.008)):
            print(f"ruido xy={sd_xy} z={sd_z}")
            for r_real in (1.0, 0.75, 0.5625, 1.7778, 1.3333):
                e = simular_ruido(sd_xy, sd_z, r_real)
                print(f"  r={r_real:<7} 2D err% {e['err2d_media']:+6.1f} (sd {e['err2d_sd']:4.1f})"
                      f"   3D err% {e['err3d_media']:+6.1f} (sd {e['err3d_sd']:4.1f})")
        return 0
    if args.cmd == "islr":
        resultado = analizar_islr(args.parquet, args.train_csv, args.pkl_ref)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(resultado, indent=1, ensure_ascii=False, default=float),
                               encoding="utf-8")
        print(json.dumps({k: v for k, v in resultado.items() if k != "secuencias"},
                         indent=1, ensure_ascii=False, default=float))
        if args.revision:
            revisar_islr(args.parquet, args.train_csv, args.revision)
        return 0
    if args.cmd == "comparar-ref":
        resultado = comparar_refs(args.pkl, args.videos, args.modelo)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(resultado["resumen"], indent=1, ensure_ascii=False))
        return 0
    if args.cmd == "video-ref":
        resultado = validar_videos(args.videos, args.modelo)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(resultado["resumen"], indent=1, ensure_ascii=False))
        return 0

    resultado = analizar_swl(args.pkl, args.metadatos)
    resultado["ejemplos"] = [s["id"] for s in ejemplos(resultado["secuencias"])]
    for s in resultado["secuencias"]:
        s["candidata_mas_cercana"] = mas_cercano_candidata(s.get("r_usada"))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.revision:
        dibujar_revision(args.pkl, resultado["secuencias"], args.revision)
    print(json.dumps(resultado["resumen"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
