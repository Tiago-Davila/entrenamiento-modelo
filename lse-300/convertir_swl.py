"""Convierte SWL-LSE (300 señas de salud) al contrato Eva v3 de 40 x 168.

Lee los .pkl de holistic_legacy (desde MEDIAPIPE.zip sin descomprimir, o desde
carpetas de muestra), los mapea al vector de 168 coordenadas de lsa-94, deshace
el espejado del extractor de los autores y aplica eva_contract.build_input_v3
con W = H = 480 (H/W = 1, ver LSE300_DATOS.md).

Salidas en lse-300/data/processed_v3/: X, y, subjects (sesión), split
(0 train, 1 val, 2 test, 3 referencia), reps, nombres, invalidas.json,
distribucion.json y meta.json.

Si la validación de escala (ancho de hombros / torso) de una partición numérica
se aparta más de 10% de la de las secuencias de referencia, el script aborta sin
escribir nada.

Uso (desde la raíz del repo):
    venv/Scripts/python.exe -I lse-300/convertir_swl.py
    venv/Scripts/python.exe -I lse-300/convertir_swl.py \
        --fuente data/multilengua/swl_lse/pkl --fuente data/multilengua/swl_lse/pkl_ref \
        --salida %TEMP%/muestra_v3 --solo-validar
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "lsa-94"))
sys.path.insert(0, str(RAIZ / "multilengua"))
import eva_contract as ec  # noqa: E402
import verificar_datasets as v  # noqa: E402

AQUI = Path(__file__).resolve().parent
DATOS = RAIZ / "data" / "multilengua" / "swl_lse"
ZIP_MEDIAPIPE = DATOS / "MEDIAPIPE.zip"
ZIP_ANOTACIONES = DATOS / "ANNOTATIONS.zip"
CSV_REFERENCIA = DATOS / "videos_ref_annotations.csv"
SALIDA_DEFAULT = AQUI / "data" / "processed_v3"
CATALOGO = AQUI / "catalogo_lse300.json"
X_LSA94 = RAIZ / "lsa-94" / "data" / "processed_94" / "X.npy"

MD5_MEDIAPIPE = "e128df76232e0c7dcd7a74935cd38335"  # record.json de Zenodo 13691887
TAMANO_MEDIAPIPE = 3499865422
DOI = "10.5281/zenodo.13691887"

# Videos de referencia y pkl_ref: 480x480 (verificado en #3). Las 7 700
# secuencias de señantes se asumen 480x480 (ver LSE300_DATOS.md, validación).
ANCHO_PX = 480
ALTO_PX = 480

UMBRAL_SESION_S = 3600           # 1 hora, ver LSE300_DATOS.md (histograma de saltos)
TOLERANCIA_ESCALA = 0.10         # 10%: por encima se frena la conversión

SPLIT_NOMBRES = {"train": 0, "val": 1, "test": 2}
SPLIT_REFERENCIA = 3
SPLIT_ETIQUETAS = {0: "train", 1: "val", 2: "test", 3: "referencia"}

# Pose local (11..24) con los pares izquierda/derecha intercambiados:
# 11<->12, 13<->14, ..., 23<->24. Posiciones locales = índice - 11.
PERMUTACION_POSE = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12)
CADERA_IZQ_LOCAL = 23 - 11
CADERA_DER_LOCAL = 24 - 11


# --------------------------------------------------------------------------
# Mapeo al vector de 168 y corrección de espejado
# --------------------------------------------------------------------------

def a_vector_168(pose: np.ndarray, izq: np.ndarray, der: np.ndarray,
                 corregir: bool = True) -> np.ndarray:
    """(T,33,3) pose, (T,21,3) mano izq., (T,21,3) mano der. -> (T,168) float32.

    Ausente (NaN) -> ceros. Con corregir=True se deshace el espejado del
    extractor de los autores: x -> 1-x, bloques de manos intercambiados y pares
    de pose izquierda/derecha intercambiados. Con corregir=False queda el
    vector tal como lo entregó holistic_legacy (sólo para medir el antes).
    """
    T = pose.shape[0]
    tronco = pose[:, list(ec.POSE_SOURCE_INDICES)]
    if corregir:
        nueva_izq = _espejar(der)
        nueva_der = _espejar(izq)
        tronco = _espejar(tronco)[:, list(PERMUTACION_POSE)]
    else:
        nueva_izq, nueva_der = izq, der
    out = np.concatenate([
        nueva_izq.reshape(T, ec.HAND_LANDMARKS * 3),
        nueva_der.reshape(T, ec.HAND_LANDMARKS * 3),
        tronco.reshape(T, ec.POSE_LANDMARKS * 3),
    ], axis=1).astype(np.float32)
    out[np.isnan(out)] = 0.0
    return out


def _espejar(puntos: np.ndarray) -> np.ndarray:
    """Copia con x -> 1-x; los NaN (ausentes) quedan NaN."""
    out = puntos.astype(np.float32, copy=True)
    out[..., 0] = 1.0 - out[..., 0]
    return out


def entrada_v3(crudo: np.ndarray) -> np.ndarray:
    """Vector crudo (T,168) -> entrada del contrato v3 (40,168), con W = H = 480.

    ValueError si la secuencia no cumple el contrato (sin cuadros o sin hombros).
    """
    return ec.build_input_v3(crudo, ANCHO_PX, ALTO_PX)


# --------------------------------------------------------------------------
# Validación de escala: ancho de hombros / longitud del torso (píxeles, H/W = 1)
# --------------------------------------------------------------------------

def proporcion_hombro_torso(crudo: np.ndarray) -> float | None:
    """Mediana por cuadro de ancho de hombros / torso de una secuencia cruda.

    Torso = distancia del punto medio de hombros (11, 12) al de caderas (23, 24).
    None si ningún cuadro tiene los cuatro puntos.
    """
    px = ec.to_pixels(crudo, ANCHO_PX, ALTO_PX)
    base = ec.POSE_OFFSET // 3
    izq_c, der_c = base + CADERA_IZQ_LOCAL, base + CADERA_DER_LOCAL
    presentes = np.any(px != 0.0, axis=-1)
    ok = (presentes[:, ec.LEFT_SHOULDER_POINT] & presentes[:, ec.RIGHT_SHOULDER_POINT]
          & presentes[:, izq_c] & presentes[:, der_c])
    if not ok.any():
        return None
    p = px[ok, :, :2]
    hombros = np.linalg.norm(p[:, ec.LEFT_SHOULDER_POINT] - p[:, ec.RIGHT_SHOULDER_POINT], axis=-1)
    medio_h = (p[:, ec.LEFT_SHOULDER_POINT] + p[:, ec.RIGHT_SHOULDER_POINT]) / 2.0
    medio_c = (p[:, izq_c] + p[:, der_c]) / 2.0
    torso = np.linalg.norm(medio_h - medio_c, axis=-1)
    valido = torso > 1e-6
    if not valido.any():
        return None
    return float(np.median(hombros[valido] / torso[valido]))


def violaciones_escala(medianas: dict[str, float | None], referencia: float | None,
                       tolerancia: float = TOLERANCIA_ESCALA) -> list[str]:
    """Particiones numéricas cuya mediana se aparta más de `tolerancia` de la referencia."""
    if referencia is None:
        return ["sin secuencias de referencia para comparar"]
    fallas = []
    for nombre, valor in medianas.items():
        if valor is None:
            fallas.append(f"{nombre}: sin datos")
        elif abs(valor / referencia - 1.0) > tolerancia:
            fallas.append(f"{nombre}: {valor:.4f} vs referencia {referencia:.4f} "
                          f"({(valor / referencia - 1.0) * 100:+.1f}%)")
    return fallas


# --------------------------------------------------------------------------
# Sesiones (agrupación por marca de tiempo)
# --------------------------------------------------------------------------

def asignar_sesiones(ids: list[int], umbral_s: float) -> dict[int, int]:
    """id (marca de tiempo en microsegundos) -> sesión 0..S-1, cortando si el salto supera umbral_s."""
    orden = sorted(ids)
    sesion_de: dict[int, int] = {}
    actual = 0
    previo = None
    for sid in orden:
        if previo is not None and (sid - previo) / 1e6 > umbral_s:
            actual += 1
        sesion_de[sid] = actual
        previo = sid
    return sesion_de


def sesiones_cruzadas(sesion_de: dict[int, int], split_de: dict[int, int]) -> dict:
    """Cuántas sesiones tienen secuencias en más de una partición numérica (train/val/test)."""
    particiones: dict[int, set[int]] = {}
    cantidad: dict[int, int] = {}
    for sid, ses in sesion_de.items():
        s = split_de[sid]
        if s == SPLIT_REFERENCIA:
            continue
        particiones.setdefault(ses, set()).add(s)
        cantidad[ses] = cantidad.get(ses, 0) + 1
    cruzadas = {s for s, p in particiones.items() if len(p) > 1}
    return {
        "sesiones": len(particiones),
        "cruzadas": len(cruzadas),
        "secuenciasEnSesionesCruzadas": int(sum(cantidad[s] for s in cruzadas)),
        "trainTest": sum(1 for p in particiones.values() if {0, 2} <= p),
        "trainVal": sum(1 for p in particiones.values() if {0, 1} <= p),
        "valTest": sum(1 for p in particiones.values() if {1, 2} <= p),
    }


# --------------------------------------------------------------------------
# Estadísticas de manos (antes y después de corregir)
# --------------------------------------------------------------------------

def _movimiento(pts: np.ndarray) -> float | None:
    """(T,K,2) -> desplazamiento medio por cuadro de los K puntos presentes en dos cuadros seguidos."""
    presentes = np.any(pts != 0.0, axis=-1)
    par = presentes[1:] & presentes[:-1]
    if not par.any():
        return None
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    return float(d[par].mean())


def estadisticas_bloques(crudo: np.ndarray) -> dict:
    """Presencia y movimiento de cada bloque de un vector crudo (T,168), antes del muestreo."""
    T = crudo.shape[0]
    izq = crudo[:, 0:63].reshape(T, ec.HAND_LANDMARKS, 3)
    der = crudo[:, 63:126].reshape(T, ec.HAND_LANDMARKS, 3)
    tronco = crudo[:, 126:168].reshape(T, ec.POSE_LANDMARKS, 3)
    return {
        "cuadros": T,
        "cuadrosSinIzq": int((~np.any(izq != 0.0, axis=(1, 2))).sum()),
        "cuadrosSinDer": int((~np.any(der != 0.0, axis=(1, 2))).sum()),
        "movIzq": _movimiento(izq[..., :2]),
        "movDer": _movimiento(der[..., :2]),
        "movPose": _movimiento(tronco[..., :2]),
    }


def resumir_bloques(filas: list[dict]) -> dict:
    """Agrega estadisticas_bloques de muchas secuencias (proporciones por cuadro, medias por secuencia)."""
    cuadros = sum(f["cuadros"] for f in filas)
    mov = {k: [f[k] for f in filas if f[k] is not None] for k in ("movIzq", "movDer", "movPose")}
    return {
        "secuencias": len(filas),
        "sinManoIzq": round(sum(f["cuadrosSinIzq"] for f in filas) / cuadros, 4),
        "sinManoDer": round(sum(f["cuadrosSinDer"] for f in filas) / cuadros, 4),
        "movimientoMedioIzq": round(float(np.mean(mov["movIzq"])), 5) if mov["movIzq"] else None,
        "movimientoMedioDer": round(float(np.mean(mov["movDer"])), 5) if mov["movDer"] else None,
        "movimientoMedioPose": round(float(np.mean(mov["movPose"])), 5) if mov["movPose"] else None,
    }


# --------------------------------------------------------------------------
# Distribución del contrato (comparación con LSA-94)
# --------------------------------------------------------------------------

def distribucion(X: np.ndarray) -> dict:
    """Proporción de ceros por bloque, rangos por eje y movimiento medio de X (n,40,168)."""
    n, T, _ = X.shape
    pts = X.reshape(n, T, ec.POINTS, 3)
    bloques = {"manoIzq": slice(0, 21), "manoDer": slice(21, 42), "pose": slice(42, 56)}
    ceros = {}
    for nombre, sl in bloques.items():
        bloque = pts[:, :, sl, :]
        ceros[nombre] = round(float(np.mean(~np.any(bloque != 0.0, axis=(2, 3)))), 4)
    presentes = np.any(pts != 0.0, axis=-1)
    ejes = {}
    for i, eje in enumerate(("x", "y", "z")):
        vals = pts[..., i][presentes]
        ejes[eje] = {
            "p1": round(float(np.percentile(vals, 1)), 4),
            "p50": round(float(np.percentile(vals, 50)), 4),
            "p99": round(float(np.percentile(vals, 99)), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        }
    mov = {}
    for nombre, sl in bloques.items():
        bloque = pts[:, :, sl, :2]
        pres = np.any(bloque != 0.0, axis=-1)
        par = pres[:, 1:] & pres[:, :-1]
        d = np.linalg.norm(bloque[:, 1:] - bloque[:, :-1], axis=-1)
        mov[nombre] = round(float(d[par].mean()), 5) if par.any() else None
    return {"proporcionCeros": ceros, "ejes": ejes, "movimientoMedioPorCuadro": mov}


# --------------------------------------------------------------------------
# Fuentes de .pkl
# --------------------------------------------------------------------------

class FuenteZip:
    """Lee MEDIAPIPE/<id>.pkl directamente del zip, sin descomprimirlo."""

    def __init__(self, ruta: Path):
        self.ruta = ruta
        self._zip = zipfile.ZipFile(ruta)
        self._nombres = {}
        for nombre in self._zip.namelist():
            if nombre.startswith("MEDIAPIPE/") and nombre.endswith(".pkl"):
                self._nombres[Path(nombre).stem] = nombre

    def disponibles(self) -> set[str]:
        return set(self._nombres)

    def leer(self, sid: str) -> list[dict]:
        with self._zip.open(self._nombres[sid]) as f:
            return v.cargar_pkl(f)

    def cerrar(self):
        self._zip.close()


class FuenteCarpetas:
    """Lee <id>.pkl de una o más carpetas (muestras locales)."""

    def __init__(self, carpetas: list[Path]):
        self._rutas: dict[str, Path] = {}
        for carpeta in carpetas:
            for ruta in carpeta.glob("*.pkl"):
                self._rutas.setdefault(ruta.stem, ruta)

    def disponibles(self) -> set[str]:
        return set(self._rutas)

    def leer(self, sid: str) -> list[dict]:
        return v.cargar_pkl(self._rutas[sid])

    def cerrar(self):
        pass


# --------------------------------------------------------------------------
# Anotaciones y referencias
# --------------------------------------------------------------------------

def leer_anotaciones(ruta_zip: Path) -> dict[str, tuple[str, int]]:
    """id -> (partición oficial, clase 0..299) de ANNOTATIONS.zip."""
    anot: dict[str, tuple[str, int]] = {}
    with zipfile.ZipFile(ruta_zip) as z:
        for nombre_split in SPLIT_NOMBRES:
            contenido = z.read(f"ANNOTATIONS/{nombre_split}_labels.csv").decode("utf-8")
            for fila in csv.reader(contenido.splitlines()):
                if len(fila) != 2:
                    continue
                if fila[0] in anot:
                    raise ValueError(f"id repetido en las anotaciones: {fila[0]}")
                anot[fila[0]] = (nombre_split, int(fila[1]))
    return anot


def leer_referencias(ruta_csv: Path) -> dict[str, int]:
    """nombre de video rec... -> CLASS_ID 0..299 de videos_ref_annotations.csv."""
    with open(ruta_csv, encoding="utf-8", newline="") as f:
        return {fila["FILENAME"]: int(fila["CLASS_ID"]) for fila in csv.DictReader(f)}


def leer_glosas(ruta_csv: Path) -> list[str]:
    """LABEL de cada CLASS_ID 0..299, en orden (texto original del CSV)."""
    with open(ruta_csv, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    por_id = {int(fila["CLASS_ID"]): fila["LABEL"] for fila in filas}
    return [por_id[i] for i in range(len(por_id))]


def md5_archivo(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 22), b""):
            h.update(bloque)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Conversión
# --------------------------------------------------------------------------

def convertir(fuente, ids_procesar: list[str]) -> dict:
    """Procesa cada id: devuelve entradas válidas, inválidas y estadísticas antes/después."""
    filas = []
    invalidas = []
    antes, despues = [], []
    for sid in ids_procesar:
        pose, izq, der = v.secuencia_holistic(fuente.leer(sid))
        crudo = a_vector_168(pose, izq, der, corregir=True)
        try:
            entrada = entrada_v3(crudo)
        except ValueError as e:
            invalidas.append({"id": sid, "motivo": str(e)})
            continue
        antes.append(estadisticas_bloques(a_vector_168(pose, izq, der, corregir=False)))
        despues.append(estadisticas_bloques(crudo))
        filas.append({
            "id": sid,
            "x": entrada,
            "ratio": proporcion_hombro_torso(crudo),
            "esReferencia": not sid.isdigit(),
        })
    return {"filas": filas, "invalidas": invalidas, "antes": antes, "despues": despues}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fuente", type=Path, action="append",
                    help="MEDIAPIPE.zip o carpeta(s) con .pkl (repetible). Default: el zip.")
    ap.add_argument("--salida", type=Path, default=SALIDA_DEFAULT)
    ap.add_argument("--umbral-sesion", type=float, default=UMBRAL_SESION_S)
    ap.add_argument("--solo-validar", action="store_true",
                    help="Corre la validación de escala y no escribe salidas.")
    args = ap.parse_args(argv)

    fuentes = args.fuente or [ZIP_MEDIAPIPE]
    if len(fuentes) == 1 and fuentes[0].suffix.lower() == ".zip":
        if fuentes[0].stat().st_size != TAMANO_MEDIAPIPE:
            raise SystemExit(f"{fuentes[0]} no tiene el tamaño de Zenodo ({TAMANO_MEDIAPIPE} bytes): "
                             "descarga incompleta")
        md5 = md5_archivo(fuentes[0])
        if md5 != MD5_MEDIAPIPE:
            raise SystemExit(f"md5 de {fuentes[0]} = {md5}, se esperaba {MD5_MEDIAPIPE}")
        fuente = FuenteZip(fuentes[0])
    else:
        fuente = FuenteCarpetas(fuentes)

    anot = leer_anotaciones(ZIP_ANOTACIONES)
    refs = leer_referencias(CSV_REFERENCIA)
    glosas = leer_glosas(CSV_REFERENCIA)

    # Las referencias (rec...) figuran como train en ANNOTATIONS: se apartan a su propia partición.
    split_de: dict[str, int] = {}
    clase_de: dict[str, int] = {}
    for sid, (nombre_split, clase) in anot.items():
        if sid.isdigit():
            split_de[sid] = SPLIT_NOMBRES[nombre_split]
        else:
            split_de[sid] = SPLIT_REFERENCIA
        clase_de[sid] = clase
    faltan_refs = [n for n in refs if n not in anot]
    desacuerdos = [n for n in refs if n in anot and anot[n][1] != refs[n]]
    if faltan_refs or desacuerdos:
        raise SystemExit(f"referencias inconsistentes: sin anotación {faltan_refs[:5]}, "
                         f"clase distinta {desacuerdos[:5]}")

    # Sesiones sobre TODAS las secuencias numéricas (no sólo las disponibles).
    numericos = [int(s) for s in anot if s.isdigit()]
    sesion_num = asignar_sesiones(numericos, args.umbral_sesion)

    esperados = sorted(anot, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
    disponibles = fuente.disponibles()
    ids = [s for s in esperados if s in disponibles]
    faltantes = len(esperados) - len(ids)
    print(f"secuencias en anotaciones: {len(esperados)}, disponibles en la fuente: {len(ids)}")

    res = convertir(fuente, ids)
    fuente.cerrar()
    filas = res["filas"]

    # Validación de escala por partición (sólo contra las referencias)
    def ratios(cond):
        return [f["ratio"] for f in filas if cond(f) and f["ratio"] is not None]

    def mediana(xs):
        return float(np.median(xs)) if xs else None

    med_ref = mediana(ratios(lambda f: f["esReferencia"]))
    medianas = {
        "train": mediana(ratios(lambda f: not f["esReferencia"] and split_de[f["id"]] == 0)),
        "val": mediana(ratios(lambda f: not f["esReferencia"] and split_de[f["id"]] == 1)),
        "test": mediana(ratios(lambda f: not f["esReferencia"] and split_de[f["id"]] == 2)),
        "numericos": mediana(ratios(lambda f: not f["esReferencia"])),
    }
    fallas = violaciones_escala(medianas, med_ref)
    print(f"escala (ancho hombros / torso, mediana): referencia {med_ref}, numéricos {medianas}")
    if fallas:
        print("FRENO: la escala no coincide con la referencia:")
        for f in fallas:
            print("  -", f)
        return 2
    if args.solo_validar:
        print("validación de escala OK (no se escriben salidas con --solo-validar)")
        return 0

    # Salidas
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    X = (np.stack([f["x"] for f in filas]).astype(np.float32) if filas
         else np.zeros((0, ec.FRAMES, ec.COORDS), np.float32))
    nombres = np.array([f["id"] for f in filas])
    y = np.array([clase_de[f["id"]] + 1 for f in filas], dtype=np.int32)
    split = np.array([split_de[f["id"]] for f in filas], dtype=np.int8)
    subjects = np.array([sesion_num[int(f["id"])] if f["id"].isdigit() else -1
                         for f in filas], dtype=np.int32)
    reps = np.zeros(len(filas), dtype=np.int32)
    np.save(salida / "X.npy", X)
    np.save(salida / "y.npy", y)
    np.save(salida / "subjects.npy", subjects)
    np.save(salida / "split.npy", split)
    np.save(salida / "reps.npy", reps)
    np.save(salida / "nombres.npy", nombres)
    (salida / "invalidas.json").write_text(
        json.dumps(res["invalidas"], ensure_ascii=False, indent=2), encoding="utf-8")

    dist = {
        "salida": distribucion(X) if len(X) else None,
        "lsa94": distribucion(np.load(X_LSA94)) if X_LSA94.exists() else None,
        "antesDeCorregir": resumir_bloques(res["antes"]),
        "despuesDeCorregir": resumir_bloques(res["despues"]),
    }
    (salida / "distribucion.json").write_text(
        json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8")

    sens = {}
    for umbral in (300, 1800, 3600, 7200, 14400):
        ses_u = {int(s): x for s, x in asignar_sesiones(numericos, umbral).items()}
        sens[str(umbral)] = sesiones_cruzadas(ses_u, {int(s): split_de[s] for s in anot if s.isdigit()})

    meta = {
        "contractVersion": ec.CONTRACT_VERSION,
        "frames": ec.FRAMES,
        "coordenadas": ec.COORDS,
        "catalogo": {"archivo": CATALOGO.name, "clases": len(glosas),
                     "indice": "y = CLASS_ID + 1 (1..300), igual que LSA-94"},
        "fuente": {"dataset": "SWL-LSE", "zenodo": DOI, "licencia": "CC BY 4.0",
                   "archivo": "MEDIAPIPE.zip", "md5": MD5_MEDIAPIPE, "tamanoBytes": TAMANO_MEDIAPIPE,
                   "extractor": "holistic_legacy",
                   "usadas": [str(f) for f in fuentes]},
        "resolucion": {"W": ANCHO_PX, "H": ALTO_PX, "HW": ALTO_PX / ANCHO_PX,
                       "motivo": "referencias verificadas 480x480; señantes asumidos (ver LSE300_DATOS.md)"},
        "espejado": {
            "corregido": True,
            "transformacion": "x -> 1-x; bloques de manos intercambiados; pares de pose 11<->12, 13<->14, ..., 23<->24",
            "motivo": "el extractor de los autores aplica cv2.flip(image, 1) antes de MediaPipe",
        },
        "sesiones": {"umbralSegundos": args.umbral_sesion, "criterio": "salto entre IDs consecutivos > umbral",
                     "resumen": sesiones_cruzadas(sesion_num, {int(s): split_de[s] for s in anot if s.isdigit()}),
                     "sensibilidad": sens},
        "particiones": {SPLIT_ETIQUETAS[k]: int((split == k).sum()) for k in SPLIT_ETIQUETAS},
        "validacionEscala": {"referencia": med_ref, **medianas, "toleranciaRelativa": TOLERANCIA_ESCALA,
                             "definicion": "mediana por secuencia de ancho de hombros / torso, luego mediana"},
        "conteos": {"anotaciones": len(esperados), "disponibles": len(ids), "faltantes": faltantes,
                    "validas": len(filas), "invalidas": len(res["invalidas"])},
        "referenciasConsistentes": True,
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (salida / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"listo: {len(filas)} secuencias en {salida}, {len(res['invalidas'])} inválidas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
