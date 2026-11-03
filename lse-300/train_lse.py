#!/usr/bin/env python3
"""
train_lse.py — Entrena y exporta eva-lse300-v1 (SWL-LSE, 299 clases) con la
misma receta que eva-lsa94-v4. No copia el entrenamiento: importa las
funciones de ../lsa-94/train_keras_tflite.py.

Particiones (--particion):
  sesion   (principal) train / val / test agrupados por sesión: ninguna sesión
           aparece en más de una partición. Los videos de referencia
           (split 3) van siempre a train.
  oficial  (secundaria) split.npy del dataset. Puede tener fuga de señante
           (51 de 303 sesiones cruzan particiones); sirve para comparar con el
           paper de los autores.

Pasos (--modo):
  eval   entrena con train y valida con val (sin early stopping: la época
         sugerida sale de la curva de validación). Evalúa en test, calcula el
         umbral de confianza con val y guarda métricas en runs/.
  final  entrena con train + val + refs durante la cantidad de épocas de la
         curva de validación del eval (--desde-eval) y exporta a exports/.

Clase excluida: la #N/A (CLASS_ID 222 del catálogo, etiqueta 223 en
y.npy). Se quita junto con sus muestras y el resto se renumera 1..299 en
el mismo orden. El índice original SWL-LSE de cada clase queda en las métricas.

Uso: ver lse-300/EVA_LSE300_ENTRENAMIENTO.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

LSE_DIR = Path(__file__).resolve().parent
RAIZ = LSE_DIR.parent
for _ruta in (RAIZ, RAIZ / "lsa-94"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import train_keras_tflite as tk  # noqa: E402  (carga ../.env antes de importar Keras)
from multilengua.manifiesto import ORIGEN_SWL_LSE  # noqa: E402

DATA = LSE_DIR / "data" / "processed_v3"
CATALOGO = LSE_DIR / "catalogo_lse300.json"
EXPORTS = LSE_DIR / "exports"
RUNS = LSE_DIR / "runs"
LSA94_PESOS = RAIZ / "lsa-94" / "exports" / "modelo.weights.h5"
MODEL_VERSION = "eva-lse300-v1"
GRUPO = "swl_lse"
ETIQUETA_EXCLUIDA = "#N/A"
SPLIT_REF = 3
ASIGN_TRAIN, ASIGN_VAL, ASIGN_TEST, ASIGN_REF = 0, 1, 2, 3
CLASES_LSA94 = 94
EXACTITUD_OBJETIVO = 0.90
ACEPTADAS_MIN = 30
EXPORTACION_LSE = {
    "nombre_tflite": "modelo_lse.tflite",
    "manifiesto": "manifest.json",
    "language": "lse",
    "origen": ORIGEN_SWL_LSE,
}


# --------------------------------------------------------------------------
# Catálogo y datos
# --------------------------------------------------------------------------

def cargar_catalogo(path: Path = CATALOGO) -> dict:
    """Quita la clase #N/A del catálogo de 300 y deja las 299 restantes en orden.

    Devuelve glosas (299), claseIdSwl (CLASS_ID 0-based de cada clase nueva),
    la clase excluida y las señas relevantes en etiquetas nuevas 1..299.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    etiquetas, glosas = raw["etiquetasOriginales"], raw["glosas"]
    if len(etiquetas) != len(glosas):
        raise ValueError("etiquetasOriginales y glosas tienen largos distintos")
    excluidas = [i for i, e in enumerate(etiquetas) if e.strip() == ETIQUETA_EXCLUIDA]
    if len(excluidas) != 1:
        raise ValueError(f"Se esperaba una sola clase {ETIQUETA_EXCLUIDA}, hay {len(excluidas)}")
    ex = excluidas[0]
    conservadas = [i for i in range(len(glosas)) if i != ex]
    nuevo_de_viejo = {viejo: nuevo for nuevo, viejo in enumerate(conservadas)}
    # relevantesHelpi usa etiquetas 1..300 (CLASS_ID + 1), igual que y.npy.
    relevantes = {}
    for categoria, ids in raw["relevantesHelpiPorCategoria"].items():
        relevantes[categoria] = sorted(nuevo_de_viejo[i - 1] + 1 for i in ids if i - 1 != ex)
    return {
        "version": raw["version"],
        "glosas": [glosas[i] for i in conservadas],
        "claseIdSwl": conservadas,
        "excluida": {"claseIdSwl": ex, "etiquetaOriginal": etiquetas[ex], "glosa": glosas[ex]},
        "relevantesPorCategoria": relevantes,
    }


def cargar_datos(data_dir: Path, cat: dict):
    """Carga X, y (etiquetas nuevas 1..299), sesión, split y nombres sin la clase excluida."""
    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.int64)
    sesion = np.load(data_dir / "subjects.npy").astype(np.int64)
    split = np.load(data_dir / "split.npy").astype(np.int64)
    nombres = np.load(data_dir / "nombres.npy")
    if X.shape[1:] != (tk.FRAMES, tk.COORDS):
        raise SystemExit(f"Forma inesperada {X.shape}")
    lut = np.zeros(int(y.max()) + 1, dtype=np.int64)          # 0 = excluida
    for nuevo, viejo in enumerate(cat["claseIdSwl"]):
        lut[viejo + 1] = nuevo + 1
    y_nuevo = lut[y]
    ok = y_nuevo > 0
    return X[ok], y_nuevo[ok], sesion[ok], split[ok], nombres[ok], int((~ok).sum())


# --------------------------------------------------------------------------
# Particiones
# --------------------------------------------------------------------------

def particion_oficial(split: np.ndarray) -> np.ndarray:
    """split.npy ya usa la misma codificación: 0 train, 1 val, 2 test, 3 referencia."""
    return split.astype(np.int8)


def particion_por_sesion(sesion, y, split, seed=42, fracciones=(0.70, 0.15, 0.15),
                         iteraciones=200_000, tolerancia=0.01):
    """Asigna sesiones (no secuencias) a train, val y test.

    Recocido simulado: intenta maximizar cuántas clases tienen al menos una
    muestra en cada partición, con los tamaños cerca de 70/15/15 de las
    secuencias de señantes (tolerancia de 1 % del total). Las referencias
    (split 3) van a train. Es determinista para una semilla dada.
    """
    n = len(y)
    asign = np.full(n, ASIGN_REF, dtype=np.int8)
    no_ref = split != SPLIT_REF
    sesiones, inv = np.unique(sesion[no_ref], return_inverse=True)
    n_cls = int(y.max())
    C = np.zeros((len(sesiones), n_cls), dtype=np.int64)
    np.add.at(C, (inv, y[no_ref] - 1), 1)
    tam_s = C.sum(1)
    N = int(tam_s.sum())
    meta = np.array(fracciones, dtype=float) * N
    tol = tolerancia * N
    rng = np.random.default_rng(seed)
    S = len(sesiones)

    est = np.zeros(S, dtype=np.int64)               # 0 train, 1 val, 2 test
    acum = {2: 0.0, 1: 0.0}
    for i in rng.permutation(S):
        if acum[2] < meta[2]:
            est[i] = 2
            acum[2] += tam_s[i]
        elif acum[1] < meta[1]:
            est[i] = 1
            acum[1] += tam_s[i]

    cnt = np.zeros((3, n_cls), dtype=np.int64)
    tam = np.zeros(3)
    for p in range(3):
        cnt[p] = C[est == p].sum(0)
        tam[p] = tam_s[est == p].sum()

    def energia(cn, tm):
        cobertura = int((cn > 0).sum())
        exceso = np.maximum(0.0, np.abs(tm - meta) - tol).sum()
        return -float(cobertura) + 10.0 * exceso / tol

    E = energia(cnt, tam)
    mejor_E, mejor = E, est.copy()
    T0, T1 = 2.0, 0.02
    for it in range(iteraciones):
        T = T0 * (T1 / T0) ** (it / iteraciones)
        i = int(rng.integers(S))
        if rng.random() < 0.5:
            movs = [(i, int(est[i]), int((est[i] + rng.integers(1, 3)) % 3))]
        else:
            j = int(rng.integers(S))
            if est[j] == est[i]:
                continue
            movs = [(i, int(est[i]), int(est[j])), (j, int(est[j]), int(est[i]))]
        cn, tm = cnt.copy(), tam.copy()
        for k, p, q in movs:
            cn[p] -= C[k]
            cn[q] += C[k]
            tm[p] -= tam_s[k]
            tm[q] += tam_s[k]
        E2 = energia(cn, tm)
        if E2 <= E or rng.random() < math.exp(-(E2 - E) / T):
            for k, _, q in movs:
                est[k] = q
            cnt, tam, E = cn, tm, E2
            if E < mejor_E:
                mejor_E, mejor = E, est.copy()

    asign[no_ref] = mejor[inv].astype(np.int8)
    return asign


def sesiones_en_varias_particiones(sesion, asign) -> int:
    """Cuántas sesiones tienen muestras en más de una partición (sin contar refs)."""
    donde = defaultdict(set)
    for s, a in zip(sesion, asign):
        if a != ASIGN_REF:
            donde[int(s)].add(int(a))
    return sum(1 for v in donde.values() if len(v) > 1)


def hash_particion(asign: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(asign).tobytes()).hexdigest()


def resumen_particion(asign, sesion, y0, n_cls) -> dict:
    """Tamaños, sesiones y cobertura de clases por partición (train incluye refs)."""
    partes = {"train": (ASIGN_TRAIN, ASIGN_REF), "val": (ASIGN_VAL,), "test": (ASIGN_TEST,)}
    out = {}
    for nombre, codigos in partes.items():
        m = np.isin(asign, codigos)
        conteo = np.bincount(y0[m], minlength=n_cls)
        sin = [int(c) for c in np.flatnonzero(conteo == 0)]
        out[nombre] = {
            "muestras": int(m.sum()),
            "sesiones": int(len(np.unique(sesion[m & (asign != ASIGN_REF)]))),
            "clasesSinMuestra": len(sin),
            "clasesSinMuestraIdx": sin,
            "muestrasPorClase": {"min": int(conteo.min()), "mediana": float(np.median(conteo)),
                                 "max": int(conteo.max())},
        }
    out["sesionesEnVariasParticiones"] = sesiones_en_varias_particiones(sesion, asign)
    return out


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def metricas_prediccion(logits, y0, glosas, relevantes_por_categoria):
    """Top-1, top-3, exactitud por clase, peores clases, relevantes y confusión."""
    n_cls = len(glosas)
    pred = logits.argmax(1)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (y0, pred), 1)
    filas = []
    for c in range(n_cls):
        n = int(cm[c].sum())
        if n == 0:
            continue
        acierto = int(cm[c, c])
        fuera = cm[c].copy()
        fuera[c] = 0
        d = int(fuera.argmax()) if fuera.any() else None
        filas.append({
            "clase": c + 1, "glosa": glosas[c], "n": n,
            "exactitud": round(acierto / n, 4),
            "principalConfusion": (
                {"glosa": glosas[d], "veces": int(fuera[d])} if d is not None and fuera[d] else None),
        })
    peores = sorted(filas, key=lambda f: (f["exactitud"], -f["n"]))[:15]
    relevantes = {}
    todas = []
    for categoria, etiquetas in relevantes_por_categoria.items():
        m = np.isin(y0 + 1, etiquetas)
        todas.extend(etiquetas)
        relevantes[categoria] = {
            "n": int(m.sum()),
            "exactitud": round(float((pred[m] == y0[m]).mean()), 4) if m.any() else None,
        }
    m = np.isin(y0 + 1, sorted(set(todas)))
    relevantes["total"] = {
        "n": int(m.sum()),
        "exactitud": round(float((pred[m] == y0[m]).mean()), 4) if m.any() else None,
    }
    resumen = {
        "n": int(len(y0)),
        "top1": round(float((pred == y0).mean()), 4),
        "top3": round(float((top3 == y0[:, None]).any(1).mean()), 4),
        "exactitudMediaPorClase": round(float(np.mean([f["exactitud"] for f in filas])), 4),
        "clasesEvaluadas": len(filas),
        "peoresClases": peores,
        "relevantesHelpi": relevantes,
        "exactitudPorClase": filas,
    }
    return resumen, cm


def curva_cobertura(probs, y0):
    """Para cada umbral de confianza (softmax): cobertura y exactitud de lo aceptado."""
    conf = probs.max(1)
    ok = probs.argmax(1) == y0
    filas = []
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        m = conf >= t
        n = int(m.sum())
        filas.append({
            "umbral": float(t),
            "cobertura": round(float(m.mean()), 4),
            "aceptadas": n,
            "exactitudAceptadas": round(float(ok[m].mean()), 4) if n else None,
        })
    return filas


def elegir_umbral(curva, exactitud_objetivo=EXACTITUD_OBJETIVO, aceptadas_min=ACEPTADAS_MIN):
    """Menor umbral con exactitud >= objetivo en lo aceptado: es el de mayor cobertura."""
    for fila in curva:
        if (fila["exactitudAceptadas"] is not None
                and fila["exactitudAceptadas"] >= exactitud_objetivo
                and fila["aceptadas"] >= aceptadas_min):
            return fila
    return None


def aplicar_umbral(probs, y0, umbral: float) -> dict:
    conf = probs.max(1)
    m = conf >= umbral
    ok = probs.argmax(1) == y0
    return {
        "umbral": umbral,
        "cobertura": round(float(m.mean()), 4),
        "aceptadas": int(m.sum()),
        "exactitudAceptadas": round(float(ok[m].mean()), 4) if m.any() else None,
    }


def epoca_sugerida(curva, ventana=5) -> int:
    """Época (1-based) donde termina la mejor media móvil de la curva de validación."""
    c = np.asarray(curva, dtype=float)
    k = min(ventana, len(c))
    suave = np.convolve(c, np.ones(k) / k, mode="valid")
    return int(np.argmax(suave)) + k


def guardar_matriz(cm, glosas, base: Path):
    """Matriz de confusión como JSON (conteos) y PNG (escala log)."""
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.with_suffix(".json")).write_text(json.dumps({
        "glosas": glosas, "matriz": cm.tolist(),
    }, ensure_ascii=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    im = ax.imshow(np.log1p(cm), cmap="magma", interpolation="nearest")
    ax.set_title("Matriz de confusion (test, escala log1p). Filas: real; columnas: predicha")
    ax.set_xlabel("predicha (clase 1..299)")
    ax.set_ylabel("real (clase 1..299)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------

def argumentos_receta(args) -> argparse.Namespace:
    """Argumentos de la receta de LSA-94 (arquitectura, augmentación, balance)."""
    return tk.construir_parser().parse_args([
        "--data", str(DATA),
        "--catalog", str(CATALOGO),
        "--model-version", MODEL_VERSION,
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
    ])


def correr_eval(args, targs, X, y0, sesion, asign, cat, nombre: str) -> dict:
    glosas = cat["glosas"]
    n_cls = len(glosas)
    grupos = {GRUPO: (1, n_cls)}
    tr = np.isin(asign, [ASIGN_TRAIN, ASIGN_REF])
    va = asign == ASIGN_VAL
    te = asign == ASIGN_TEST
    init = LSA94_PESOS if args.init_lsa94 else None
    if init is not None and not init.exists():
        raise SystemExit(f"No existe {init}")

    print(f"Eval '{nombre}': train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"epocas={targs.epochs} init={'LSA-94' if init else 'aleatoria'}", flush=True)
    t0 = time.time()
    modelo, hist = tk.entrenar(
        targs, X[tr], y0[tr], X[va], y0[va], n_cls, targs.epochs, targs.seed,
        early_stopping=False, sujetos_tr=sesion[tr],
        pesos_iniciales=init, n_clases_iniciales=CLASES_LSA94 if init else None)
    minutos = round((time.time() - t0) / 60, 1)
    curva_val = [float(v) for v in hist["val_accuracy"]]

    logits_va = modelo.predict(X[va], verbose=0, batch_size=256)
    logits_te = modelo.predict(X[te], verbose=0, batch_size=256)
    pv, pt = softmax(logits_va), softmax(logits_te)

    test, cm = metricas_prediccion(logits_te, y0[te], glosas, cat["relevantesPorCategoria"])
    val_top1 = round(float((logits_va.argmax(1) == y0[va]).mean()), 4)
    curva = curva_cobertura(pv, y0[va])
    umbral = elegir_umbral(curva)
    umbral_test = aplicar_umbral(pt, y0[te], umbral["umbral"]) if umbral else None
    ep = epoca_sugerida(curva_val)

    guardar_matriz(cm, glosas, RUNS / f"{nombre}_matriz")
    print(f"  test top1 {test['top1']:.4f}  top3 {test['top3']:.4f}  "
          f"media por clase {test['exactitudMediaPorClase']:.4f}  ({minutos} min)")
    print(f"  val top1 {val_top1:.4f}  mejor val {max(curva_val):.4f}  "
          f"epoca sugerida {ep}")
    if umbral:
        print(f"  umbral {umbral['umbral']:.2f}: cobertura val {umbral['cobertura']:.4f} "
              f"exactitud aceptadas {umbral['exactitudAceptadas']:.4f}")
    else:
        print("  ningun umbral llega a la exactitud objetivo en validacion")

    resultado = {
        "modo": "eval",
        "nombre": nombre,
        "particion": args.particion,
        "modelVersion": MODEL_VERSION,
        "catalogVersion": cat["version"],
        "clasesModelo": n_cls,
        "contractVersion": tk.CONTRACT_VERSION,
        "particionSha256": hash_particion(asign),
        "conjuntos": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "init": "lsa94_pesos_sin_salida" if init else "aleatoria",
        "epocas": targs.epochs,
        "epocaSugerida": ep,
        "minutos": minutos,
        "valTop1Final": val_top1,
        "valCurva": [round(v, 4) for v in curva_val],
        "historia": tk.historia_redondeada(hist),
        "test": test,
        "umbral": {
            "exactitudObjetivo": EXACTITUD_OBJETIVO,
            "aceptadasMinimas": ACEPTADAS_MIN,
            "elegidoEnValidacion": umbral,
            "testConEseUmbral": umbral_test,
            "curvaValidacion": curva,
        },
        "metadatosSenante": "no disponibles en el dataset (sin marca de sordo, interprete o estudiante)",
        "hiperparametros": tk.hiperparametros(targs),
        "entorno": tk.entorno_entrenamiento(),
        "parametros": int(modelo.count_params()),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{nombre}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return resultado, modelo


def correr_final(args, targs, X, y0, sesion, asign, cat, desde: dict):
    glosas = cat["glosas"]
    n_cls = len(glosas)
    if hash_particion(asign) != desde["particionSha256"]:
        raise SystemExit("La partición actual no coincide con la del eval: revisar datos o semilla.")
    epocas = args.epocas_final or desde["epocaSugerida"]
    tr = np.isin(asign, [ASIGN_TRAIN, ASIGN_VAL, ASIGN_REF])
    te = asign == ASIGN_TEST
    init = LSA94_PESOS if args.init_lsa94 else None
    print(f"Final: train+val+refs={tr.sum()} epocas={epocas} (eval sugirio "
          f"{desde['epocaSugerida']})", flush=True)
    t0 = time.time()
    modelo, hist = tk.entrenar(
        targs, X[tr], y0[tr], None, None, n_cls, epocas, targs.seed,
        early_stopping=False, sujetos_tr=sesion[tr],
        pesos_iniciales=init, n_clases_iniciales=CLASES_LSA94 if init else None)
    minutos = round((time.time() - t0) / 60, 1)

    EXPORTS.mkdir(parents=True, exist_ok=True)
    guardar_matriz(np.array(desde["_matriz"]), glosas, EXPORTS / "matriz_confusion_eval")
    metricas = {
        "modo": "final_particion_sesion",
        "contractVersion": tk.CONTRACT_VERSION,
        "particion": args.particion,
        "particionSha256": desde["particionSha256"],
        "entrenadoCon": "train + val + referencias (la partición de test no se usó)",
        "epocas": epocas,
        "epocaSugerida": desde["epocaSugerida"],
        "minutos": minutos,
        "trainAccuracyFinal": round(float(hist["accuracy"][-1]), 4),
        "historia": tk.historia_redondeada(hist),
        "evaluacionDelModeloDeEval": {k: desde[k] for k in
                                      ("test", "umbral", "valCurva", "conjuntos")},
        "umbralRecomendado": desde["umbral"]["elegidoEnValidacion"],
        "hiperparametros": tk.hiperparametros(targs),
        "catalogo": {"version": cat["version"], "archivo": "catalogo_senas.json",
                     "origenArchivo": str(CATALOGO.name), "grupos": {GRUPO: [1, n_cls]},
                     "excluida": cat["excluida"]},
        "clases": [{"clase": i + 1, "glosa": g, "claseIdSwl": int(cat["claseIdSwl"][i])}
                   for i, g in enumerate(glosas)],
        "metadatosSenante": "no disponibles en el dataset",
        "entorno": tk.entorno_entrenamiento(),
    }
    tk.exportar(modelo, targs, EXPORTS, X[te], y0[te], glosas, cat["version"], n_cls,
                metricas, grupos={GRUPO: (1, n_cls)}, exportacion=EXPORTACION_LSE,
                script=Path(__file__).resolve())
    (EXPORTS / "umbral.json").write_text(json.dumps(
        desde["umbral"]["elegidoEnValidacion"], ensure_ascii=False, indent=2), encoding="utf-8")
    return metricas


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--exportar-desde" in argv:               # proceso hijo de la exportación
        ap_hijo = argparse.ArgumentParser()
        ap_hijo.add_argument("--exportar-desde", type=Path, required=True)
        return tk.exportar_desde_pesos(ap_hijo.parse_args(argv).exportar_desde)

    ap = argparse.ArgumentParser(description="Entrena y exporta eva-lse300-v1.")
    ap.add_argument("--modo", choices=("eval", "final"), required=True)
    ap.add_argument("--particion", choices=("sesion", "oficial"), default="sesion")
    ap.add_argument("--nombre", default=None, help="nombre de la corrida en runs/")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epocas-final", type=int, default=None,
                    help="por defecto, la época sugerida por el eval")
    ap.add_argument("--desde-eval", type=Path, default=None,
                    help="runs/<nombre>.json del eval de la partición por sesión")
    ap.add_argument("--init-lsa94", action="store_true",
                    help="arranca de los pesos de LSA-94 (sin la capa de salida)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cat = cargar_catalogo()
    X, y0_1, sesion, split, nombres, excluidas = cargar_datos(DATA, cat)
    y0 = y0_1 - 1
    n_cls = len(cat["glosas"])
    if args.particion == "sesion":
        asign = particion_por_sesion(sesion, y0_1, split, seed=args.seed)
        ses_cruzadas = sesiones_en_varias_particiones(sesion, asign)
        if ses_cruzadas:
            raise SystemExit(f"Partición por sesión inválida: {ses_cruzadas} sesiones cruzadas")
    else:
        asign = particion_oficial(split)
    resumen = resumen_particion(asign, sesion, y0, n_cls)
    nombre = args.nombre or (f"{args.particion}_e{args.epochs}" + ("_lsa94init" if args.init_lsa94 else ""))
    print(f"Datos: X={X.shape} clases={n_cls} (excluidas muestras: {excluidas}) "
          f"particion={args.particion} hash={hash_particion(asign)[:12]}")
    print("Particion:", json.dumps({k: resumen[k] for k in ("train", "val", "test")},
                                   ensure_ascii=False))
    targs = argumentos_receta(args)

    if args.modo == "eval":
        resultado, _ = correr_eval(args, targs, X, y0, sesion, asign, cat, nombre)
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{nombre}_particion.json").write_text(json.dumps({
            "particionSha256": hash_particion(asign), "resumen": resumen,
            "sesionesPorParticion": {
                "train": sorted({int(s) for s in sesion[np.isin(asign, [0, 3])] if s >= 0}),
                "val": sorted({int(s) for s in sesion[asign == 1]}),
                "test": sorted({int(s) for s in sesion[asign == 2]}),
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if not args.desde_eval:
        raise SystemExit("--modo final requiere --desde-eval runs/<nombre>.json")
    desde = json.loads(args.desde_eval.read_text(encoding="utf-8"))
    matriz = json.loads((args.desde_eval.with_name(args.desde_eval.stem + "_matriz.json"))
                        .read_text(encoding="utf-8"))
    desde["_matriz"] = matriz["matriz"]
    correr_final(args, targs, X, y0, sesion, asign, cat, desde)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
