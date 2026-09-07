#!/usr/bin/env python3
"""
train_keras_tflite.py — Entrena el clasificador LSA en Keras y lo exporta a
.tflite listo para LiteRT en Android, junto con los fixtures de verificacion.

Por que Keras y no PyTorch: la conversion PyTorch -> .tflite para LSTM pasa por
ONNX y es fragil. Keras convierte de forma nativa.

Por que unroll=True: sin eso la conversion genera ops de TensorList que exigen
la libreria Flex (SELECT_TF_OPS) en Android, varios MB extra y peor soporte de
delegates. Con unroll el grafo queda en ops elementales. Requiere largo de
secuencia fijo, que es justo nuestro caso (40 frames).

Salidas en el directorio --out:
    modelo_lsa.tflite     -> va a app/src/main/assets/
    catalogo_senas.json   -> va a assets/ (acoplado al modelo, misma version)
    fixture_android.json  -> va a app/src/androidTest/assets/
    metricas.json         -> para el registro de experimentos

Uso:
    pip install tensorflow scikit-learn
    python train_keras_tflite.py --data ./data_64_v2 --out ./export
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

FRAMES = 40
COORDS = 201

# Catalogo de las 64 senas de LSA64, en orden oficial (indice de clase 1..64).
# CRITICO: este catalogo esta acoplado al modelo. Los indices que devuelve el
# clasificador solo tienen sentido contra el catalogo de la misma version.
SENAS = [
    "Opaco", "Rojo", "Verde", "Amarillo", "Brillante", "Celeste", "Colores",
    "Rosa", "Mujer", "Enemigo", "Hijo", "Hombre", "Lejos", "Dibujar", "Nacer",
    "Aprender", "Llamar", "Skimmer", "Ubicacion", "Atrapar", "Gracias",
    "Aceptar", "Sordo", "Cuchillo", "Otro", "Ninguno", "Nombre", "Paciencia",
    "Perfume", "Deporte", "Cafe", "Uruguay", "Argentina", "Pais", "A_pesar_de",
    "Preguntar", "Cumpleanos", "Desayuno", "Foto", "Hambre", "Mapa",
    "Moneda", "Musica", "Barco", "Despues", "Duro", "Comida", "Aceite",
    "Fideos", "Pescado", "Acuerdo", "Duda", "Argolla", "Comprar", "Copa",
    "Bailar", "Novia", "Cerveza", "Guardar", "Candado", "Aguja", "Sur",
    "Aspirina", "Cruz",
]


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

def cargar(data_dir: Path):
    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy")
    ruta_suj = data_dir / "subjects.npy"
    if not ruta_suj.exists():
        raise SystemExit(
            f"Falta {ruta_suj}. El split por sujeto es obligatorio para "
            f"reportar metricas: regenerá el dataset con la version de "
            f"preprocess.py que guarda subjects.npy.")
    sujetos = np.load(ruta_suj)

    if X.shape[1] != FRAMES or X.shape[2] != COORDS:
        raise SystemExit(
            f"Forma inesperada {X.shape}: se esperaba (n, {FRAMES}, {COORDS}). "
            f"El contrato de keypoints no coincide.")

    clases = sorted(set(y.tolist()))
    mapa = {c: i for i, c in enumerate(clases)}     # etiquetas 1..64 -> 0..63
    y0 = np.array([mapa[v] for v in y], dtype=np.int32)
    return X, y0, sujetos, clases


def split_por_sujeto(X, y, sujetos, val_s, test_s):
    m_test = np.isin(sujetos, list(test_s))
    m_val = np.isin(sujetos, list(val_s))
    m_tr = ~(m_test | m_val)
    return (X[m_tr], y[m_tr]), (X[m_val], y[m_val]), (X[m_test], y[m_test])


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

def construir(n_clases: int, hidden: int = 128, dropout: float = 0.3):
    """Equivalente a la LSTM de PyTorch de la fase exploratoria.

    El dropout va como capa aparte (no dentro de la LSTM) y unroll=True para
    que la conversion a TFLite use solo ops nativas.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES, COORDS), name="keypoints"),
        tf.keras.layers.LSTM(hidden, return_sequences=True, unroll=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(hidden, unroll=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_clases, name="logits"),
    ], name="clasificador_lsa")


# --------------------------------------------------------------------------
# Exportacion
# --------------------------------------------------------------------------

FLEX_HINT = ("flex", "tensorlist")


def exportar_tflite(modelo, destino: Path) -> bytes:
    """Convierte exigiendo ops nativas. Si necesitara Flex, aborta:
    preferimos enterarnos acá y no en el dispositivo."""
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        blob = conv.convert()
    except Exception as e:
        raise SystemExit(
            "La conversion requiere SELECT_TF_OPS (libreria Flex en Android).\n"
            "Revisá que las capas LSTM tengan unroll=True y que el dropout no "
            "esté dentro de la LSTM.\n"
            f"Detalle: {str(e)[:300]}")

    destino.write_bytes(blob)

    # Verificacion de ops: ninguna debe delatar Flex
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    ops = sorted({d["op_name"] for d in it._get_ops_details()})
    sospechosas = [o for o in ops if any(h in o.lower() for h in FLEX_HINT)]
    if sospechosas:
        raise SystemExit(f"El .tflite usa ops que necesitan Flex: {sospechosas}")

    print(f"  ops del grafo: {', '.join(ops)}")
    return blob


def verificar_equivalencia(modelo, blob: bytes, X: np.ndarray, n: int = 24):
    """Keras y TFLite deben dar la misma salida. Si difieren, el modelo
    desplegado no es el modelo evaluado."""
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    ent = it.get_input_details()[0]
    sal = it.get_output_details()[0]

    muestras = X[:n]
    esperado = modelo.predict(muestras, verbose=0)

    obtenido = []
    for x in muestras:
        it.set_tensor(ent["index"], x[None, ...].astype(np.float32))
        it.invoke()
        obtenido.append(it.get_tensor(sal["index"])[0])
    obtenido = np.array(obtenido)

    maxdiff = float(np.abs(esperado - obtenido).max())
    coinciden = int((esperado.argmax(1) == obtenido.argmax(1)).sum())
    return maxdiff, coinciden, len(muestras)


def escribir_fixture(destino: Path, blob: bytes, X, y, clases, n=8, seed=0):
    """Fixture para el test instrumentado de Android.

    Es la pieza clave de la etapa 1: permite verificar el modelo en el
    dispositivo SIN camara ni MediaPipe. Si este test pasa, el despliegue del
    modelo esta resuelto y cualquier falla posterior es del pipeline de captura.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]

    casos = []
    for i in idx:
        x = X[i].astype(np.float32)
        it.set_tensor(ent["index"], x[None, ...])
        it.invoke()
        logits = it.get_tensor(sal["index"])[0]
        pred = int(logits.argmax())
        casos.append({
            "indiceMuestra": int(i),
            "claseEsperada": pred,                       # lo que debe predecir
            "senaEsperada": SENAS[pred] if pred < len(SENAS) else f"clase_{pred}",
            "etiquetaReal": int(y[i]),                   # verdad del dataset
            "acierta": bool(pred == int(y[i])),
            "logitsEsperados": [round(float(v), 5) for v in logits],
            "secuencia": [[round(float(v), 6) for v in frame] for frame in x],
        })

    destino.write_text(json.dumps({
        "descripcion": "Fixture para verificar el modelo en Android sin camara.",
        "frames": FRAMES,
        "coordenadas": COORDS,
        "toleranciaLogits": 1e-3,
        "casos": casos,
    }, ensure_ascii=False), encoding="utf-8")
    return len(casos)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./export")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--paciencia", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-subjects", nargs="*", type=int, default=[9])
    ap.add_argument("--test-subjects", nargs="*", type=int, default=[10])
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, sujetos, clases = cargar(data_dir)
    n_clases = len(clases)
    print(f"Datos: X={X.shape}  clases={n_clases}  sujetos={sorted(set(sujetos.tolist()))}")

    val_s, test_s = set(args.val_subjects), set(args.test_subjects)
    if val_s & test_s:
        raise SystemExit("val-subjects y test-subjects no pueden solaparse.")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_por_sujeto(X, y, sujetos, val_s, test_s)
    print(f"Split POR SUJETO -> train={len(ytr)}  val={len(yva)} (suj {sorted(val_s)})  "
          f"test={len(yte)} (suj {sorted(test_s)})")

    modelo = construir(n_clases, args.hidden, args.dropout)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"Parametros: {modelo.count_params():,}")

    print("\n--- Entrenamiento ---")
    hist = modelo.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.paciencia,
            restore_best_weights=True, verbose=1)],
    )

    mejor_val = float(max(hist.history["val_accuracy"]))
    mejor_ep = int(np.argmax(hist.history["val_accuracy"])) + 1
    te_loss, te_acc = modelo.evaluate(Xte, yte, verbose=0)

    print("\n=== Resultado ===")
    print(f"mejor val acc: {mejor_val:.4f} (epoca {mejor_ep})")
    print(f"TEST acc: {te_acc:.4f}  loss: {te_loss:.4f}")
    print(f"(baseline al azar con {n_clases} clases: {1/n_clases:.4f})")

    # --- Matriz de confusion y metricas por clase ---
    logits_te = modelo.predict(Xte, verbose=0)
    pred_te = logits_te.argmax(1)
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for real, p in zip(yte, pred_te):
        cm[real, p] += 1

    confusiones = []
    for i in range(n_clases):
        for j in range(n_clases):
            if i != j and cm[i, j] > 0:
                confusiones.append((int(cm[i, j]), SENAS[i] if i < len(SENAS) else str(i),
                                    SENAS[j] if j < len(SENAS) else str(j)))
    confusiones.sort(reverse=True)
    if confusiones:
        print("\nPares mas confundidos (real -> predicho):")
        for c, a, b in confusiones[:10]:
            print(f"  {c}x  {a} -> {b}")

    # --- Exportacion ---
    print("\n--- Exportacion a TFLite ---")
    ruta_tflite = out_dir / "modelo_lsa.tflite"
    blob = exportar_tflite(modelo, ruta_tflite)
    print(f"  {ruta_tflite.name}: {len(blob)/1024:.0f} KB")

    maxdiff, ok, tot = verificar_equivalencia(modelo, blob, Xte)
    print(f"  equivalencia Keras vs TFLite: maxdiff {maxdiff:.2e}, "
          f"misma prediccion {ok}/{tot}")
    if maxdiff > 1e-3 or ok != tot:
        raise SystemExit("El .tflite NO es equivalente al modelo entrenado.")

    n_fix = escribir_fixture(out_dir / "fixture_android.json", blob, Xte, yte, clases)
    print(f"  fixture_android.json: {n_fix} casos")

    (out_dir / "catalogo_senas.json").write_text(json.dumps({
        "version": "1.0",
        "cantidadClases": n_clases,
        "nota": "Los indices son 0..N-1 y corresponden a la salida del modelo. "
                "Acoplado al .tflite de la misma version.",
        "senas": [{"indice": i, "claseDataset": int(c),
                   "nombre": SENAS[i] if i < len(SENAS) else f"clase_{i}"}
                  for i, c in enumerate(clases)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "metricas.json").write_text(json.dumps({
        "testAccuracy": round(float(te_acc), 4),
        "valAccuracyMejor": round(mejor_val, 4),
        "epocaMejor": mejor_ep,
        "baselineAzar": round(1 / n_clases, 4),
        "split": "por_sujeto",
        "sujetosVal": sorted(val_s),
        "sujetosTest": sorted(test_s),
        "seed": args.seed,
        "hiperparametros": {
            "lr": args.lr, "batch": args.batch, "hidden": args.hidden,
            "dropout": args.dropout, "epochs": args.epochs,
            "paciencia": args.paciencia, "unroll": True,
        },
        "tfVersion": tf.__version__,
        "paresMasConfundidos": [
            {"veces": c, "real": a, "predicho": b} for c, a, b in confusiones[:15]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nListo. Archivos en {out_dir.resolve()}")
    print("  modelo_lsa.tflite      -> app/src/main/assets/")
    print("  catalogo_senas.json    -> app/src/main/assets/")
    print("  fixture_android.json   -> app/src/androidTest/assets/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())