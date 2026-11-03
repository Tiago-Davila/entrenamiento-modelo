#!/usr/bin/env python3
"""
train_keras_tflite.py — Entrena el clasificador LSA (Eva, contrato v3) en
Keras y lo exporta a .tflite listo para LiteRT en Android, junto con los
fixtures de verificacion.

Por que Keras y no PyTorch: la conversion PyTorch -> .tflite pasa por ONNX y
es fragil. Keras convierte de forma nativa. Toda capa debe convertir con ops
builtin (sin Flex): exportar_tflite() aborta si no.

Novedades respecto de v2:
  - Entrada en contrato v3 (anchos de hombro; ver eva_contract.py).
  - Augmentacion NUEVA en cada epoca (AugmentedSequence), no copias fijas.
  - Features dentro del grafo (FeaturesEva): mano relativa a la muñeca y
    velocidades. Android sigue entregando solo el tensor 40x168.
  - Arquitecturas: 'lstm' (la de v2, unroll=True) y 'conv_tx' (Conv1D
    depthwise + transformer, receta del 1er puesto de Kaggle GISLR 2023).
  - AdamW + coseno con warmup + label smoothing.
  - Modos:
      split (default)  entrena con 8 sujetos, valida con --val-subjects y
                       reporta test con --test-subjects.
      --cv loso        deja un sujeto afuera por vez; elige la cantidad de
                       epocas por la curva media de validacion. No exporta.
      --final          entrena con TODOS los sujetos una cantidad fija de
                       epocas (la elegida en CV) y exporta.

Salidas en el directorio --out:
    modelo_lsa.tflite     -> app/src/main/assets/lsa/
    catalogo_senas.json   -> app/src/main/assets/lsa/ (acoplado al modelo)
    lsa-manifest.json     -> app/src/main/assets/lsa/
    fixture_android.json  -> app/src/androidTest/assets/
    metricas.json         -> registro de experimentos

Uso:
    python train_keras_tflite.py --data ../data/processed_v3 --out ./exports --arch conv_tx
    python train_keras_tflite.py --data ../data/processed_v3 --cv loso --arch lstm --epochs 150
    python train_keras_tflite.py --data ../data/processed_v3 --final --epochs 140 --out ./exports
"""

from __future__ import annotations

import entorno  # noqa: F401  (carga ../.env: KERAS_BACKEND antes de importar keras)

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras import layers, ops
from augment import AugmentConfig, Augmenter
from eva_contract import CATALOG_PATH, CONTRACT_VERSION, COORDS, FRAMES, load_catalog


def _tf():
    """TensorFlow solo hace falta para exportar a TFLite: se importa tarde para
    no cargarlo cuando se entrena con el backend torch (GPU)."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf


def entorno_entrenamiento() -> dict:
    info = {"keras": keras.__version__, "backend": keras.backend.backend()}
    if info["backend"] == "torch":
        import torch
        info["torch"] = torch.__version__
        info["dispositivo"] = (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else "cpu")
    return info

# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

def cargar(data_dir: Path, catalog: list[str]):
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    version = meta.get("contractVersion", 2)
    if version != CONTRACT_VERSION:
        raise SystemExit(
            f"{data_dir} es contrato v{version}; este script entrena v{CONTRACT_VERSION}. "
            f"Convertí con convert_v2_to_v3.py o reprocesá con preprocess_tasks.py.")

    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy")
    ruta_suj = data_dir / "subjects.npy"
    if not ruta_suj.exists():
        raise SystemExit(f"Falta {ruta_suj}: el split por sujeto es obligatorio.")
    sujetos = np.load(ruta_suj)

    if X.shape[1] != FRAMES or X.shape[2] != COORDS:
        raise SystemExit(
            f"Forma inesperada {X.shape}: se esperaba (n, {FRAMES}, {COORDS}).")

    clases = sorted(set(y.tolist()))
    expected = list(range(1, len(catalog) + 1))
    if clases != expected:
        raise SystemExit(
            "Las etiquetas del dataset no son las clases oficiales 1..64. "
            f"Llegaron {clases}; no se puede asociar un índice a una glosa con seguridad.")
    y0 = (y.astype(np.int32) - 1)                    # etiquetas 1..64 -> 0..63
    return X, y0, sujetos, clases


class AugmentedSequence(keras.utils.PyDataset):
    """Lotes de entrenamiento con una augmentacion nueva por epoca.

    La semilla de cada lote sale de (seed, epoca, lote): el entrenamiento es
    reproducible y a la vez cada epoca ve variantes distintas. Reemplaza a
    las 2 copias fijas de v2, que el modelo terminaba memorizando.

    Con workers > 1 la augmentacion corre en un pool de procesos persistente
    que prepara la epoca siguiente mientras se entrena la actual (numpy con
    hilos queda limitado por el GIL; y los workers de Keras en Windows
    vuelven a crear procesos en cada epoca). El resultado es identico al de
    workers=1: las semillas no dependen del proceso.
    """

    def __init__(self, X, y, n_clases, batch, augmenter: Augmenter | None,
                 seed: int, workers: int = 1):
        super().__init__(workers=1, use_multiprocessing=False)
        self.X, self.y = X, y
        self.n_clases = n_clases
        self.batch = batch
        self.augmenter = augmenter
        self.seed = seed
        self.epoch = 0
        self.orden = self._orden(0)
        self.pool = None
        self._pendientes: dict[int, object] = {}
        self._lotes: list[np.ndarray] | None = None
        if augmenter is not None and workers > 1:
            import multiprocessing as mp
            from augment import worker_init
            self.pool = mp.get_context("spawn").Pool(
                workers, initializer=worker_init, initargs=(X, augmenter.cfg))
            self._lanzar(0)

    def _orden(self, epoca):
        return np.random.default_rng([self.seed, epoca]).permutation(len(self.X))

    def _lanzar(self, epoca):
        from augment import worker_batch
        orden = self._orden(epoca)
        tareas = [(orden[i * self.batch:(i + 1) * self.batch], (self.seed, epoca, i))
                  for i in range(len(self))]
        self._pendientes[epoca] = self.pool.map_async(worker_batch, tareas)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch)

    def __getitem__(self, i):
        idx = self.orden[i * self.batch:(i + 1) * self.batch]
        if self.augmenter is None:
            xb = self.X[idx]
        elif self.pool is not None:
            if self._lotes is None:
                self._lotes = self._pendientes.pop(self.epoch).get()
                self._lanzar(self.epoch + 1)        # prepara la siguiente
            xb = self._lotes[i]
        else:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            xb = np.stack([self.augmenter(x, rng) for x in self.X[idx]])
        return xb.astype(np.float32), keras.utils.to_categorical(self.y[idx], self.n_clases)

    def on_epoch_end(self):
        self.epoch += 1
        self.orden = self._orden(self.epoch)
        self._lotes = None

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

N_POINTS = COORDS // 3
N_HAND = 21
POSE_START = 2 * N_HAND
MIDDLE_MCP = 9            # landmark 9 de la mano: nudillo del dedo medio


@keras.saving.register_keras_serializable(package="eva")
class FeaturesEva(layers.Layer):
    """Deriva features dentro del grafo. Usa keras.ops para correr igual con
    backend torch (entrenamiento en GPU) y tensorflow (exportación); con
    tensorflow se convierten a ops builtin de TFLite.

    Entrada (B, 40, 168) contrato v3. Salida (B, 40, 462):
      - posiciones v3, con la z de la pose en cero: MediaPipe la estima mal
        (en LSA64 llega a -7 anchos de hombro) y depende de la cámara;
      - cada mano relativa a su muñeca y dividida por el tamaño de palma
        (muñeca -> nudillo medio): la forma de la mano sin importar dónde
        esté ni su tamaño. Cero si la mano no está;
      - velocidades x[t] - x[t-1] (0 en t=0 y donde falte el punto en alguno
        de los dos cuadros), escaladas para quedar en el rango de las demás.
    """

    VEL_SCALE = 5.0

    def call(self, x):
        p = ops.reshape(x, (-1, FRAMES, N_POINTS, 3))
        present = ops.cast(ops.max(ops.abs(p), axis=-1, keepdims=True) > 0.0, "float32")

        pose_z_mask = np.ones((1, 1, N_POINTS, 3), dtype=np.float32)
        pose_z_mask[:, :, POSE_START:, 2] = 0.0
        pos = p * ops.convert_to_tensor(pose_z_mask)

        manos = []
        for inicio in (0, N_HAND):
            h = pos[:, :, inicio:inicio + N_HAND]
            wrist = h[:, :, 0:1]
            mcp = h[:, :, MIDDLE_MCP:MIDDLE_MCP + 1, 0:2]
            palma = ops.sqrt(ops.sum(ops.square(mcp - wrist[..., 0:2]), axis=-1, keepdims=True))
            h_pres = present[:, :, inicio:inicio + 1]
            local = (h - wrist) / (palma + 1e-3) * h_pres
            manos.append(ops.reshape(local, (-1, FRAMES, N_HAND * 3)))

        ambos = present[:, 1:] * present[:, :-1]
        vel = (pos[:, 1:] - pos[:, :-1]) * ambos * self.VEL_SCALE
        vel = ops.concatenate([ops.zeros_like(vel[:, :1]), vel], axis=1)

        return ops.concatenate([
            ops.reshape(pos, (-1, FRAMES, COORDS)),
            manos[0], manos[1],
            ops.reshape(vel, (-1, FRAMES, COORDS)),
        ], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], FRAMES, COORDS * 2 + N_HAND * 3 * 2)


def construir_lstm(n_clases: int, hidden: int, dropout: float):
    """La LSTM de v2 sobre las features nuevas. unroll=True evita TensorList
    (que exigiria Flex) y requiere largo fijo, que es nuestro caso."""
    entrada = layers.Input(shape=(FRAMES, COORDS), name="keypoints")
    x = FeaturesEva(name="features")(entrada)
    x = layers.Dense(hidden, activation="swish")(x)
    x = layers.LSTM(hidden, return_sequences=True, unroll=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden, unroll=True)(x)
    x = layers.Dropout(dropout)(x)
    salida = layers.Dense(n_clases, name="logits")(x)
    return keras.Model(entrada, salida, name="eva_lstm")


def _bloque_conv(x, dim, kernel, dropout):
    """Conv1D depthwise con expansion (bloque del 1er puesto de GISLR)."""
    r = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(dim * 2, activation="swish")(x)
    x = layers.DepthwiseConv1D(kernel, padding="same")(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(dim)(x)
    x = layers.Dropout(dropout)(x)
    return layers.Add()([r, x])


def _bloque_transformer(x, dim, heads, dropout):
    r = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.MultiHeadAttention(heads, dim // heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([r, x])
    r = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(dim * 2, activation="swish")(x)
    x = layers.Dense(dim)(x)
    x = layers.Dropout(dropout)(x)
    return layers.Add()([r, x])


def construir_conv_tx(n_clases: int, dim: int, dropout: float):
    entrada = layers.Input(shape=(FRAMES, COORDS), name="keypoints")
    x = FeaturesEva(name="features")(entrada)
    x = layers.Dense(dim)(x)
    # Posicion temporal aprendida: largo fijo de 40 cuadros.
    x = PosicionTemporal(name="posicion")(x)
    for _ in range(3):
        x = _bloque_conv(x, dim, 11, dropout)
    x = _bloque_transformer(x, dim, 4, dropout)
    for _ in range(3):
        x = _bloque_conv(x, dim, 11, dropout)
    x = _bloque_transformer(x, dim, 4, dropout)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    salida = layers.Dense(n_clases, name="logits")(x)
    return keras.Model(entrada, salida, name="eva_conv_tx")


@keras.saving.register_keras_serializable(package="eva")
class PosicionTemporal(layers.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(shape=(1, FRAMES, input_shape[-1]),
                                   initializer="zeros", name="pos")

    def call(self, x):
        return x + self.pos


def construir(args, n_clases: int):
    if args.arch == "lstm":
        return construir_lstm(n_clases, args.hidden, args.dropout)
    return construir_conv_tx(n_clases, args.dim, args.dropout)


def compilar(modelo, args, pasos_por_epoca: int, epocas: int):
    total = max(1, pasos_por_epoca * epocas)
    warmup = min(total // 10, pasos_por_epoca * 5)
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr * 0.01, decay_steps=total - warmup,
        alpha=0.01, warmup_target=args.lr, warmup_steps=warmup)
    modelo.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=args.weight_decay, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True,
                                                  label_smoothing=args.label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def entrenar(args, Xtr, ytr, Xva, yva, n_clases, epocas, seed, early_stopping):
    """Entrena un modelo. Devuelve (modelo, historia)."""
    keras.utils.set_random_seed(seed)
    augmenter = None if args.no_augment else Augmenter(augment_config(args))
    datos = AugmentedSequence(Xtr, ytr, n_clases, args.batch, augmenter, seed,
                              workers=args.workers)
    modelo = construir(args, n_clases)
    compilar(modelo, args, len(datos), epocas)
    callbacks = []
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.paciencia,
            restore_best_weights=True, verbose=1))
    val = None
    if Xva is not None and len(Xva):
        val = (Xva, keras.utils.to_categorical(yva, n_clases))
    try:
        hist = modelo.fit(datos, validation_data=val, epochs=epocas,
                          verbose=2 if args.verbose else 0, callbacks=callbacks)
    finally:
        datos.close()
    return modelo, hist.history


def augment_config(args) -> AugmentConfig:
    return AugmentConfig(p_yaw=args.p_yaw)


# --------------------------------------------------------------------------
# Exportacion
# --------------------------------------------------------------------------

FLEX_HINT = ("flex", "tensorlist")


def exportar_tflite(modelo, destino: Path) -> bytes:
    """Convierte exigiendo ops nativas. Si necesitara Flex, aborta:
    preferimos enterarnos acá y no en el dispositivo."""
    tf = _tf()
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
    it = _tf().lite.Interpreter(model_content=blob)
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


def escribir_fixture(destino: Path, blob: bytes, X, y, catalog, n=8, seed=0):
    """Fixture para el test instrumentado de Android (ModelReferenceTest).

    Permite verificar el modelo en el dispositivo SIN camara ni MediaPipe.
    Si este test pasa, el despliegue del modelo esta resuelto y cualquier
    falla posterior es del pipeline de captura.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    it = _tf().lite.Interpreter(model_content=blob)
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
            "claseEsperada": pred,
            "senaEsperada": catalog[pred] if pred < len(catalog) else f"clase_{pred}",
            "etiquetaReal": int(y[i]),
            "acierta": bool(pred == int(y[i])),
            "logitsEsperados": [round(float(v), 5) for v in logits],
            "secuencia": [[round(float(v), 6) for v in frame] for frame in x],
        })

    destino.write_text(json.dumps({
        "descripcion": "Fixture para verificar el modelo en Android sin camara.",
        "contractVersion": CONTRACT_VERSION,
        "frames": FRAMES,
        "coordenadas": COORDS,
        "toleranciaLogits": 1e-3,
        "casos": casos,
    }, ensure_ascii=False), encoding="utf-8")
    return len(casos)


def exportar(modelo, args, out_dir: Path, X_ref, y_ref, catalog, catalog_version,
             n_clases, metricas: dict):
    """Exporta el modelo. Con backend torch delega en un subproceso con
    backend tensorflow (el conversor TFLite necesita un modelo TF)."""
    if keras.backend.backend() != "tensorflow":
        exportar_via_tensorflow(modelo, args, out_dir, X_ref, y_ref, n_clases, metricas)
        return
    exportar_tf(modelo, args, out_dir, X_ref, y_ref, catalog, catalog_version,
                n_clases, metricas)


N_REF_BACKENDS = 64


def exportar_via_tensorflow(modelo, args, out_dir: Path, X_ref, y_ref, n_clases, metricas):
    """Guarda pesos y salidas de referencia y exporta en un proceso con
    KERAS_BACKEND=tensorflow. El hijo reconstruye la misma arquitectura, carga
    los pesos y exige que sus salidas coincidan con las del modelo entrenado
    antes de convertir: si no coinciden, el .tflite no sería este modelo."""
    trabajo = out_dir / "_exportacion"
    trabajo.mkdir(parents=True, exist_ok=True)
    modelo.save_weights(trabajo / "pesos.weights.h5")
    np.savez(trabajo / "ref.npz", X=X_ref, y=y_ref,
             logits=modelo.predict(X_ref[:N_REF_BACKENDS], verbose=0))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (trabajo / "job.json").write_text(json.dumps({
        "args": config, "nClases": n_clases, "metricas": metricas,
    }, ensure_ascii=False), encoding="utf-8")

    print("\n--- Exportacion: subproceso con backend tensorflow ---", flush=True)
    env = {**os.environ, "KERAS_BACKEND": "tensorflow"}
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--exportar-desde", str(trabajo)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"La exportacion fallo (codigo {r.returncode}). "
                         f"Pesos conservados en {trabajo}")
    shutil.move(str(trabajo / "pesos.weights.h5"), str(out_dir / "modelo.weights.h5"))
    shutil.rmtree(trabajo)


def exportar_desde_pesos(trabajo: Path) -> int:
    """Proceso hijo de exportar_via_tensorflow (backend tensorflow)."""
    if keras.backend.backend() != "tensorflow":
        raise SystemExit("--exportar-desde requiere KERAS_BACKEND=tensorflow")
    job = json.loads((trabajo / "job.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(**job["args"])
    args.catalog = Path(args.catalog)
    n_clases = job["nClases"]
    catalog_version, catalog = load_catalog(args.catalog)
    ref = np.load(trabajo / "ref.npz")

    modelo = construir(args, n_clases)
    modelo.load_weights(trabajo / "pesos.weights.h5")
    X_ref = ref["X"]
    tf_logits = modelo.predict(X_ref[:N_REF_BACKENDS], verbose=0)
    diff = float(np.abs(tf_logits - ref["logits"]).max())
    iguales = int((tf_logits.argmax(1) == ref["logits"].argmax(1)).sum())
    print(f"  equivalencia entrenado vs tensorflow: maxdiff {diff:.2e}, "
          f"misma prediccion {iguales}/{len(tf_logits)}")
    if diff > 1e-3 or iguales != len(tf_logits):
        raise SystemExit("El modelo reconstruido en tensorflow no coincide con el entrenado.")

    metricas = job["metricas"]
    metricas["equivalenciaBackendsMaxDiff"] = diff
    metricas["tfVersion"] = _tf().__version__
    exportar_tf(modelo, args, trabajo.parent, X_ref, ref["y"], catalog, catalog_version,
                n_clases, metricas)
    return 0


def exportar_tf(modelo, args, out_dir: Path, X_ref, y_ref, catalog, catalog_version,
                n_clases, metricas: dict):
    print("\n--- Exportacion a TFLite ---")
    ruta_tflite = out_dir / "modelo_lsa.tflite"
    blob = exportar_tflite(modelo, ruta_tflite)
    print(f"  {ruta_tflite.name}: {len(blob)/1024:.0f} KB")

    maxdiff, ok, tot = verificar_equivalencia(modelo, blob, X_ref)
    print(f"  equivalencia Keras vs TFLite: maxdiff {maxdiff:.2e}, "
          f"misma prediccion {ok}/{tot}")
    if maxdiff > 1e-3 or ok != tot:
        raise SystemExit("El .tflite NO es equivalente al modelo entrenado.")

    n_fix = escribir_fixture(out_dir / "fixture_android.json", blob, X_ref, y_ref, catalog)
    print(f"  fixture_android.json: {n_fix} casos")

    catalog_bytes = json.dumps({"version": catalog_version, "glosas": catalog},
                               ensure_ascii=False, indent=2).encode("utf-8")
    (out_dir / "catalogo_senas.json").write_bytes(catalog_bytes)
    (out_dir / "lsa-manifest.json").write_text(json.dumps({
        "modelVersion": args.model_version,
        "catalogVersion": catalog_version,
        "modelSha256": hashlib.sha256(blob).hexdigest(),
        "catalogSha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "numClasses": n_clases,
        "outputsProbabilities": False,
        "frames": FRAMES,
        "coords": COORDS,
        "contractVersion": CONTRACT_VERSION,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    metricas.update({
        "modelVersion": args.model_version,
        "tamanoKB": round(len(blob) / 1024),
        "equivalenciaMaxDiff": maxdiff,
    })
    (out_dir / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nListo. Archivos en {out_dir.resolve()}")
    print("  modelo_lsa.tflite, catalogo_senas.json y lsa-manifest.json -> app/src/main/assets/lsa/")
    print("  fixture_android.json   -> app/src/androidTest/assets/")


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------

def evaluar(modelo, X, y, n_clases, catalog):
    logits = modelo.predict(X, verbose=0, batch_size=256)
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    acc = float((pred == y).mean())
    acc5 = float(np.mean([y[i] in top5[i] for i in range(len(y))]))
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for real, p in zip(y, pred):
        cm[real, p] += 1
    confusiones = sorted(
        ((int(cm[i, j]), catalog[i], catalog[j])
         for i in range(n_clases) for j in range(n_clases) if i != j and cm[i, j] > 0),
        reverse=True)
    por_clase = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return {
        "accuracy": round(acc, 4),
        "top5": round(acc5, 4),
        "clasesBajo50": [catalog[i] for i in np.flatnonzero(por_clase < 0.5)],
        "paresMasConfundidos": [
            {"veces": c, "real": a, "predicho": b} for c, a, b in confusiones[:15]],
    }


def hiperparametros(args) -> dict:
    return {
        "arch": args.arch, "lr": args.lr, "batch": args.batch,
        "hidden": args.hidden, "dim": args.dim, "dropout": args.dropout,
        "weightDecay": args.weight_decay, "labelSmoothing": args.label_smoothing,
        "epochs": args.epochs, "paciencia": args.paciencia,
        "augment": None if args.no_augment else asdict(augment_config(args)),
    }


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="./exports")
    ap.add_argument("--arch", choices=("lstm", "conv_tx"), default="conv_tx")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128, help="lstm: unidades")
    ap.add_argument("--dim", type=int, default=192, help="conv_tx: ancho")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--p-yaw", type=float, default=0.0,
                    help="probabilidad de giro sobre el eje vertical (A/B)")
    ap.add_argument("--paciencia", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6,
                    help="procesos para la augmentacion (1: en el proceso principal)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    ap.add_argument("--model-version", default="eva-lsa64-v3")
    ap.add_argument("--val-subjects", nargs="*", type=int, default=[9])
    ap.add_argument("--test-subjects", nargs="*", type=int, default=[10])
    ap.add_argument("--cv", choices=("loso",), default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="con --cv: sujetos a dejar afuera (default: todos)")
    ap.add_argument("--final", action="store_true",
                    help="entrena con todos los sujetos --epochs epocas y exporta")
    ap.add_argument("--exportar-desde", type=Path, default=None,
                    help=argparse.SUPPRESS)  # uso interno: exportar_via_tensorflow
    args = ap.parse_args()
    if args.exportar_desde:
        return exportar_desde_pesos(args.exportar_desde)
    if not args.data:
        ap.error("falta --data")

    data_dir, out_dir = Path(args.data), Path(args.out)
    catalog_version, catalog = load_catalog(args.catalog)
    X, y, sujetos, clases = cargar(data_dir, catalog)
    n_clases = len(clases)
    todos = sorted(set(sujetos.tolist()))
    print(f"Datos: X={X.shape}  clases={n_clases}  sujetos={todos}  arch={args.arch}  "
          f"entorno={entorno_entrenamiento()}", flush=True)

    # ---------------- Validacion cruzada dejando un sujeto afuera ----------------
    if args.cv:
        folds = args.folds or todos
        curvas, resultados = [], []
        # Cada fold se guarda al terminar: si el proceso se corta, al
        # relanzar el mismo comando se retoman solo los folds pendientes.
        etiqueta = f"{args.arch}{'_noaug' if args.no_augment else ''}{'_yaw' if args.p_yaw else ''}"
        partes = out_dir / f"cv_{etiqueta}_folds"
        partes.mkdir(parents=True, exist_ok=True)
        for s in folds:
            parte = partes / f"sujeto_{int(s):02d}_ep{args.epochs}.json"
            if parte.exists():
                r = json.loads(parte.read_text(encoding="utf-8"))
                curvas.append(r.pop("curva"))
                resultados.append(r)
                print(f"  sujeto {s:2d}: retomado de {parte.name} (final {r['accuracy']:.4f})")
                continue
            t0 = time.time()
            m_te = sujetos == s
            modelo, hist = entrenar(args, X[~m_te], y[~m_te], X[m_te], y[m_te],
                                    n_clases, args.epochs, args.seed, early_stopping=False)
            curva = hist["val_accuracy"]
            curvas.append(curva)
            r = evaluar(modelo, X[m_te], y[m_te], n_clases, catalog)
            r.update({"sujeto": int(s), "mejorEpoca": int(np.argmax(curva)) + 1,
                      "mejorVal": round(float(max(curva)), 4),
                      "minutos": round((time.time() - t0) / 60, 1)})
            resultados.append(r)
            parte.write_text(json.dumps({**r, "curva": [float(v) for v in curva]},
                                        ensure_ascii=False), encoding="utf-8")
            print(f"  sujeto {s:2d}: final {r['accuracy']:.4f}  top5 {r['top5']:.4f}  "
                  f"mejor {r['mejorVal']:.4f} (ep {r['mejorEpoca']})  {r['minutos']} min")
            keras.backend.clear_session()

        media = np.mean(np.array(curvas), axis=0)
        ep = int(np.argmax(media)) + 1
        finales = [r["accuracy"] for r in resultados]
        print(f"\nLOSO {args.arch}: final {np.mean(finales):.4f} ± {np.std(finales):.4f}  "
              f"| curva media max {media.max():.4f} en epoca {ep}")
        out_dir.mkdir(parents=True, exist_ok=True)
        nombre = f"cv_{etiqueta}.json"
        (out_dir / nombre).write_text(json.dumps({
            "arch": args.arch, "contractVersion": CONTRACT_VERSION,
            "accuracyFinalMedia": round(float(np.mean(finales)), 4),
            "accuracyFinalDesvio": round(float(np.std(finales)), 4),
            "curvaMediaMax": round(float(media.max()), 4),
            "epocaSugerida": ep,
            "curvaMedia": [round(float(v), 4) for v in media],
            "folds": resultados,
            "hiperparametros": hiperparametros(args),
            "entorno": entorno_entrenamiento(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Guardado en {out_dir / nombre}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Modelo final con todos los sujetos ----------------
    if args.final:
        modelo, hist = entrenar(args, X, y, None, None, n_clases, args.epochs,
                                args.seed, early_stopping=False)
        metricas = {
            "modo": "final_todos_los_sujetos",
            "contractVersion": CONTRACT_VERSION,
            "sujetos": todos,
            "epocas": args.epochs,
            "trainAccuracyFinal": round(float(hist["accuracy"][-1]), 4),
            "nota": "Sin test propio: la estimacion de generalizacion sale del CV LOSO "
                    "y de evaluar_videos.py sobre grabaciones del telefono.",
            "hiperparametros": hiperparametros(args),
            "catalogo": {"version": catalog_version, "archivo": str(args.catalog)},
            "entorno": entorno_entrenamiento(),
        }
        exportar(modelo, args, out_dir, X, y, catalog, catalog_version, n_clases, metricas)
        return 0

    # ---------------- Split por sujeto (train / val / test) ----------------
    val_s, test_s = set(args.val_subjects), set(args.test_subjects)
    if val_s & test_s:
        raise SystemExit("val-subjects y test-subjects no pueden solaparse.")
    m_te = np.isin(sujetos, list(test_s))
    m_va = np.isin(sujetos, list(val_s))
    m_tr = ~(m_te | m_va)
    print(f"Split POR SUJETO -> train={m_tr.sum()}  val={m_va.sum()} (suj {sorted(val_s)})  "
          f"test={m_te.sum()} (suj {sorted(test_s)})")
    modelo, hist = entrenar(args, X[m_tr], y[m_tr], X[m_va], y[m_va], n_clases,
                            args.epochs, args.seed, early_stopping=True)
    print(f"Parametros: {modelo.count_params():,}")

    mejor_val = float(max(hist["val_accuracy"]))
    mejor_ep = int(np.argmax(hist["val_accuracy"])) + 1
    test = evaluar(modelo, X[m_te], y[m_te], n_clases, catalog)
    print("\n=== Resultado ===")
    print(f"mejor val acc: {mejor_val:.4f} (epoca {mejor_ep})")
    print(f"TEST acc: {test['accuracy']:.4f}  top5: {test['top5']:.4f}")
    print(f"(baseline al azar con {n_clases} clases: {1/n_clases:.4f})")
    if test["paresMasConfundidos"]:
        print("\nPares mas confundidos (real -> predicho):")
        for c in test["paresMasConfundidos"][:10]:
            print(f"  {c['veces']}x  {c['real']} -> {c['predicho']}")

    metricas = {
        "modo": "split_por_sujeto",
        "contractVersion": CONTRACT_VERSION,
        "testAccuracy": test["accuracy"],
        "testTop5": test["top5"],
        "valAccuracyMejor": round(mejor_val, 4),
        "epocaMejor": mejor_ep,
        "baselineAzar": round(1 / n_clases, 4),
        "sujetosVal": sorted(val_s),
        "sujetosTest": sorted(test_s),
        "seed": args.seed,
        "parametros": int(modelo.count_params()),
        "hiperparametros": hiperparametros(args),
        "catalogo": {"version": catalog_version, "archivo": str(args.catalog)},
        "entorno": entorno_entrenamiento(),
        "clasesBajo50": test["clasesBajo50"],
        "paresMasConfundidos": test["paresMasConfundidos"],
    }
    exportar(modelo, args, out_dir, X[m_te], y[m_te], catalog, catalog_version,
             n_clases, metricas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
