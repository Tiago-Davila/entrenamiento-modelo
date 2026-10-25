#!/usr/bin/env python3
"""
train_keras_tflite.py — Entrena el clasificador Eva LSA-94 (94 señas,
contrato v3) en Keras y lo exporta a .tflite listo para LiteRT en Android,
junto con los fixtures de verificacion.

Por que Keras y no PyTorch: la conversion PyTorch -> .tflite pasa por ONNX y
es fragil. Keras convierte de forma nativa. Toda capa debe convertir con ops
builtin (sin Flex): exportar_tflite() aborta si no.

Base (de Eva LSA64 v3): augmentacion nueva en cada epoca, features dentro del
grafo, arquitecturas 'lstm' y 'conv_tx', AdamW + coseno + label smoothing.

Novedades de LSA-94:
  - Catalogo de 94 clases en grupos (lsa64, temporales, abecedario); las
    metricas y confusiones se reportan por grupo.
  - FeaturesEva94 (--features forma): agrega descriptores de forma de mano
    (distancias entre dedos y flexion de articulaciones) para separar letras
    y señas parecidas. Android sigue entregando solo el tensor 40x168.
  - Muestreo balanceado por clase (--balance): las clases nuevas tienen 4
    muestras contra 54 de LSA64.
  - Perfil de augmentacion fuerte para clases de un solo señante.
  - Modos:
      split (default)  val = sujeto 9 + rep 3 del 012; test = sujeto 10 +
                       rep 4 del 012.
      --cv loso        deja afuera un sujeto de LSA64 por vez (mide 1-64
                       entre personas). El 012 queda siempre en train.
      --cv reps        deja afuera una repeticion del sujeto 012 por vez: la
                       unica medida posible de las clases que solo grabo el.
      --final          entrena con TODO una cantidad fija de epocas (la
                       elegida en los CV) y exporta.

Salidas en el directorio --out:
    modelo_lsa.tflite     -> app/src/main/assets/lsa/
    catalogo_senas.json   -> app/src/main/assets/lsa/ (acoplado al modelo)
    lsa-manifest.json     -> app/src/main/assets/lsa/
    fixture_android.json  -> app/src/androidTest/assets/
    metricas.json         -> registro de experimentos

Uso (ver EVA_LSA94_ENTRENAMIENTO.md):
    python train_keras_tflite.py --data ./data/processed_94 --out ./runs/split
    python train_keras_tflite.py --data ./data/processed_94 --out ./runs --cv reps --epochs 100
    python train_keras_tflite.py --data ./data/processed_94 --out ./runs --cv loso --epochs 100
    python train_keras_tflite.py --data ./data/processed_94 --final --epochs <N> --out ./exports
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
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras import layers, ops
from augment import AugmentConfig, Augmenter, perfil_un_senante
from eva_contract import (CATALOG_PATH, CONTRACT_VERSION, COORDS, FRAMES, load_catalog,
                          load_groups)


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
    """Devuelve X, y (0..N-1), sujetos, repeticiones y clases (1..N)."""
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
    reps = np.load(data_dir / "reps.npy")

    if X.shape[1] != FRAMES or X.shape[2] != COORDS:
        raise SystemExit(
            f"Forma inesperada {X.shape}: se esperaba (n, {FRAMES}, {COORDS}).")

    clases = sorted(set(y.tolist()))
    expected = list(range(1, len(catalog) + 1))
    if clases != expected:
        raise SystemExit(
            f"Las etiquetas del dataset no son las clases oficiales 1..{len(catalog)}. "
            f"Llegaron {clases}; no se puede asociar un índice a una glosa con seguridad.")
    y0 = (y.astype(np.int32) - 1)                    # etiquetas 1..N -> 0..N-1
    return X, y0, sujetos, reps, clases


def indices_balanceados(y: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Orden de una época con cuota por clase proporcional a n_c**alpha.

    alpha=1 es la distribución natural; alpha=0 da la misma cantidad a cada
    clase. El total por época sigue siendo len(y). Dentro de una clase cada
    muestra se usa floor(k/n) veces y el resto se elige al azar sin repetir,
    así ninguna muestra queda sistemáticamente afuera.
    """
    clases, conteo = np.unique(y, return_counts=True)
    peso = conteo.astype(np.float64) ** alpha
    cuota = peso / peso.sum() * len(y)
    k = np.floor(cuota).astype(np.int64)
    # Lo que falta para llegar a len(y) va a las fracciones más grandes.
    faltan = len(y) - int(k.sum())
    k[np.argsort(-(cuota - k), kind="stable")[:faltan]] += 1
    partes = []
    for c, kc in zip(clases, k):
        idx = np.flatnonzero(y == c)
        vueltas, extra = divmod(int(kc), len(idx))
        partes.append(np.tile(idx, vueltas))
        partes.append(rng.choice(idx, size=extra, replace=False))
    return rng.permutation(np.concatenate(partes))


def perfil_por_muestra(y: np.ndarray, sujetos: np.ndarray) -> np.ndarray:
    """1 para las muestras de clases que en este train tienen un solo señante."""
    unico = {int(c) for c in np.unique(y) if len(np.unique(sujetos[y == c])) == 1}
    return np.array([1 if int(c) in unico else 0 for c in y], dtype=np.int8)


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
                 seed: int, workers: int = 1, balance: float = 1.0,
                 augmenter_fuerte: Augmenter | None = None,
                 perfil: np.ndarray | None = None):
        super().__init__(workers=1, use_multiprocessing=False)
        self.X, self.y = X, y
        self.n_clases = n_clases
        self.batch = batch
        self.augmenter = augmenter
        self.augs = (augmenter, augmenter_fuerte or augmenter)
        self.perfil = perfil
        self.seed = seed
        self.balance = balance
        self.epoch = 0
        self.orden = self._orden(0)
        self.pool = None
        self._pendientes: dict[int, object] = {}
        self._lotes: list[np.ndarray] | None = None
        if augmenter is not None and workers > 1:
            import multiprocessing as mp
            from augment import worker_init
            self.pool = mp.get_context("spawn").Pool(
                workers, initializer=worker_init,
                initargs=(X, augmenter.cfg, self.augs[1].cfg, perfil))
            self._lanzar(0)

    def _orden(self, epoca):
        rng = np.random.default_rng([self.seed, epoca])
        if self.balance >= 1.0:
            return rng.permutation(len(self.X))         # idéntico a v3
        return indices_balanceados(self.y, self.balance, rng)

    def _lanzar(self, epoca):
        from augment import worker_batch
        orden = self._orden(epoca)
        tareas = [(orden[i * self.batch:(i + 1) * self.batch], (self.seed, epoca, i))
                  for i in range(len(self))]
        self._pendientes[epoca] = self.pool.map_async(worker_batch, tareas)

    def __len__(self):
        return math.ceil(len(self.orden) / self.batch)

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
            from augment import aumentar_lote
            rng = np.random.default_rng([self.seed, self.epoch, i])
            xb = aumentar_lote(self.X, idx, self.augs, self.perfil, rng)
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


TIPS = (4, 8, 12, 16, 20)
PARES_PUNTAS = tuple((a, b) for i, a in enumerate(TIPS) for b in TIPS[i + 1:])
# Pulgar contra las articulaciones de los otros dedos: separa A/E/S/M/N/T,
# que difieren en dónde apoya el pulgar.
PARES_PULGAR = tuple((4, j) for j in (5, 6, 9, 10, 13, 14, 17, 18))
PARES_DIST = PARES_PUNTAS + PARES_PULGAR                       # 18
# Articulaciones (a, b, c): ángulo en b entre a->b y b->c. Cada dedo arranca
# en la muñeca: 3 articulaciones por dedo, 15 por mano.
CADENAS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
           (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
ARTICULACIONES = tuple((c[k], c[k + 1], c[k + 2]) for c in CADENAS for k in range(3))
N_FORMA = len(PARES_DIST) + len(ARTICULACIONES)                # 33 por mano


@keras.saving.register_keras_serializable(package="eva")
class FeaturesEva94(FeaturesEva):
    """FeaturesEva + descriptores de forma de mano, para señas parecidas.

    Por mano (33): distancias entre puntas y del pulgar a las articulaciones de
    los otros dedos, divididas por la palma; y coseno de flexión de las 15
    articulaciones. Son invariantes a dónde está la mano y a su giro, así que
    el modelo no tiene que deducir la configuración de los dedos a partir de
    coordenadas. Cero si la mano no está. Todo con gather y aritmética:
    convierte a ops builtin de TFLite.

    Salida (B, 40, 462 + 66).
    """

    def _forma(self, h, h_pres):
        def puntos(indices):
            return ops.take(h, np.array(indices, dtype=np.int32), axis=2)

        wrist = h[:, :, 0:1]
        mcp = h[:, :, MIDDLE_MCP:MIDDLE_MCP + 1]
        palma = ops.sqrt(ops.sum(ops.square(mcp[..., 0:2] - wrist[..., 0:2]), axis=-1))
        a = puntos([p for p, _ in PARES_DIST])
        b = puntos([q for _, q in PARES_DIST])
        dist = ops.sqrt(ops.sum(ops.square(a - b), axis=-1) + 1e-8) / (palma + 1e-3)

        p0 = puntos([i for i, _, _ in ARTICULACIONES])
        p1 = puntos([j for _, j, _ in ARTICULACIONES])
        p2 = puntos([k for _, _, k in ARTICULACIONES])
        u, v = p1 - p0, p2 - p1
        norma = ops.sqrt(ops.sum(ops.square(u), axis=-1) * ops.sum(ops.square(v), axis=-1) + 1e-8)
        coseno = ops.sum(u * v, axis=-1) / norma

        return ops.concatenate([dist, coseno], axis=-1) * h_pres[..., 0]

    def call(self, x):
        base = super().call(x)
        p = ops.reshape(x, (-1, FRAMES, N_POINTS, 3))
        present = ops.cast(ops.max(ops.abs(p), axis=-1, keepdims=True) > 0.0, "float32")
        formas = [self._forma(p[:, :, i:i + N_HAND], present[:, :, i:i + 1])
                  for i in (0, N_HAND)]
        return ops.concatenate([base] + formas, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], FRAMES, COORDS * 2 + N_HAND * 3 * 2 + N_FORMA * 2)


def capa_features(features: str):
    return FeaturesEva94(name="features") if features == "forma" else FeaturesEva(name="features")


def construir_lstm(n_clases: int, hidden: int, dropout: float, features: str = "forma"):
    """La LSTM de v2 sobre las features nuevas. unroll=True evita TensorList
    (que exigiria Flex) y requiere largo fijo, que es nuestro caso."""
    entrada = layers.Input(shape=(FRAMES, COORDS), name="keypoints")
    x = capa_features(features)(entrada)
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


def construir_conv_tx(n_clases: int, dim: int, dropout: float, features: str = "forma"):
    entrada = layers.Input(shape=(FRAMES, COORDS), name="keypoints")
    x = capa_features(features)(entrada)
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
        return construir_lstm(n_clases, args.hidden, args.dropout, args.features)
    return construir_conv_tx(n_clases, args.dim, args.dropout, args.features)


def compilar(modelo, args, pasos_por_epoca: int, epocas: int):
    total = max(1, pasos_por_epoca * epocas)
    warmup = min(total // 10, pasos_por_epoca * 5)
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr * 0.01, decay_steps=total - warmup,
        alpha=0.01, warmup_target=args.lr, warmup_steps=warmup)
    modelo.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=args.weight_decay, clipnorm=1.0,
                                         use_ema=args.ema, ema_momentum=0.995),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True,
                                                  label_smoothing=args.label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def entrenar(args, Xtr, ytr, Xva, yva, n_clases, epocas, seed, early_stopping,
             sujetos_tr=None):
    """Entrena un modelo. Devuelve (modelo, historia).

    sujetos_tr permite detectar las clases de un solo señante, que usan el
    perfil de augmentación fuerte (salvo --sin-perfil-fuerte).
    """
    keras.utils.set_random_seed(seed)
    augmenter = fuerte = perfil = None
    if not args.no_augment:
        augmenter = Augmenter(augment_config(args))
        if sujetos_tr is not None and not args.sin_perfil_fuerte:
            perfil = perfil_por_muestra(ytr, sujetos_tr)
            fuerte = Augmenter(perfil_un_senante(augment_config(args)))
    datos = AugmentedSequence(Xtr, ytr, n_clases, args.batch, augmenter, seed,
                              workers=args.workers, balance=args.balance,
                              augmenter_fuerte=fuerte, perfil=perfil)
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
    cfg = AugmentConfig(p_yaw=args.p_yaw)
    if args.sin_aug_mano:                 # A/B: augmentación v3 sin lo de mano
        cfg = replace(cfg, p_finger_scale=0.0, p_hand_rotate=0.0, p_smooth_jitter=0.0)
    return cfg


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


def indices_fixture(y, grupos: dict, n: int, seed: int = 0) -> np.ndarray:
    """n casos repartidos entre los grupos del catálogo presentes en y, para
    que el test de Android cubra también las clases nuevas."""
    rng = np.random.default_rng(seed)
    por_grupo = [rng.permutation(np.flatnonzero((y >= a - 1) & (y <= b - 1)))
                 for a, b in grupos.values()]
    por_grupo = [g for g in por_grupo if len(g)]
    # Reparto por turnos: si un grupo se queda sin muestras, siguen los otros.
    idx, k = [], 0
    while len(idx) < min(n, len(y)):
        for g in por_grupo:
            if k < len(g) and len(idx) < n:
                idx.append(int(g[k]))
        k += 1
    return np.sort(np.array(idx, dtype=np.int64))


def escribir_fixture(destino: Path, blob: bytes, X, y, catalog, grupos, n=12, seed=0):
    """Fixture para el test instrumentado de Android (ModelReferenceTest).

    Permite verificar el modelo en el dispositivo SIN camara ni MediaPipe.
    Si este test pasa, el despliegue del modelo esta resuelto y cualquier
    falla posterior es del pipeline de captura.
    """
    idx = indices_fixture(y, grupos, n, seed)

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

    n_fix = escribir_fixture(out_dir / "fixture_android.json", blob, X_ref, y_ref, catalog,
                             load_groups(Path(args.catalog)))
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

def _metricas_basicas(logits, y):
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "top5": round(float(np.mean([y[i] in top5[i] for i in range(len(y))])), 4),
        "n": int(len(y)),
    }


def _confusiones(cm, catalog, clases, limite):
    pares = sorted(
        ((int(cm[i, j]), catalog[i], catalog[j])
         for i in clases for j in range(cm.shape[1]) if i != j and cm[i, j] > 0),
        reverse=True)
    return [{"veces": c, "real": a, "predicho": b} for c, a, b in pares[:limite]]


def evaluar(modelo, X, y, n_clases, catalog, grupos: dict):
    """Métricas globales y por grupo del catálogo (lsa64, temporales,
    abecedario). Las confusiones se listan globales y dentro de cada grupo:
    ahí están los pares parecidos que más importan (letras entre sí)."""
    logits = modelo.predict(X, verbose=0, batch_size=256)
    pred = logits.argmax(1)
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for real, p in zip(y, pred):
        cm[real, p] += 1
    presentes = sorted(set(y.tolist()))
    recall = {catalog[c]: round(float(cm[c, c] / cm[c].sum()), 4) for c in presentes}

    por_grupo = {}
    for nombre, (a, b) in grupos.items():
        m = (y >= a - 1) & (y <= b - 1)
        if not m.any():
            continue
        clases_g = [c for c in presentes if a - 1 <= c <= b - 1]
        dentro = np.zeros(n_clases, dtype=bool)
        dentro[a - 1:b] = True
        por_grupo[nombre] = {
            **_metricas_basicas(logits[m], y[m]),
            # una seña del grupo predicha como una de OTRO grupo
            # (p. ej. una letra predicha como seña de LSA64)
            "erroresHaciaOtroGrupo": int(cm[np.ix_(clases_g, np.flatnonzero(~dentro))].sum()),
            "paresMasConfundidos": _confusiones(cm * dentro[None, :], catalog, clases_g, 10),
        }

    return {
        **_metricas_basicas(logits, y),
        "porGrupo": por_grupo,
        "recallPorClase": recall,
        "clasesBajo50": [g for g, r in recall.items() if r < 0.5],
        "paresMasConfundidos": _confusiones(cm, catalog, presentes, 15),
        "matrizConfusion": cm.tolist(),
    }


def imprimir_evaluacion(r: dict, titulo: str):
    print(f"{titulo}: acc {r['accuracy']:.4f}  top5 {r['top5']:.4f}  (n={r['n']})")
    for nombre, g in r["porGrupo"].items():
        print(f"    {nombre:11s} acc {g['accuracy']:.4f}  top5 {g['top5']:.4f}  n={g['n']:4d}  "
              f"errores hacia otro grupo {g['erroresHaciaOtroGrupo']}")
    if r["paresMasConfundidos"]:
        print("  Pares mas confundidos (real -> predicho):")
        for c in r["paresMasConfundidos"][:10]:
            print(f"    {c['veces']}x  {c['real']} -> {c['predicho']}")


def hiperparametros(args) -> dict:
    sin_fuerte = args.no_augment or args.sin_perfil_fuerte
    return {
        "arch": args.arch, "features": args.features, "lr": args.lr, "batch": args.batch,
        "hidden": args.hidden, "dim": args.dim, "dropout": args.dropout,
        "weightDecay": args.weight_decay, "labelSmoothing": args.label_smoothing,
        "epochs": args.epochs, "paciencia": args.paciencia,
        "balance": args.balance, "ema": args.ema, "perfilFuerte": not sin_fuerte,
        "augment": None if args.no_augment else asdict(augment_config(args)),
        "augmentFuerte": None if sin_fuerte else asdict(perfil_un_senante(augment_config(args))),
    }


def parse_reps(valores: list[str]) -> list[tuple[int, int]]:
    """['12:3'] -> [(12, 3)]: la repetición 3 del sujeto 12."""
    out = []
    for v in valores:
        try:
            s, r = v.split(":")
            out.append((int(s), int(r)))
        except ValueError:
            raise SystemExit(f"Formato de repetición inválido '{v}': se espera SUJETO:REP (ej. 12:4)")
    return out


def mascara_reps(sujetos, reps, pares) -> np.ndarray:
    m = np.zeros(len(sujetos), dtype=bool)
    for s, r in pares:
        m |= (sujetos == s) & (reps == r)
    return m


def sujetos_unicos(y, sujetos) -> set[int]:
    """Sujetos que son el único señante de alguna clase: dejarlos afuera en
    LOSO dejaría esas clases sin ninguna muestra de entrenamiento."""
    return {int(np.unique(sujetos[y == c])[0]) for c in np.unique(y)
            if len(np.unique(sujetos[y == c])) == 1}


def etiqueta_experimento(args) -> str:
    if args.etiqueta:
        return args.etiqueta
    partes = [args.arch, args.features, f"bal{args.balance:g}"]
    partes += [n for n, activo in (("noaug", args.no_augment), ("yaw", args.p_yaw),
                                   ("ema", args.ema), ("sinfuerte", args.sin_perfil_fuerte),
                                   ("sinaugmano", args.sin_aug_mano)) if activo]
    return "_".join(partes)


def resumir_por_grupo(resultados: list[dict], grupos: dict) -> dict:
    """Aciertos sumados sobre todos los folds: cada muestra aparece en un solo
    fold, así que es la accuracy sobre el total de muestras evaluadas."""
    por_grupo = {}
    for nombre in grupos:
        gs = [r["porGrupo"][nombre] for r in resultados if nombre in r["porGrupo"]]
        if gs:
            n = sum(g["n"] for g in gs)
            por_grupo[nombre] = {
                "accuracy": round(sum(g["accuracy"] * g["n"] for g in gs) / n, 4),
                "top5": round(sum(g["top5"] * g["n"] for g in gs) / n, 4),
                "n": n,
            }
    return por_grupo


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="./exports")
    ap.add_argument("--arch", choices=("lstm", "conv_tx"), default="conv_tx")
    ap.add_argument("--features", choices=("forma", "v3"), default="forma",
                    help="forma: FeaturesEva94 (+ forma de mano); v3: FeaturesEva de lsa64")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128, help="lstm: unidades")
    ap.add_argument("--dim", type=int, default=192, help="conv_tx: ancho")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--balance", type=float, default=0.5,
                    help="cuota por clase ~ n_c**balance: 1 natural, 0 balanceado")
    ap.add_argument("--ema", action="store_true", help="promedio exponencial de pesos")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--sin-perfil-fuerte", action="store_true",
                    help="clases de un solo señante con la misma augmentación que el resto")
    ap.add_argument("--sin-aug-mano", action="store_true",
                    help="desactiva dedos/giro de mano/deriva (augmentación de v3)")
    ap.add_argument("--p-yaw", type=float, default=0.0,
                    help="probabilidad de giro sobre el eje vertical (A/B)")
    ap.add_argument("--paciencia", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6,
                    help="procesos para la augmentacion (1: en el proceso principal)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    ap.add_argument("--model-version", default="eva-lsa94-v4")
    ap.add_argument("--val-subjects", nargs="*", type=int, default=[9])
    ap.add_argument("--test-subjects", nargs="*", type=int, default=[10])
    ap.add_argument("--val-reps", nargs="*", default=["12:3"],
                    help="repeticiones SUJETO:REP para validación (split)")
    ap.add_argument("--test-reps", nargs="*", default=["12:4"],
                    help="repeticiones SUJETO:REP para test (split)")
    ap.add_argument("--cv", choices=("loso", "reps"), default=None,
                    help="loso: deja un sujeto afuera; reps: deja afuera una repetición "
                         "de --sujeto-reps (única medida de las clases de un señante)")
    ap.add_argument("--sujeto-reps", type=int, default=12)
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="con --cv: sujetos (loso) o repeticiones (reps) a dejar afuera")
    ap.add_argument("--etiqueta", default=None, help="nombre del experimento en --cv")
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
    grupos = load_groups(args.catalog)
    X, y, sujetos, reps, clases = cargar(data_dir, catalog)
    n_clases = len(clases)
    todos = sorted(set(sujetos.tolist()))
    print(f"Datos: X={X.shape}  clases={n_clases}  sujetos={todos}  arch={args.arch}  "
          f"features={args.features}  balance={args.balance}  "
          f"entorno={entorno_entrenamiento()}", flush=True)

    # ---------------- Validacion cruzada ----------------
    if args.cv:
        if args.cv == "loso":
            solos = sujetos_unicos(y, sujetos)
            folds = args.folds or [s for s in todos if s not in solos]
            nombre_fold = "sujeto"

            def mascara(f):
                return sujetos == f
        else:
            m_s = sujetos == args.sujeto_reps
            if not m_s.any():
                raise SystemExit(f"No hay muestras del sujeto {args.sujeto_reps}")
            folds = args.folds or sorted(set(reps[m_s].tolist()))
            nombre_fold = "rep"

            def mascara(f):
                return m_s & (reps == f)

        curvas, resultados = [], []
        # Cada fold se guarda al terminar: si el proceso se corta, al
        # relanzar el mismo comando se retoman solo los folds pendientes.
        etiqueta = f"{args.cv}_{etiqueta_experimento(args)}"
        partes = out_dir / f"cv_{etiqueta}_folds"
        partes.mkdir(parents=True, exist_ok=True)
        for f in folds:
            parte = partes / f"{nombre_fold}_{int(f):02d}_ep{args.epochs}_s{args.seed}.json"
            if parte.exists():
                r = json.loads(parte.read_text(encoding="utf-8"))
                curvas.append(r.pop("curva"))
                resultados.append(r)
                print(f"  {nombre_fold} {f:2d}: retomado de {parte.name} (final {r['accuracy']:.4f})")
                continue
            t0 = time.time()
            m_te = mascara(f)
            modelo, hist = entrenar(args, X[~m_te], y[~m_te], X[m_te], y[m_te],
                                    n_clases, args.epochs, args.seed, early_stopping=False,
                                    sujetos_tr=sujetos[~m_te])
            curva = hist["val_accuracy"]
            curvas.append(curva)
            r = evaluar(modelo, X[m_te], y[m_te], n_clases, catalog, grupos)
            r.update({"fold": int(f), "mejorEpoca": int(np.argmax(curva)) + 1,
                      "mejorVal": round(float(max(curva)), 4),
                      "minutos": round((time.time() - t0) / 60, 1)})
            resultados.append(r)
            parte.write_text(json.dumps({**r, "curva": [float(v) for v in curva]},
                                        ensure_ascii=False), encoding="utf-8")
            imprimir_evaluacion(r, f"  {nombre_fold} {f:2d} ({r['minutos']} min, "
                                   f"mejor {r['mejorVal']:.4f} ep {r['mejorEpoca']})")
            keras.backend.clear_session()

        media = np.mean(np.array(curvas), axis=0)
        ep = int(np.argmax(media)) + 1
        finales = [r["accuracy"] for r in resultados]
        por_grupo = resumir_por_grupo(resultados, grupos)
        print(f"\nCV {etiqueta}: final {np.mean(finales):.4f} ± {np.std(finales):.4f}  "
              f"| curva media max {media.max():.4f} en epoca {ep}")
        for nombre, g in por_grupo.items():
            print(f"    {nombre:11s} acc {g['accuracy']:.4f}  top5 {g['top5']:.4f}  n={g['n']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        nombre = f"cv_{etiqueta}_s{args.seed}.json"
        (out_dir / nombre).write_text(json.dumps({
            "cv": args.cv, "etiqueta": etiqueta, "seed": args.seed,
            "arch": args.arch, "contractVersion": CONTRACT_VERSION,
            "accuracyFinalMedia": round(float(np.mean(finales)), 4),
            "accuracyFinalDesvio": round(float(np.std(finales)), 4),
            "porGrupo": por_grupo,
            "curvaMediaMax": round(float(media.max()), 4),
            "epocaSugerida": ep,
            "curvaMedia": [round(float(v), 4) for v in media],
            "folds": [{k: v for k, v in r.items() if k != "matrizConfusion"} for r in resultados],
            "matrizConfusionTotal": np.sum([r["matrizConfusion"] for r in resultados], axis=0).tolist(),
            "hiperparametros": hiperparametros(args),
            "entorno": entorno_entrenamiento(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Guardado en {out_dir / nombre}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Modelo final con todos los sujetos ----------------
    if args.final:
        modelo, hist = entrenar(args, X, y, None, None, n_clases, args.epochs,
                                args.seed, early_stopping=False, sujetos_tr=sujetos)
        metricas = {
            "modo": "final_todos_los_sujetos",
            "contractVersion": CONTRACT_VERSION,
            "sujetos": todos,
            "epocas": args.epochs,
            "trainAccuracyFinal": round(float(hist["accuracy"][-1]), 4),
            "nota": "Sin test propio: la estimacion de generalizacion sale de los CV "
                    "(loso para LSA64, reps para las clases de un solo señante) y de "
                    "evaluar_videos.py sobre grabaciones de otras personas.",
            "hiperparametros": hiperparametros(args),
            "catalogo": {"version": catalog_version, "archivo": str(args.catalog),
                         "grupos": grupos},
            "entorno": entorno_entrenamiento(),
        }
        exportar(modelo, args, out_dir, X, y, catalog, catalog_version, n_clases, metricas)
        return 0

    # ---------------- Split (train / val / test) ----------------
    val_s, test_s = set(args.val_subjects), set(args.test_subjects)
    val_r, test_r = parse_reps(args.val_reps), parse_reps(args.test_reps)
    m_te = np.isin(sujetos, list(test_s)) | mascara_reps(sujetos, reps, test_r)
    m_va = np.isin(sujetos, list(val_s)) | mascara_reps(sujetos, reps, val_r)
    if (m_te & m_va).any():
        raise SystemExit("validación y test no pueden solaparse.")
    m_tr = ~(m_te | m_va)
    faltan = sorted(set(range(n_clases)) - set(y[m_tr].tolist()))
    if faltan:
        raise SystemExit(f"Clases sin muestras de entrenamiento: {[catalog[c] for c in faltan]}")
    print(f"Split -> train={m_tr.sum()}  val={m_va.sum()} (suj {sorted(val_s)} + {args.val_reps})  "
          f"test={m_te.sum()} (suj {sorted(test_s)} + {args.test_reps})")
    modelo, hist = entrenar(args, X[m_tr], y[m_tr], X[m_va], y[m_va], n_clases,
                            args.epochs, args.seed, early_stopping=True,
                            sujetos_tr=sujetos[m_tr])
    print(f"Parametros: {modelo.count_params():,}")

    mejor_val = float(max(hist["val_accuracy"]))
    mejor_ep = int(np.argmax(hist["val_accuracy"])) + 1
    test = evaluar(modelo, X[m_te], y[m_te], n_clases, catalog, grupos)
    print("\n=== Resultado ===")
    print(f"mejor val acc: {mejor_val:.4f} (epoca {mejor_ep})")
    imprimir_evaluacion(test, "TEST")
    print(f"(baseline al azar con {n_clases} clases: {1/n_clases:.4f})")

    metricas = {
        "modo": "split",
        "contractVersion": CONTRACT_VERSION,
        "testAccuracy": test["accuracy"],
        "testTop5": test["top5"],
        "testPorGrupo": test["porGrupo"],
        "valAccuracyMejor": round(mejor_val, 4),
        "epocaMejor": mejor_ep,
        "baselineAzar": round(1 / n_clases, 4),
        "sujetosVal": sorted(val_s), "repsVal": args.val_reps,
        "sujetosTest": sorted(test_s), "repsTest": args.test_reps,
        "seed": args.seed,
        "parametros": int(modelo.count_params()),
        "hiperparametros": hiperparametros(args),
        "catalogo": {"version": catalog_version, "archivo": str(args.catalog),
                     "grupos": grupos},
        "entorno": entorno_entrenamiento(),
        "clasesBajo50": test["clasesBajo50"],
        "recallPorClase": test["recallPorClase"],
        "paresMasConfundidos": test["paresMasConfundidos"],
        "matrizConfusion": test["matrizConfusion"],
    }
    exportar(modelo, args, out_dir, X[m_te], y[m_te], catalog, catalog_version,
             n_clases, metricas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
