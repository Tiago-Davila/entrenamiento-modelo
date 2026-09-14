# Integracion en Android

Guia para integrar los modelos exportados (`exports/`) en una app Android
que traduce LSA a texto usando MediaPipe (captura de landmarks) + los
modelos encoder/decoder TFLite (o su alternativa ONNX Runtime).

---

## 1. Archivos a copiar

Desde `exports/`, copiar a `app/src/main/assets/`:

| Archivo                | Origen                          | Descripcion |
|-------------------------|----------------------------------|-------------|
| `encoder_int8.tflite`   | `exports/encoder_int8.tflite`   | Encoder Transformer + cabeza CTC (~10 MB) |
| `decoder_int8.tflite`   | `exports/decoder_int8.tflite`   | Decoder autoregresivo (~3 MB) |
| `vocab.json`            | `exports/vocab.json`            | Vocabulario (indice -> palabra) |
| `hand_landmarker.task`  | Descargar de MediaPipe (ver abajo) | Detector de landmarks de manos |

Descargar `hand_landmarker.task`:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

> Si tu build no tiene TFLite disponible (`--skip-tflite` en `04_export.py`),
> usá `encoder_int8.onnx` / `decoder_int8.onnx` con **ONNX Runtime Mobile**
> en lugar de TFLite — el pipeline de la app es igual, solo cambia el runtime.

---

## 2. Dependencias en `build.gradle`

```gradle
dependencies {
    // MediaPipe Tasks (deteccion de manos)
    implementation "com.google.mediapipe:tasks-vision:0.10.14"

    // TFLite (si usaste encoder_int8.tflite / decoder_int8.tflite)
    implementation "org.tensorflow:tensorflow-lite:2.15.0"
    implementation "org.tensorflow:tensorflow-lite-support:0.4.4"

    // Alternativa: ONNX Runtime Mobile (si exportaste solo ONNX)
    // implementation "com.microsoft.onnxruntime:onnxruntime-mobile:1.17.0"
}
```

---

## 3. Pipeline de inferencia

```
Camara (30 fps)
   │
   ▼
MediaPipe HandLandmarker  →  21 landmarks x mano x frame (x,y,z)
   │
   ▼
Buffer circular de frames (acumular hasta MAX_FRAMES = 75, ~2.5s @ 30fps)
   │
   ▼
Preprocesar (igual que dataset.py: normalize_keypoints + pad_or_crop)
   │
   ▼
Encoder TFLite  →  embeddings [1, 75, 256]
   │
   ▼
Decoder TFLite (loop autoregresivo, token a token)
   │
   ▼
Vocabulario (vocab.json)  →  texto en español
```

### 3.1 Captura y buffer de keypoints

- Usar `HandLandmarker` de MediaPipe Tasks en modo `LIVE_STREAM`.
- Por cada frame, extraer los 21 landmarks de mano izquierda y 21 de mano
  derecha (si una mano no se detecta, rellenar con ceros para esa mano).
- Concatenar en un array `[42, 3]` por frame (`x, y, z`), igual que el
  formato usado en `extract_hand_keypoints_from_pose` (ver
  [01_prepare_data.py](01_prepare_data.py)).
- Acumular frames en un buffer circular hasta juntar una ventana de
  `MAX_FRAMES = 75` frames (`config.py`).

### 3.2 Preprocesamiento (debe reproducir `dataset.py` exactamente)

Antes de pasarle la ventana al encoder, replicar en Kotlin/Java la misma
normalización que hace `normalize_keypoints()`:

1. Por cada frame, centrar `x,y` restando el centroide de los 42 puntos
   de ese frame.
2. Calcular la escala global = máximo valor absoluto de `x,y` en toda
   la ventana, y dividir todos los `x,y` por ese valor.
3. Si la ventana tiene menos de 75 frames, rellenar con ceros al final
   (`pad_or_crop`); si tiene más, hacer sub-muestreo uniforme con
   `np.linspace` (elegir 75 índices equiespaciados).

El tensor final de entrada al encoder es `[1, 75, 126]` float32
(`126 = 42 landmarks * 3 coords`, `INPUT_DIM` en `config.py`).

### 3.3 Encoder

```kotlin
val interpreter = Interpreter(loadModelFile("encoder_int8.tflite"))
// input:  keypoints [1, 75, 126] float32
// output: embeddings [1, 75, 256] float32  (usar este)
//         ctc_logits (no es necesario para inferencia con decoder)
interpreter.run(inputKeypoints, outputEmbeddings)
```

### 3.4 Decoder (loop autoregresivo)

El decoder se ejecuta token por token (no hay beam search embebido en
el modelo exportado). Reproducir en Android el mismo `greedy_decode`
que usa `model.py` en Python:

```kotlin
val BOS_ID = 1   // config.py
val EOS_ID = 2
val MAX_DEC_LEN = 64

val generated = mutableListOf(BOS_ID)
val decoderInterpreter = Interpreter(loadModelFile("decoder_int8.tflite"))

for (step in 0 until MAX_DEC_LEN) {
    // tgt_tokens: [1, generated.size] int64
    // memory:     [1, 75, 256] float32 (embeddings del encoder, fijo en todo el loop)
    // logits:     [1, generated.size, vocab_size]
    decoderInterpreter.run(
        mapOf("tgt_tokens" to generated.toLongArray(), "memory" to encoderEmbeddings),
        outputLogits
    )
    val nextId = argmax(outputLogits[0].last())  // ultimo timestep
    generated.add(nextId)
    if (nextId == EOS_ID) break
}
```

> Nota de rendimiento: reejecutar el decoder completo en cada paso
> (sin cache de atención) es más simple de portar pero más lento.
> Para 64 pasos máximo con `d_model=256` y 3 capas, es viable en CPU
> mobile; si hace falta más velocidad, se puede exportar el decoder
> con KV-cache más adelante.

### 3.5 Vocabulario y post-proceso

`vocab.json` es una lista de tokens donde el índice = el ID usado por
el modelo (mismo formato que `Vocabulary` en [dataset.py](dataset.py)):

```json
["<PAD>", "<BOS>", "<EOS>", "<UNK>", "hola", "como", ...]
```

Para convertir la secuencia de IDs generada a texto:
1. Cargar `vocab.json` como `List<String>`.
2. Recorrer los IDs generados, **omitiendo** `PAD_ID=0`, `BOS_ID=1`,
   `EOS_ID=2`, `UNK_ID=3` (tokens especiales, igual que `Vocabulary.decode`).
3. Unir las palabras restantes con espacios.

---

## 4. Checklist de integración

- [ ] Copiar `encoder_int8.tflite`, `decoder_int8.tflite`, `vocab.json` a `assets/`.
- [ ] Descargar `hand_landmarker.task` y copiarlo también a `assets/`.
- [ ] Configurar `HandLandmarker` en modo `LIVE_STREAM` con la cámara.
- [ ] Implementar el buffer de 75 frames + normalización (igual a `dataset.py`).
- [ ] Cargar encoder con `Interpreter`, correr sobre la ventana de keypoints.
- [ ] Implementar el loop autoregresivo del decoder (greedy, `MAX_DEC_LEN=64`).
- [ ] Mapear IDs a texto con `vocab.json`, filtrando tokens especiales.
- [ ] Probar con clips conocidos del test set y comparar contra
      [03_evaluate.py](03_evaluate.py) para validar que la salida coincide.

---

## 5. Verificación de consistencia (recomendado)

Antes de integrar en la app, es útil correr `03_evaluate.py --examples N`
y guardar unos pares (keypoints de entrada -> texto esperado). Reproducir
esos mismos ejemplos en Android permite confirmar que la normalización y
el loop de decodificación están bien portados antes de probar con la
cámara en vivo.
