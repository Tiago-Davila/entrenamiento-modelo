# Eva LSA64 v3 — contrato independiente de la cámara

## Por qué existe v3

El modelo v2 daba 89% en el test de LSA64, pero en el teléfono reconocía unas
pocas señas. La causa era el contrato, no el modelo:

- LSA64 se grabó en 1920×1080 horizontal con el señante a ~2 m: el ancho de
  hombros era siempre 0,19–0,21 del ancho de la imagen.
- La app analizaba 480×640 vertical con la persona cerca: los mismos gestos
  llegaban ~3 veces más grandes y con x/y deformados ~2,4×.
- v2 solo centraba en los hombros, así que el modelo recibía valores que no
  había visto nunca.
- Con menos de 40 cuadros (el teléfono captura a 12–24 fps), v2 repetía el
  último cuadro al final y la seña quedaba comprimida al principio.

## Contrato v3 (definición completa en `eva_contract.py`)

Misma forma 40 × 168 y mismo layout de bloques que v2. Sobre los cuadros
crudos de MediaPipe:

1. Muestreo entero: `T >= 40` → `(i*(T-1))//39`; `T < 40` → `(i*T)//40`
   (cada cuadro se repite parejo).
2. A píxeles: `X = x·W`, `Y = y·H`, `Z = z·W`.
3. Centro en el punto medio de los hombros (X, Y). Z no se centra.
4. Todo se divide por la mediana del ancho de hombros.

La entrada queda en "anchos de hombro". El espejo Kotlin es
`app/.../keypoints/KeypointContract.kt`, y la paridad se verifica con
`contract_v3_fixture.json` (generado por `write_contract_fixture.py`).

## Preparar el dataset

No hace falta volver a correr MediaPipe: v2 → v3 es una conversión exacta.

```powershell
Set-Location -LiteralPath 'D:\2026-Politecnico\helpi\entrenamiento-modelo\lsa64'
..\venv\Scripts\python.exe .\convert_v2_to_v3.py --src ..\data\processed --out ..\data\processed_v3 --videos .\data\processed\raw\all
```

Para videos nuevos (propios o del teléfono) `preprocess_tasks.py` ya escribe
v3 y guarda `sizes.npy` con W×H de cada video:

```powershell
..\venv\Scripts\python.exe .\preprocess_tasks.py --videos <carpeta> --out <salida> --model .\assets\holistic_landmarker.task --workers 2
```

## Validar

```powershell
..\venv\Scripts\python.exe .\test_eva_contract.py
..\venv\Scripts\python.exe .\augment.py
```

## Entrenar

La augmentación se genera de nuevo en cada época (`AugmentedSequence`). La
capa `FeaturesEva` deriva, dentro del grafo, la forma de cada mano relativa a
su muñeca y las velocidades.

```powershell
# 1. Comparar arquitecturas con el split por sujeto (val 9, test 10)
..\venv\Scripts\python.exe .\train_keras_tflite.py --data ..\data\processed_v3 --out .\runs\split_conv_tx --arch conv_tx --epochs 150 --verbose
..\venv\Scripts\python.exe .\train_keras_tflite.py --data ..\data\processed_v3 --out .\runs\split_lstm --arch lstm --epochs 150 --verbose

# 2. Validación cruzada dejando un sujeto afuera (elige la cantidad de épocas)
..\venv\Scripts\python.exe .\train_keras_tflite.py --data ..\data\processed_v3 --out .\runs --cv loso --arch conv_tx --epochs 100 --folds 2 4 6 8 10

# 3. Modelo final con los 10 sujetos y las épocas sugeridas por el CV
..\venv\Scripts\python.exe .\train_keras_tflite.py --data ..\data\processed_v3 --out .\exports --final --arch conv_tx --epochs <N>
```

Opcional: `ab_augment.py` compara variantes de augmentación con varias
semillas (por ejemplo, activar el giro sobre el eje vertical).

### GPU

TensorFlow no usa GPU en Windows nativo, así que se entrena con Keras sobre
PyTorch (CUDA). La configuración vive en `entrenamiento-modelo/.env`:

```text
KERAS_BACKEND=torch        # GPU; "tensorflow" para entrenar en CPU
```

`lsa64/entorno.py` lo carga antes de importar Keras (una variable definida en
la terminal tiene prioridad). Instalación, una sola vez, en el mismo `venv`:

```powershell
..\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

La exportación a `.tflite` corre sola en un subproceso con backend
tensorflow: reconstruye el modelo, carga los pesos y verifica que las salidas
coincidan con las del modelo entrenado antes de convertir. Los pesos quedan
en `exports/modelo.weights.h5`.

La augmentación corre en un pool de procesos (`--workers`, 6 por defecto)
que prepara la época siguiente mientras la GPU entrena. Con la RTX 5050 una
época de `conv_tx` tarda ~5 s (en CPU, ~15 s).

## Medir con videos del teléfono

Es la métrica que importa. Grabar con el teléfono **en horizontal**, un video
por seña, nombrado como LSA64 (`SSS_PPP_RRR.mp4`: seña, persona, repetición),
y comparar el modelo viejo con el nuevo:

```powershell
..\venv\Scripts\python.exe .\evaluar_videos.py --videos <carpeta> --modelos .\exports_v2 .\exports
```

Cada modelo se evalúa con el contrato que declara su `lsa-manifest.json`.

## Artefactos para Android

```text
modelo_lsa.tflite, catalogo_senas.json, lsa-manifest.json  -> app/src/main/assets/lsa/
fixture_android.json                                        -> app/src/androidTest/assets/
```

El manifiesto declara `"contractVersion": 3`. La app rechaza un modelo cuya
versión no coincida con `KeypointContract.CONTRACT_VERSION`, así que el
modelo v2 deja de cargarse en la app v3 (y al revés).
