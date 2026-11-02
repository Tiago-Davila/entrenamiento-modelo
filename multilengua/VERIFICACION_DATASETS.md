# Verificación de datasets multilengua: SWL-LSE y Google ISLR

Issue: Tiago-Davila/entrenamiento-modelo#3 · rama `003-multilengua-senas`

Pregunta: ¿se pueden llevar SWL-LSE y Google ISLR al contrato de keypoints v3
(`lsa-94/eva_contract.py`) sin deformar las señas? Para eso hay que conocer la
relación de aspecto H/W de cada secuencia, si el cuadro está espejado y si las
manos están etiquetadas como en la app (CameraX sin espejar, MediaPipe
HolisticLandmarker).

## Resumen

| Dataset | Parte | Decisión |
|---|---|---|
| SWL-LSE, 300 secuencias de referencia (`rec…`) | Videos de 480×480 | **Convertible** con H/W = 1. Los pkl están espejados: la conversión deshace el espejado (x → 1−x, intercambio de manos y de pares de pose). Ver sección 2. |
| SWL-LSE, 7 700 secuencias de señantes (IDs numéricos) | Resolución de origen desconocida | **Bloqueada.** W×H no se recupera por secuencia con error aceptable. |
| Google ISLR (`asl-signs`), 40 parquet de 21 señantes | 40 secuencias, 706 cuadros | **Convertible** con W×H = 1944×2592 (H/W = 4/3), la resolución publicada de PopSign; concuerda con las estimaciones de los datos. **Espejado sin confirmar**: la prueba x11 > x12 no lo detecta. Queda por confirmar con la mano dominante en el workspace `asl-250` (#5). |

Lo que sale de los datos: las pruebas de espejado de la primera versión (x11 > x12 en
el 100% de los cuadros, mano izquierda junto a la muñeca 15 en ~98%) **no detectan
un espejado**: MediaPipe asigna izquierda y derecha por la anatomía, y la
corrección de espejado (sección 2) deja esas dos pruebas igual. Por lo tanto no
sirven para afirmar que un cuadro no está espejado. El espejado de SWL-LSE se
sabe por el código de los autores, y la conversión lo deshace (sección 2).

## Método

### 1. Estimador de relación de aspecto (issue, punto 1)

Para cada secuencia se busca r = H/W en la grilla [0.3, 3.5] (paso 0.001) que
minimiza el coeficiente de variación (CV) de las longitudes de los huesos
rígidos de la mano (21 aristas de MediaPipe más la palma 0-17). Para cada r, la
longitud es `sqrt(dx² + (r·dy)²)` en unidades de W, con x, y normalizados.
Se usan ambas manos y se promedia el CV.

El estimador se valida de tres maneras, y ninguna de las tres alcanza ±5% por
secuencia con datos reales (ver resultados):

- Sintético sin ruido: error de +2.1% (desvío 1.9) con rotación de 0.6 rad
  (`simular-ruido`). Con rotación de 0.3 rad, el test unitario exige ±5% y pasa.
- Sintético con el ruido medido en SWL-LSE (ver abajo): error de hasta −20% a +17%.
- Videos de referencia de resolución conocida (480×480, r real = 1), procesados
  con el mismo task de MediaPipe que usa la app: la mediana es 0.959.

**Control por tronco**: el mismo estimador sobre los huesos de pose (hombros,
brazos y caderas) da otra estimación. En SWL-LSE la mediana por tronco (0.815)
no coincide con la de manos (0.978), lo que confirma que una secuencia sola no
alcanza.

**Variante 3D**: con z en unidades de W (MediaPipe documenta la misma escala que
x), la longitud 3D es invariante a la rotación. Con ruido real también se
equivoca (en la simulación, con ruido de 0.005 el error llega a −25% y con 0.01
a −37%), así que no cambia la conclusión. Se implementó como `con_z=True` y se
usa en la simulación.

### 2. Espejado (punto 2)

En un cuadro no espejado de frente, el hombro izquierdo del sujeto (pose 11)
tiene x mayor que el derecho (pose 12). Se cuenta la proporción de cuadros con
x11 > x12.

**Esta prueba no detecta espejado.** MediaPipe asigna izquierda y derecha por la
anatomía del sujeto. La transformación de corrección (x → 1−x, intercambio de los
bloques de manos y de los pares de pose 11↔12, 13↔14, …, 23↔24) conserva
x11 > x12 y la cercanía de la mano izquierda a la muñeca 15: las dos pruebas dan
el mismo resultado con o sin espejado.

**SWL-LSE sí está espejado.** El código de los autores
(`mvazquezgts/SWL-LSE`, `mediapipe_keypoints/src/api/genKeypointsHolisticLegacy.py`,
línea 30) hace `cv2.flip(image, 1)` antes de MediaPipe, así que los `.pkl` salen de
cuadros espejados. La app (CameraX) y LSA64 entregan cuadros sin espejar. La
conversión (`lse-300/convertir_swl.py`) deshace el espejado. Corroboración con la
mano dominante (muestra de 50): la mano que MediaPipe llama "izquierda" aparece más
(falta en 39% de los cuadros) que la "derecha" (falta en 53,7%). Tras la
corrección la derecha pasa a dominar, como corresponde a señantes diestros; el
número completo está en `lse-300/LSE300_DATOS.md`.

**ISLR**: el espejado no se pudo confirmar con estas pruebas. Queda pendiente de
verificar con la mano dominante en el workspace `asl-250` (#5).

### 3. Lateralidad (punto 3)

Para cada mano se mide la distancia entre su muñeca (landmark 0) y las muñecas
de Pose (15 izquierda, 16 derecha), en coordenadas (x, y·r). Coincide si la mano
izquierda queda más cerca de 15 y la derecha más cerca de 16. Además se compara
la etiqueta de handedness de la clave `hands` de SWL-LSE. Para "mano dominante"
se mide el desplazamiento medio por cuadro de los 21 puntos.

### 4. Faltantes (punto 4)

En SWL-LSE, una mano no detectada es `None` y la pose es un protobuf
serializado. Se decodifica el protobuf a mano (proto3: un campo en cero no se
escribe, así que vale 0), y la mano ausente queda como NaN, que luego se
convierte a ceros según el contrato.

### 5. Mapeo al vector de 168 (punto 5)

Basado en `holistic_legacy` de SWL-LSE (equivale a `HolisticLandmarker` de la app).
Por cuadro, en coordenadas normalizadas de MediaPipe:

| Rango del vector | Contenido | Origen en el .pkl |
|---|---|---|
| `[0, 63)` | mano izquierda, 21×(x,y,z) | `holistic_legacy["left_hand_landmarks"]` (serializado) |
| `[63, 126)` | mano derecha, 21×(x,y,z) | `holistic_legacy["right_hand_landmarks"]` (serializado) |
| `[126, 168)` | pose 11..24, 14×(x,y,z) | `holistic_legacy["pose_landmarks"]` (serializado), índices 11..24 |

Se descartan pose 0..10 y 25..32. Ausente = (0, 0, 0). Antes de aplicar el
contrato, x·W, y·H, z·W con W×H de la sección de decisiones.

### 6. Lectura segura de los .pkl

Los `.pkl` se leen con `_DesempaquetadorSeguro`, un `pickle.Unpickler` con
`find_class` restringido. Antes de fijar la lista se corrió `auditar`, que sólo
registra las clases pedidas (sin ejecutar nada). Resultado sobre las 50
muestras:

| Clase pedida | Archivos |
|---|---|
| `mediapipe.tasks.python.vision.pose_landmarker.PoseLandmarkerResult` | 50 |
| `mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult` | 50 |
| `mediapipe.tasks.python.components.containers.landmark.NormalizedLandmark` | 50 |
| `mediapipe.tasks.python.components.containers.landmark.Landmark` | 50 |
| `mediapipe.tasks.python.components.containers.category.Category` | 50 |
| `mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList` | 50 |
| `mediapipe.framework.formats.landmark_pb2.LandmarkList` | 50 |

No aparece ningún objeto de numpy. Las 7 clases de mediapipe quedan en la lista
blanca, y se reemplazan por un stub inerte que guarda el estado sin importar
mediapipe. La lista blanca de numpy y `OrderedDict` de la especificación también
se incluye, aunque no se usa. Cualquier otra clase levanta `UnpicklingError`
(probado en los tests con un pickle que pide `os.system`).

## Descargas

Todo queda en `data/multilengua/` (ignorado por `.gitignore`, regla `/data/`;
confirmado con `git check-ignore`). Total: **54.1 MB**.

| Archivo | Tamaño | Cómo |
|---|---|---|
| `swl_lse/record.json` | 10.7 KB | API de Zenodo |
| `swl_lse/ANNOTATIONS.zip` | 62 KB | URL directa del record |
| `swl_lse/videos_ref_annotations.csv` | 9.8 KB | URL directa del record |
| `swl_lse/pkl/` (50 `.pkl`) | 37.5 MB | `MEDIAPIPE.zip` leído con HTTP Range (`remotezip`); el zip (3.5 GB) no se bajó entero |
| `swl_lse/pkl_ref/` (20 `.pkl`) | 14.5 MB | Mismo método, nombres `rec…` de la muestra de videos |
| `swl_lse/videos_ref_muestra/VIDEOS_REF/` (20 `.mp4`) | 1.4 MB | `VIDEOS_REF.zip` con HTTP Range (20 de 300, semilla 5) |

Muestra de SWL-LSE: 50 IDs de 50 clases distintas (semilla 13, un ID al azar
por clase). De esos 50, **4 tienen nombre `rec…`** (video de referencia) y 46 son
numéricos (señantes). Todos los 300 `.pkl` `rec…` coinciden uno a uno con los
300 videos de referencia (`videos_ref_annotations.csv`).

**Señantes**: los metadatos no traen identificador de señante, así que no se
puede contar cuántos señantes hay en la muestra. Los IDs numéricos parecen marcas
de tiempo en microsegundos (interpretación probable, no documentada): las 46
muestras numéricas caen en 31 días distintos entre 2021-09-16 y 2024-03-15.

**ISLR**: `kaggle` 2.2.5 (instalado en el venv) lee el token de
`C:/Users/Tiago/.kaggle/access_token` (no `kaggle.json`). El token no se leyó ni se
imprimió. Se bajaron `train.csv` (6.4 MB) y `sign_to_prediction_index_map.json`
(3 KB). Se eligieron 50 parquet con semilla 21: 2 o 3 por señante, cubriendo los
21 señantes (`islr/muestra_50.json`). Se descargaron **40 de 50** (los 21 señantes
con 1–3 secuencias cada uno, 24 MB); los otros 10 se saltaron por decisión del
coordinador. Los parquet quedan bajo `islr/parquet/train_landmark_files/<participante>/`.
Un bucle en serie fallaba con 404 porque `lista.txt` tenía CRLF y el `\r` iba en el
nombre del archivo; corregido, los archivos se descargan bien uno por uno.

## Resultados SWL-LSE

### Muestra de 50 secuencias (2183 cuadros, T entre 20 y 60, mediana 46)

**Relación de aspecto (estimador de huesos)**

| Fuente | n | Mediana | p25 – p75 | Mín – Máx |
|---|---|---|---|---|
| Manos (r usada) | 48 | 0.978 | 0.871 – 1.078 | 0.569 – 1.601 |
| Tronco (control) | 50 | 0.815 | 0.602 – 1.017 | 0.300 – 2.744 |

CV medio mínimo de las manos por relación candidata (mediana sobre secuencias):
1:1 = 0.202, 4:3 = 0.207, 3:4 = 0.216, 16:9 = 0.231, 9:16 = 0.247. La diferencia
entre candidatas es chica y no decide.

**Espejado y lateralidad**

| Verificación | Resultado |
|---|---|
| x11 > x12 | 2183 / 2183 (100%) |
| Izquierda cerca de muñeca 15 | 1309 / 1332 (98.3%) |
| Derecha cerca de muñeca 16 | 993 / 1010 (98.3%) |
| Etiqueta `Left` de `hands` cerca de 15 | 1278 / 1337 (95.6%) |
| Etiqueta `Right` de `hands` cerca de 16 | 918 / 966 (95.0%) |
| Secuencias donde la izquierda se mueve más / la derecha se mueve más | 16 / 17 |

**Faltantes**

| Caso | Cuadros | Proporción |
|---|---|---|
| Sin mano izquierda | 851 / 2183 | 39.0% |
| Sin mano derecha | 1173 / 2183 | 53.7% |
| Sin ninguna mano | 619 / 2183 | 28.4% |
| Sin pose | 0 | 0% |
| Sin hombros | 0 (ninguna secuencia sin hombros en todos sus cuadros) | 0% |

Las dos fuentes de pose del mismo `.pkl` (`holistic_legacy` y `pose`) difieren en
0.027 en promedio (máximo por secuencia 0.093). No son el mismo modelo.

### Validación del estimador con videos de resolución conocida (MediaPipe real)

Se corrió `HolisticLandmarker` (el task de `lsa-94/assets/holistic_landmarker.task`)
sobre los 20 videos de referencia, todos de **480×480** (r real = 1.0):

| Estimador | n | Mediana | p25 – p75 | Mín – Máx |
|---|---|---|---|---|
| Manos | 20 | 0.959 | 0.818 – 1.095 | 0.483 – 1.750 |
| Tronco | 20 | 0.814 | 0.738 – 0.959 | 0.525 – 1.272 |

Con r real = 1 el estimador de manos tiene mediana cercana a 1, pero su
dispersión por secuencia (±0.14 de IQR) es grande. Con MediaPipe real la
dispersión de la muestra SWL-LSE (IQR 0.87–1.08) es comparable, así que los
números de la muestra no permiten distinguir una relación de otra por secuencia.

**Espejado y lateralidad en estos videos** (control de convención de MediaPipe):
x11 > x12 en 878 / 878 cuadros; izquierda cerca de 15 en 350 / 360 (97.2%),
derecha cerca de 16 en 593 / 605 (98.0%).

### Simulación de ruido (con el ruido real y el tamaño de mano real)

Ruido medido en las landmarks reales: MAD de las segundas diferencias (script
exploratorio, no incluido en el repo), ≈ 0.005 en x/y y ≈ 0.004 en z (unidades de W).
El tamaño del molde se calibra a la longitud media de los huesos reales de
SWL-LSE: **0.0433** (mediana 0.034; 49 182 huesos). Error de r en %, media (desvío):

| r real | 2D, sin ruido | 2D, ruido | 3D, ruido |
|---|---|---|---|
| 1.0 | +2.1 (1.9) | +3.4 (4.4) | −1.3 (1.8) |
| 0.75 | +2.1 (1.9) | +11.4 (5.2) | +6.4 (1.9) |
| 0.5625 (16:9) | +2.1 (1.9) | +17.2 (5.9) | +11.5 (2.0) |
| 1.7778 (9:16) | +2.1 (1.9) | −19.6 (2.8) | −24.5 (1.8) |
| 1.3333 (3:4) | +2.1 (1.9) | −7.2 (3.5) | −11.7 (1.8) |

Con este ruido el estimador es sesgado hacia 1: no alcanza ±5% para ninguna r
distinta de 1. Por eso una mediana cercana a 1 (como la de SWL-LSE) favorece 1:1
frente a 16:9 o 9:16, pero no permite confirmarlo por secuencia.

### Comparación directa con los pkl de referencia (no depende de alinear el tiempo)

Para los 20 `.pkl` `rec…` con su video de 480×480, se comparó la pose del pkl con
la de MediaPipe sobre el mismo mp4. Se midió el cociente
`(σy_pkl/σy_mp4) / (σx_pkl/σx_mp4)`. Si el pkl estuviera en un cuadro H/W distinto
de 1, este cociente sería ≈ 0.56 o ≈ 1.78.

| n | Mediana | p25 – p75 | Mín – Máx | Cuadros iguales |
|---|---|---|---|---|
| 20 | **1.047** | 1.018 – 1.067 | 0.981 – 1.096 | 8 de 20 |

Resultado: **H/W = 1** dentro de ~5% (el 5% de desvío sistemático viene probablemente
de que el modelo de pose del .pkl (Heavy) no es el mismo que el del task de la app).

**Espejado y lateralidad en los 20 pkl de referencia**: x11 > x12 en 838 / 838;
izquierda cerca de 15 en 593 / 593 (100%); derecha cerca de 16 en 334 / 347 (96.3%).

## Decisiones por dataset

### SWL-LSE, parte de referencia (300 secuencias `rec…`): convertible

- **W×H = 480×480 (H/W = 1).** Verificado en 20 de 300 videos (los 20 medidos
  son 480×480) y en 20 pkl contra MediaPipe (cociente 1.047). Las otras 280 se
  asumen 480×480 por correspondencia de nombres, no están medidas.
- **Espejado a deshacer**: `cv2.flip` del extractor de los autores (sección 2). La
  conversión aplica x → 1−x, intercambia los bloques de manos y los pares de pose.
  La prueba de la sección 2 no lo detecta, así que no se usa como control.
- Faltantes: ceros según el contrato.
- Como el contrato divide por el ancho de hombros, W sólo afecta a través de la
  relación de aspecto. Con H/W = 1 cualquier W da el mismo resultado.

**Implicación para la app**: el extractor de la app debe recibir cuadros sin
espejar, incluso con la cámara frontal, como LSA64. Si CameraX entregara el cuadro
espejado, la lateralidad se invertiría y la traducción saldría mal sin error. La
conversión de SWL-LSE lleva los datos a esa misma convención. Verificarlo en la app
con un cuadro de prueba queda pendiente (etapa 2 de `CLAUDE.md`).

### SWL-LSE, parte de señantes (7 700 secuencias numéricas): bloqueada

- El estimador no recupera W×H por secuencia con error aceptable. Con ruido real
  el error es de −20% a +17% según la relación, y con los datos reales la
  dispersión por secuencia es de ±0.1 en r (IQR 0.87–1.08 en la muestra).
- La mediana de la muestra numérica (0.98, n = 46) favorece 1:1 frente a 16:9, 9:16,
  4:3 y 3:4. Según la simulación con ruido real, esas relaciones darían medianas de
  ~0.66, ~1.43, ~0.84 y ~1.24. No alcanza para confirmarlo por secuencia, porque el
  estimador tiene sesgo y dispersión por secuencia.
- **Propuesta para destrabar**: confirmar la resolución de origen con los autores
  de SWL-LSE (o con la publicación del dataset, Technologies 2024, DOI
  10.3390/technologies12100205). Si es 1:1, convertir con H/W = 1 y con la misma
  corrección de espejado que la parte de referencia. Si no es 1:1, hace
  falta la resolución por secuencia (los videos originales no están en el record).

### Google ISLR (40 parquet, 21 señantes): convertible con W×H = 1944×2592

- **W×H = 1944×2592 (vertical 3:4, H/W = 4/3).** Es la resolución publicada de
  PopSign ASL v1.0, la fuente de los videos de la competencia: grabación con la
  cámara frontal de un Pixel 4A a 1944×2592 y 30 fps (Starner et al., *PopSign
  ASL v1.0*, NeurIPS 2023 Datasets and Benchmarks; PopSign v2.0 informa la misma
  resolución). El contrato v3 es invariante a la escala, así que solo importa la
  relación de aspecto; si MediaPipe corrió sobre un cuadro reducido, la relación
  se conserva.
- **Concuerda con los datos**: huesos de la mano, mediana 1.35; calibración
  ancho/torso, 1.41; el CV mínimo cae en 3:4. Ninguna de estas estimaciones
  alcanzaba por sí sola, pero las tres son compatibles con 4/3 y descartan 16:9.
- **Supuesto**: los landmarks de Kaggle salen de esas mismas grabaciones (el
  suplemento de PopSign v1.0 dice que se usaron para la competencia). Control
  pendiente para el workspace `asl-250/` (#5): después de convertir con H/W = 4/3,
  la proporción ancho de hombros / torso de ISLR debe quedar cerca de la de
  SWL-LSE de referencia (0.665).

Evidencia previa, solo con los datos (sin la resolución publicada):

- El parquet no trae resolución. Tres estimaciones independientes no coinciden
  entre sí (ver "Resultados Google ISLR"):
  - huesos de la mano, mediana 1.35 (n = 23 secuencias con datos suficientes);
  - huesos de tronco por secuencia, mediana 1.05 (n = 29);
  - proporción ancho/torso calibrada contra SWL-LSE de referencia (H/W = 1), r = 1.41.
  - El CV mínimo favorece 3:4 (0.150) y 9:16 (0.154) frente a 1:1 (0.158), y 16:9
    queda último (0.194). La diferencia es chica.
  - Con la misma muestra, las estimaciones de SWL-LSE tienen error de ±20% por
    secuencia, así que no es posible cerrar la relación con estos números.
  - **Lectura con baja confianza**: hay indicios de formato vertical (r entre ~1 y
    ~1.8), y 16:9 se descarta como el más improbable. No se puede afirmar 9:16 ni 3:4.
- **Espejado sin confirmar**: x11 > x12 en 706 / 706 cuadros y la muñeca cercana
  (197 / 197 y 251 / 251) no lo detectan (sección 2). Se convierte sin espejar y
  queda pendiente la verificación con la mano dominante en `asl-250` (#5).
- **Faltantes**: ISLR marca las manos ausentes como filas con `NaN` (no como filas
  faltantes). Se convierten a cero según el contrato.
- **Conversión**: H/W = 4/3, sin espejar ni intercambiar manos (supuesto, ver arriba), `NaN` → 0.

## Resultados Google ISLR

### Esquema del parquet (confirmado con los datos)

Columnas: `frame` (int16), `row_id`, `type` (`face`, `left_hand`, `pose`,
`right_hand`), `landmark_index` (int16), `x`, `y`, `z` (float64). Máximos observados:
pose 32, manos 20, cara 467. Un landmark no detectado aparece como fila con `NaN` en
x, y y z (en el ejemplo, las 336 filas de `right_hand` y 105 de `left_hand`). No
faltan filas.

### Mapeo al vector de 168 (punto 4)

| Rango del vector | Contenido | Filas del parquet |
|---|---|---|
| `[0, 63)` | mano izquierda, 21×(x,y,z) | `type == "left_hand"`, `landmark_index` 0..20 |
| `[63, 126)` | mano derecha, 21×(x,y,z) | `type == "right_hand"`, `landmark_index` 0..20 |
| `[126, 168)` | pose 11..24, 14×(x,y,z) | `type == "pose"`, `landmark_index` 11..24 |

Se descartan `face` (todo) y pose 0..10 y 25..32. `NaN` pasa a (0, 0, 0). La
función `leer_islr` hace este mapeo y devuelve pose (33), mano izquierda (21) y
mano derecha (21) por cuadro.

### Muestra y forma de las secuencias

40 parquet de 21 señantes, 706 cuadros en total. Largo por secuencia: mínimo 4,
mediana 13, máximo 64 cuadros. La cadera (pose 23) y el hombro 11 están presentes en
el 100% de los cuadros.

### Resultados (agregados)

| Verificación | Resultado |
|---|---|
| x11 > x12 | 706 / 706 (100%) |
| Izquierda junto a muñeca 15 | 197 / 197 (100%) |
| Derecha junto a muñeca 16 | 251 / 251 (100%) |
| Sin mano izquierda | 502 / 706 (71.1%) |
| Sin mano derecha | 432 / 706 (61.2%) |
| Sin ninguna mano | 228 / 706 (32.3%) |
| Sin pose ni hombros | 0 |
| Secuencias con ambas manos medibles | 0 |

**Mano dominante**: ninguna secuencia tiene ambas manos medibles (dos cuadros
consecutivos), así que no se puede comparar dentro de una secuencia. Medias por mano
sobre las que sí tienen datos: izquierda 0.0325 (n = 12), derecha 0.0289 (n = 15). No
alcanza para decidir.

**Relación de aspecto** (todas son estimaciones; ninguna cierra con ±5%):

| Estimador | n | Mediana | p25 – p75 | Mín – Máx |
|---|---|---|---|---|
| Huesos de la mano (r) | 23 | 1.348 | 1.122 – 1.697 | 0.844 – 2.428 |
| Huesos de tronco (r) | 29 | 1.049 | 0.762 – 1.711 | 0.551 – 3.500 |
| r usada (manos, o tronco si no hay manos) | 29 | 1.405 | 1.110 – 1.731 | 0.783 – 3.051 |

CV mediano de los huesos de la mano por candidata: 3:4 = 0.150, 9:16 = 0.154,
1:1 = 0.158, 4:3 = 0.166, 16:9 = 0.194.

**Calibración poblacional ancho/torso** (independiente de la estimación por
huesos): en SWL-LSE de referencia (H/W = 1) la mediana de ancho de hombros / longitud
del torso es 0.665. Con ISLR, la misma mediana se alcanza con **r = 1.41**. Por
señante, entre 1.25 (señante 37055) y 1.66 (señante 32319), con mediana cerca de
1.41. Supone que las proporciones corporales son parecidas entre los dos datasets, y
por eso es evidencia de población, no por secuencia.

**Por señante**: con 1 a 3 secuencias por señante y pocos cuadros con manos, las
estimaciones por señante no son confiables. El detalle está en
`islr/resumen_islr.json` (`resumen_por_senante`).

### Cinco ejemplos revisados a mano (ISLR)

Cuadro central de cinco secuencias repartidas (`islr/revision/*.png`), revisadas una
por una:

| ID | Seña | Señante | Cuadros | Espejado | Lateralidad | Lectura visual |
|---|---|---|---|---|---|---|
| 1016860110 | carrot | 16069 | 16 | sin espejar (16/16) | izq ok (11/11) | Mano izquierda junto a 15. Manos bien formadas. |
| 127453453 | shower | 26734 | 29 | sin espejar (29/29) | der ok (29/29) | Mano derecha junto a 16. Mano izquierda ausente. |
| 1169187867 | boat | 32319 | 18 | sin espejar (18/18) | izq ok (16/16) | Mano izquierda junto a 15, brazo derecho extendido fuera del cuadro. |
| 4216383559 | later | 37779 | 6 | sin espejar (6/6) | sin datos de r | Mano derecha junto a 16. Secuencia muy corta. |
| 430473164 | green | 62590 | 17 | sin espejar (17/17) | der ok (14/14) | Mano derecha junto a 16. r = 1.86 es el valor más alejado: no confiable. |

Como en SWL-LSE, la revisión visual confirma espejado y lateralidad, no la relación
de aspecto.

## Cinco ejemplos revisados a mano (SWL-LSE)

Se dibujó el cuadro central de cada uno con la mano en (x, y·r)
(`swl_lse/revision/*.png`), y se revisaron uno por uno:

| ID | Glosa | r usada | Espejado | Lateralidad (pose) | Lectura visual |
|---|---|---|---|---|---|
| 1631785232576395 | AUXILIAR | 1.080 | sin espejar (54/54) | ok (37/37, 40/40) | Manos anatómicas y bien ubicadas. |
| 1683740114979781 | RESPIRAR2 | 0.984 | sin espejar (60/60) | ok (44/44) | Sólo mano izquierda. La derecha no se detectó en ningún cuadro. |
| 1688644703424000 | CELULA | 1.587 | sin espejar (46/46) | ok (26/26, 29/29) | Manos ubicadas bien. r = 1.59 es el valor más alejado del grupo: no confiable por secuencia. |
| 1703627838147000 | PIS | 0.989 | sin espejar (24/24) | ok (14/14) | Sólo mano izquierda, cerca de 15. |
| recx797OWsa63Z6R6 | VACUNA2 | 0.874 | sin espejar (49/49) | ok (29/29) | Trazo de la mano irregular (posible ruido de tracking). |

Ninguna revisión visual distingue la relación de aspecto. La columna "Espejado" de
esta tabla se basó en la prueba x11 > x12, que no detecta espejado (sección 2): no
confirma nada sobre el espejado. La lateralidad de la columna "Lateralidad (pose)"
tampoco lo detecta.

## Reproducir

Desde la raíz del repo (los comandos usan `-I`, las rutas van como argumentos):

```
venv/Scripts/python.exe -I -m unittest discover -s multilengua -p "test_verificar_datasets.py"

venv/Scripts/python.exe -I multilengua/verificar_datasets.py auditar --pkl data/multilengua/swl_lse/pkl

venv/Scripts/python.exe -I multilengua/verificar_datasets.py swl \
    --pkl data/multilengua/swl_lse/pkl --metadatos data/multilengua/swl_lse \
    --salida data/multilengua/swl_lse/resumen_swl.json \
    --revision data/multilengua/swl_lse/revision

venv/Scripts/python.exe -I multilengua/verificar_datasets.py video-ref \
    --videos data/multilengua/swl_lse/videos_ref_muestra/VIDEOS_REF \
    --modelo lsa-94/assets/holistic_landmarker.task \
    --salida data/multilengua/swl_lse/validacion_videos_ref.json

venv/Scripts/python.exe -I multilengua/verificar_datasets.py comparar-ref \
    --pkl data/multilengua/swl_lse/pkl_ref \
    --videos data/multilengua/swl_lse/videos_ref_muestra/VIDEOS_REF \
    --modelo lsa-94/assets/holistic_landmarker.task \
    --salida data/multilengua/swl_lse/comparacion_refs.json

venv/Scripts/python.exe -I multilengua/verificar_datasets.py simular-ruido

venv/Scripts/python.exe -I multilengua/verificar_datasets.py islr \
    --parquet data/multilengua/islr/parquet --train-csv data/multilengua/islr/train.csv \
    --pkl-ref data/multilengua/swl_lse/pkl_ref \
    --salida data/multilengua/islr/resumen_islr.json \
    --revision data/multilengua/islr/revision
```

Los parquet de ISLR se bajan con `kaggle competitions download -c asl-signs -f <ruta>
-p data/multilengua/islr/parquet` (una ruta por vez, con pausa entre descargas).
La lista de rutas es `islr/lista.txt` (una por línea, `train_landmark_files/<participante>/<secuencia>.parquet`).

La descarga de muestras se hace con `remotezip` (`RemoteZip(url).extract(nombre, carpeta)`)
sobre `https://zenodo.org/api/records/13691887/files/MEDIAPIPE.zip/content`
y `.../VIDEOS_REF.zip/content`. La selección es la descrita en "Descargas".

Salida de `test_verificar_datasets.py`: `Ran 21 tests ... OK`. El test de
`multilengua/test_manifiesto.py` (del otro agente) no corre con `discover -s multilengua`
porque `multilengua/` no tiene `__init__.py` y el import es `multilengua.manifiesto`.
No forma parte de este issue.

## Resultado de la conversión completa (SWL-LSE)

Con el zip completo (3 499 865 422 bytes, md5 verificado) se convirtieron las 8 000
secuencias (300 referencias y 7 700 señantes), sin inválidas. Detalle en
`lse-300/LSE300_DATOS.md`.

- Mano dominante tras la corrección: la derecha falta en 34.8% de los cuadros
  (antes, la izquierda faltaba en 34.8% y la derecha en 56.0%). Por secuencia, la
  derecha tiene más cuadros con mano en 65.8% de los casos (LSA-94: 91.8%).
- Escala ancho de hombros / torso: referencia 0.668; train, val y test a +7.7%, +7.8%
  y +6.1%. Dentro de la tolerancia de 10%, pero no nula: H/W = 1 queda como supuesto.
- Sesiones (umbral 1 h): 303 sesiones, 51 cruzan particiones (fuga probable).

## Limitaciones

- La muestra es de 50 secuencias de SWL-LSE de 8000, sin identificador de señante.
- El espejado de ISLR no está verificado; la mano dominante de `asl-250` (#5) lo decide.
- El estimador de huesos no alcanza ±5% por secuencia con ruido real. Sólo sirve
  como indicio de población.
- La comparación de pkl contra MediaPipe usa el task de la app, no el modelo Heavy
  con el que se generaron los pkl. Eso explica el desvío sistemático de ~5%.
- Se verificaron 20 de 300 videos de referencia y 20 de 300 pkl de referencia.
- ISLR: 40 parquet de 21 señantes (de 50 planeados; 10 no se bajaron). Secuencias
  cortas (mediana 13 cuadros), así que las estimaciones de relación de aspecto son débiles.
  La evidencia independiente (calibración ancho/torso) depende de suponer proporciones
  corporales parecidas entre datasets.
- La estimación del ruido (MAD de segundas diferencias) viene de un script
  exploratorio no incluido en el repo; el valor usado en la simulación es 0.005/0.004.
