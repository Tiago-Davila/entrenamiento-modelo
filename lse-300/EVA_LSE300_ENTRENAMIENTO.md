# Eva LSE-300 (`eva-lse300-v1`): entrenamiento

Issue: Tiago-Davila/entrenamiento-modelo#9 · rama `003-multilengua-senas`

Modelo de señas aisladas de LSE (lengua de signos española) entrenado sobre
SWL-LSE (SignaMed, Zenodo 13691887, CC BY 4.0). Misma arquitectura, receta y
contrato de keypoints v3 (40 × 168) que `eva-lsa94-v4`. Los datos se convierten
con `convertir_swl.py` (ver `LSE300_DATOS.md`).

## Reproducir

Desde la raíz del repo, con los datos ya convertidos en
`lse-300/data/processed_v3/`:

```
venv/Scripts/python.exe lse-300/test_train_lse.py
venv/Scripts/python.exe lse-300/train_lse.py --modo eval --particion sesion --nombre sesion_e100 --epochs 100
venv/Scripts/python.exe lse-300/train_lse.py --modo eval --particion sesion --nombre sesion_e100_lsa94init --epochs 100 --init-lsa94
venv/Scripts/python.exe lse-300/train_lse.py --modo eval --particion oficial --nombre oficial_e100 --epochs 100
venv/Scripts/python.exe lse-300/train_lse.py --modo final --particion sesion --desde-eval lse-300/runs/sesion_e100.json
```

`train_lse.py` no copia la receta: importa construcción, entrenamiento,
augmentation, exportación, equivalencia y fixture de
`lsa-94/train_keras_tflite.py`. Las partes que eran fijas de LSA (nombre del
`.tflite`, manifiesto, lengua, origen, grupos del catálogo) quedaron como
parámetros con los valores de LSA por defecto; re-exportar `eva-lsa94-v4` desde
sus pesos da un `.tflite` idéntico byte a byte (sha256 `751eab8b…`).

## Decisiones

- **Clase excluida.** CLASS_ID 222 de SWL-LSE (base 0; etiqueta 223 en `y.npy`,
  base 1) figura como `#N/A` en el original: su glosa es desconocida y nunca
  puede publicarse. Se excluye (14 secuencias y 1 referencia). El modelo tiene
  **299 clases**; `metricas.json` (`clases`) guarda el `claseIdSwl` de cada una.
- **Partición principal por sesión.** El dataset no trae IDs de señante; las
  secuencias se agrupan en sesiones por marca de tiempo (`LSE300_DATOS.md`).
  Ninguna sesión aparece en más de una partición. Semilla 42, sobre las 7686
  secuencias de señantes: train 5311, val 1219, test 1156. Los 299 videos de
  referencia van solo a train (train total 5610). Las 299 clases tienen al
  menos una muestra en val y en test.
- **Partición oficial solo como comparación.** 51 de sus 303 sesiones cruzan
  particiones: puede tener fuga de señante y no se usa para decidir.
- **Receta de `eva-lsa94-v4`**: `conv_tx`, dim 192, dropout 0.3, weight decay
  0.05, label smoothing 0.1, lr 1e-3, batch 64, balance 0.5, augmentation y
  perfil fuerte de LSA-94 (detalle en `exports/metricas.json`).
- **Modelo final**: train + val + referencias de la partición por sesión (6829
  muestras), 95 épocas (la época sugerida por la media móvil de validación en la
  corrida de evaluación). El test de la partición por sesión no se usó.

## Resultados

| Corrida | Test n | Top-1 | Top-3 | Media por clase | Mejor val |
|---|---|---|---|---|---|
| Sesión, desde cero (**elegida**) | 1156 | **0.8884** | 0.9567 | 0.8956 | 0.8532 |
| Sesión, iniciada desde LSA-94 | 1156 | 0.8806 | 0.9524 | 0.8856 | 0.8450 |
| Oficial, desde cero (posible fuga) | 598 | 0.9097 | 0.9732 | 0.9097 | 0.9075 |

- Iniciar desde los pesos de LSA-94 (todas las capas menos la de salida) **no
  mejora**: se descarta.
- La partición oficial da más alto, coherente con la fuga de señante. La cifra a
  reportar es la de la partición por sesión.

### Umbral de confianza

Softmax de los logits; se elige en validación el menor umbral con al menos 90%
de exactitud en lo aceptado.

| | Umbral | Cobertura | Exactitud en lo aceptado |
|---|---|---|---|
| Validación (elección) | **0.47** | 0.9007 | 0.9007 |
| Test con ese umbral | 0.47 | 0.9265 | 0.9262 (1071 de 1156) |

En la partición oficial el mismo criterio da umbral 0.00 (validación ya supera
90%), así que no sirve. Umbral recomendado para la app: **0.47** (también en
`exports/umbral.json`).

### Señas relevantes para Helpi (test, partición por sesión)

| Categoría | n | Exactitud |
|---|---|---|
| Dolor | 53 | 0.8113 |
| Urgencia | 50 | 0.8800 |
| Síntomas | 118 | 0.9237 |
| Medicación | 49 | 0.8163 |
| **Total** | 270 | **0.8741** |

La lista `relevantesHelpi` del catálogo es una propuesta sin validar por personal
de salud ni personas sordas.

### Peores 15 clases (test, partición por sesión)

| Glosa | Exactitud | n |
|---|---|---|
| Amígdalas | 0.000 | 1 |
| Bombona mascara respirar | 0.000 | 1 |
| Curar (variante 2) | 0.200 | 5 |
| Meningitis (2D) | 0.250 | 4 |
| Azúcar | 0.333 | 6 |
| Coronavirus | 0.333 | 6 |
| Espalda | 0.333 | 6 |
| Cánula nasal dormirse | 0.333 | 3 |
| Gana | 0.333 | 3 |
| Azúcar (M ES) | 0.400 | 5 |
| Azúcar (M ES)(2M) | 0.400 | 5 |
| Respirar (variante 2) | 0.400 | 5 |
| Vena (M ON) | 0.400 | 5 |
| Cuidar | 0.500 | 10 |
| Hospital | 0.500 | 8 |

Con 1 a 10 muestras por clase en test, estas cifras son ruidosas. Varias son
variantes de la misma seña (Azúcar, Curar, Respirar), que el modelo confunde
entre sí. Matriz de confusión: `exports/matriz_confusion_eval.{json,png}`.

## Artefactos (`lse-300/exports/`, no se versionan)

| Archivo | Destino en helpi |
|---|---|
| `modelo_lse.tflite` (6694 KB) | `app/src/main/assets/lse/` |
| `catalogo_senas.json` (299 glosas) | `app/src/main/assets/lse/` |
| `manifest.json` (`language: "lse"`, CC BY 4.0) | `app/src/main/assets/lse/` |
| `fixture_android.json` (12 casos, tolerancia 1e-3) | `app/src/androidTest/assets/` |
| `metricas.json`, `umbral.json`, matriz de confusión | referencia |

Verificado: equivalencia Keras vs TFLite con diferencia máxima 1.9e-6 y la misma
predicción en 24 de 24 casos; el grafo usa solo ops estándar de TFLite (sin
Flex); el manifiesto valida con `multilengua.manifiesto.validar_manifiesto` y
sus SHA-256 coinciden con el modelo y el catálogo.

## Cambios en código compartido con LSA-94

- `lsa-94/train_keras_tflite.py`: la exportación y el arranque desde pesos
  reciben parámetros (`exportacion`, `grupos`, `pesos_iniciales`); el parser pasó
  a `construir_parser()`. Los valores por defecto son los de LSA.
- `lsa-94/augment.py`, `assert_contract`: la escala se mide solo con los cuadros
  que tienen ambos hombros, como en `normalize_pixels_v3`. Tres muestras de
  SWL-LSE tienen cuadros sin hombros y la versión anterior las rechazaba. Para
  LSA no cambia nada: sus pruebas siguen pasando.

Tests: `lse-300` (30), `multilengua` (`test_manifiesto` 11, `test_verificar_datasets`
21, usar `python -m unittest` desde la raíz), `lsa-94` (`test_eva_contract` 17,
`test_entrenamiento` 12).

## Limitaciones

- **Pocas muestras**: 132 de 300 clases tienen menos de 15 muestras de train.
- **Resolución asumida** H/W = 1 para las grabaciones de señantes; la proporción
  hombros/torso queda 6–8% por encima de la de los videos de referencia
  (`LSE300_DATOS.md`).
- **Espejado corregido** en la conversión (el extractor de los autores aplica
  `cv2.flip`). La app debe entregar cuadros sin espejar.
- **Sin IDs de señante**: la partición por sesión aproxima la independencia del
  señante, no la garantiza. No se puede separar la exactitud entre personas
  sordas, intérpretes y estudiantes.
- **Vocabulario** del ámbito de la salud de España; regionalismos sin verificar.
- **Vocabulario cerrado**: una seña fuera de las 299 debe resolverse como no
  reconocida mediante el umbral, nunca forzarse a la clase más cercana.
