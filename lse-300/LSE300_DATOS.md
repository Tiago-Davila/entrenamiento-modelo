# LSE-300: SWL-LSE convertido al contrato Eva v3

Workspace de la funcionalidad 003-multilengua-senas, issue Tiago-Davila/entrenamiento-modelo#6.
Fuente: SWL-LSE (SignaMed Word-Level LSE, Zenodo 13691887, CC BY 4.0), 300 señas de salud.

## Cómo reproducir

```
venv/Scripts/python.exe -I lse-300/test_convertir_swl.py
venv/Scripts/python.exe -I multilengua/test_verificar_datasets.py
venv/Scripts/python.exe -I lse-300/convertir_swl.py                 # valida escala y escribe processed_v3
venv/Scripts/python.exe -I lse-300/convertir_swl.py --solo-validar  # sólo validación de escala
```

El zip `data/multilengua/swl_lse/MEDIAPIPE.zip` (3 499 865 422 bytes, md5
`e128df76232e0c7dcd7a74935cd38335`, igual que en Zenodo) se lee sin descomprimir. El script
falla si el tamaño o el md5 no coinciden.

## Salidas (`lse-300/data/processed_v3/`)

| Archivo | Forma | Contenido |
|---|---|---|
| `X.npy` | (8000, 40, 168) float32 | entrada v3 |
| `y.npy` | (8000,) int32 | clase 1..300 (CLASS_ID + 1, igual que LSA-94) |
| `split.npy` | (8000,) int8 | 0 train, 1 val, 2 test, 3 referencia (`rec…`) |
| `subjects.npy` | (8000,) int32 | id de sesión (0..302); -1 para referencias |
| `reps.npy` | (8000,) int32 | ceros (SWL-LSE no trae número de repetición) |
| `nombres.npy` | (8000,) str | id de secuencia |
| `invalidas.json` | lista | vacía: ninguna secuencia inválida |
| `distribucion.json` | objeto | distribución del contrato y antes/después de la corrección |
| `meta.json` | objeto | contrato, fuente, H/W, corrección, sesiones, conteos |

## Decisiones

### 1. Espejado (corregido)

El código de los autores (`mvazquezgts/SWL-LSE`,
`mediapipe_keypoints/src/api/genKeypointsHolisticLegacy.py`, línea 30) aplica
`cv2.flip(image, 1)` antes de MediaPipe, así que los `.pkl` salen espejados. La
conversión deshace el espejado:

- x → 1 − x en todos los landmarks presentes (los ausentes siguen en cero);
- bloque mano izquierda ↔ mano derecha;
- pares de pose 11↔12, 13↔14, 15↔16, 17↔18, 19↔20, 21↔22, 23↔24.

Las pruebas x11 > x12 y "mano izquierda junto a la muñeca 15" no detectan el
espejado: la corrección las deja igual (ver `multilengua/VERIFICACION_DATASETS.md`,
sección 2).

**Corroboración con la mano dominante (8 000 secuencias, todos los cuadros):**

| | Sin mano izquierda | Sin mano derecha | Movimiento medio izq. | Movimiento medio der. |
|---|---|---|---|---|
| Antes de corregir | 34.8% | 56.0% | 0.03186 | 0.02956 |
| Después de corregir | 56.0% | 34.8% | 0.02956 | 0.03186 |

Tras corregir, la mano derecha es la dominante: falta en 34.8% de los
cuadros y se mueve más. Por secuencia, la derecha tiene más cuadros con mano que la
izquierda en 65.8% de los casos, la izquierda en 26.8%, y empatan 7.4%. En LSA-94 la
proporción es 91.8% derecha y 4.4% izquierda. La diferencia es esperable: LSA-94 tiene
muchas señas de una mano; SWL-LSE tiene más señas de dos manos.

### 2. Resolución: H/W = 1 (480 × 480)

Referencias (`rec…`, 480 × 480 verificados) y señantes (asumidos 480 × 480). Ver
validación de escala (sección 3). Si la resolución real de los señantes fuera
distinta, la escala ancho de hombros / torso se desviaría; la comprobación de abajo
muestra un desvío de +6% a +8%, dentro de la tolerancia pero no nulo.

### 3. Validación de escala: ancho de hombros / longitud del torso

Definición: por secuencia, mediana por cuadro de (distancia entre hombros 11–12) /
(distancia entre el punto medio de hombros y el de caderas 23–24), en píxeles con
H/W = 1. Luego mediana por partición. Tolerancia: 10% sobre la referencia.

| Partición | n | Mediana | Desvío vs referencia | ¿Pasa? |
|---|---|---|---|---|
| Referencia (`rec…`) | 300 | 0.6678 | — | — |
| Train | 6 048 | 0.7194 | +7.7% | sí |
| Val | 1 052 | 0.7203 | +7.8% | sí |
| Test | 600 | 0.7089 | +6.1% | sí |
| Señantes (total) | 7 700 | 0.7189 | +7.6% | sí |

La referencia (0.6678) coincide con la de #3 (0.665). Las particiones numéricas
quedan 6–8% por encima. Esa diferencia no se explica con H/W = 1: una relación de
≈1.07 la eliminaría en la muestra, pero no hay evidencia independiente de eso. Queda
registrada como límite de la conversión.

### 4. Partición y sesiones

**Partición**: la oficial de `ANNOTATIONS.zip` (train 6 048, val 1 052, test 600
numéricas). Las 300 `rec…` estaban en train en el archivo oficial: se apartan a
split = 3 (referencia), así no entran en train ni en test.

**Sesiones**: los IDs numéricos son marcas de tiempo en microsegundos (hipótesis no
documentada). Se ordenan y se corta sesión cuando el salto entre IDs consecutivos
supera el umbral. Histograma de saltos (log10 de segundos, 7 699 saltos): mediana
del salto 26 s, la mayoría entre 10 y 30 s dentro de una sesión; entre 10^2.5 y 10^4 s
la densidad de saltos es baja (mínimo local cerca de 10^3–10^3.5 s). El umbral
elegido es **3 600 s (1 hora)**, dentro de esa zona. Sensibilidad:

| Umbral | Sesiones | Cruzadas (train↔test o train↔val) | train↔test | train↔val | val↔test | Secuencias en sesiones cruzadas |
|---|---|---|---|---|---|---|
| 300 s (5 min) | 550 | 65 | 37 | 43 | 15 | 3592 |
| 1800 s (30 min) | 338 | 53 | 29 | 39 | 15 | 4073 |
| 3600 s (60 min) | 303 | 51 | 30 | 36 | 15 | 4180 |
| 7200 s (120 min) | 270 | 51 | 31 | 36 | 16 | 4342 |
| 14400 s (240 min) | 234 | 44 | 28 | 32 | 16 | 4706 |

Con 3 600 s: 303 sesiones, **51 sesiones cruzan particiones**
(30 train↔test, 36 train↔val, 15 val↔test), con
4180 secuencias afectadas. Una sesión no equivale a un
señante: el dataset no trae identificador de señante y una persona puede grabar en
varias sesiones. La fuga es probable, no comprobada. Evaluar sin fuga por señante
requiere una partición nueva por sesión (no se hizo acá).

### 5. Secuencias inválidas

Ninguna: las 8 000 secuencias tienen hombros en al menos un cuadro.

### 6. Catálogo

`lse-300/catalogo_lse300.json`: 300 glosas en orden CLASS_ID 0..299 → índice 1..300.
La glosa legible sale del LABEL original: separadores `^`, `.`, `-` a espacios,
variantes numéricas "2" a "(variante 2)", abreviaturas entre paréntesis en mayúscula.
El campo `etiquetasOriginales` conserva el texto tal cual.

- Clase #N/A: **CLASS_ID 222 en base 0** (índice de `catalogo_lse300.json` y de
  SWL-LSE) = **etiqueta 223 en base 1** (la de `y.npy` y la de la tabla de abajo).
  Glosa original: "No identificada (#N/A en el original)", probable celda vacía en el
  archivo de origen. Se excluye del entrenamiento (15 muestras: 14 secuencias y 1
  referencia). Nunca se publica una glosa desconocida. El modelo `eva-lse300-v1` tiene
  299 clases: `catalogo_senas.json` no la incluye.
- **Regionalismos de España: no verificados** (`regionalismosVerificados: false`).
  Pendiente de revisión.
- `relevantesHelpi`: **propuesta para revisar** (no validada por personal de salud ni
  por personas sordas): 69 señas en cuatro categorías (dolor, urgencia,
  síntomas, medicación), con detalle en `relevantesHelpiPorCategoria`.

## Muestras por clase y partición

Clase = índice 1..300. Train con menos de 15 secuencias en negrita.

Clases con menos de 15 muestras en train: **132 de 300**. Train: mínimo 6,
mediana 16, máximo 48. Val: 3–4 por clase.
Test: 2–2 por clase. Referencia: 1 por clase.

| Clase | Glosa | Train | Val | Test | Referencia |
|---|---|---|---|---|---|
| 1 | Aborto | 22 | 4 | 2 | 1 |
| 2 | Accidente | 15 | 4 | 2 | 1 |
| 3 | Accidente (A) | 9 **(<15)** | 3 | 2 | 1 |
| 4 | Accidente (variante 2) | 22 | 3 | 2 | 1 |
| 5 | Acne | 7 **(<15)** | 4 | 2 | 1 |
| 6 | Aconsejar | 42 | 4 | 2 | 1 |
| 7 | Acufeno | 21 | 4 | 2 | 1 |
| 8 | Adelgazar | 29 | 4 | 2 | 1 |
| 9 | Afectar | 48 | 3 | 2 | 1 |
| 10 | Afectar (S) | 48 | 3 | 2 | 1 |
| 11 | Agotamiento | 45 | 4 | 2 | 1 |
| 12 | Agua alcohol | 8 **(<15)** | 4 | 2 | 1 |
| 13 | Alcohol | 39 | 4 | 2 | 1 |
| 14 | Alergia | 23 | 3 | 2 | 1 |
| 15 | Alergia (variante 2) | 33 | 4 | 2 | 1 |
| 16 | Algodon | 9 **(<15)** | 4 | 2 | 1 |
| 17 | Alta | 27 | 4 | 2 | 1 |
| 18 | Alta (2M) | 9 **(<15)** | 4 | 2 | 1 |
| 19 | Alta (MP) | 30 | 4 | 2 | 1 |
| 20 | Alzheimer | 15 | 4 | 2 | 1 |
| 21 | Ambulancia | 25 | 4 | 2 | 1 |
| 22 | Ambulancia (MP) | 23 | 3 | 2 | 1 |
| 23 | Amigdalas | 8 **(<15)** | 4 | 2 | 1 |
| 24 | Amigdalas inflamar | 12 **(<15)** | 3 | 2 | 1 |
| 25 | Amigdalitis | 16 | 4 | 2 | 1 |
| 26 | Analisis de sangre | 45 | 4 | 2 | 1 |
| 27 | Analisis de sangre (2) | 19 | 4 | 2 | 1 |
| 28 | Analisis de sangre (5) | 20 | 4 | 2 | 1 |
| 29 | Analizar | 26 | 4 | 2 | 1 |
| 30 | Angina pecho | 21 | 4 | 2 | 1 |
| 31 | Ansiedad | 48 | 3 | 2 | 1 |
| 32 | Antecedentes | 8 **(<15)** | 4 | 2 | 1 |
| 33 | Apendice | 31 | 4 | 2 | 1 |
| 34 | Apetecer | 29 | 3 | 2 | 1 |
| 35 | Arcada | 13 **(<15)** | 3 | 2 | 1 |
| 36 | Ardor | 21 | 3 | 2 | 1 |
| 37 | Ardor (OP ST) | 12 **(<15)** | 3 | 2 | 1 |
| 38 | Arteria (variante 2) | 37 | 4 | 2 | 1 |
| 39 | Arteria (variante 3) | 20 | 4 | 2 | 1 |
| 40 | Arteria (variante 3) rojo menos | 24 | 4 | 2 | 1 |
| 41 | Articulacion | 10 **(<15)** | 3 | 2 | 1 |
| 42 | Articulacion (M RT) | 9 **(<15)** | 3 | 2 | 1 |
| 43 | Asma | 27 | 4 | 2 | 1 |
| 44 | Auxiliar | 31 | 3 | 2 | 1 |
| 45 | Ayudar | 9 **(<15)** | 3 | 2 | 1 |
| 46 | Azucar | 42 | 4 | 2 | 1 |
| 47 | Azucar brazo | 12 **(<15)** | 3 | 2 | 1 |
| 48 | Azucar (2M) | 40 | 4 | 2 | 1 |
| 49 | Azucar (M ES) | 37 | 4 | 2 | 1 |
| 50 | Azucar (M ES)(2M) | 10 **(<15)** | 3 | 2 | 1 |
| 51 | Bacteria | 10 **(<15)** | 3 | 2 | 1 |
| 52 | Baja | 13 **(<15)** | 3 | 2 | 1 |
| 53 | Bombona mascara | 9 **(<15)** | 4 | 2 | 1 |
| 54 | Bombona mascara respirar | 8 **(<15)** | 3 | 2 | 1 |
| 55 | Bombona respirar | 10 **(<15)** | 3 | 2 | 1 |
| 56 | Bronquitis | 7 **(<15)** | 3 | 2 | 1 |
| 57 | Bulto | 17 | 4 | 2 | 1 |
| 58 | Bulto pecho | 21 | 4 | 2 | 1 |
| 59 | Bulto (L DOR) | 11 **(<15)** | 4 | 2 | 1 |
| 60 | Bulto (L DOR) blando (MP) | 10 **(<15)** | 3 | 2 | 1 |
| 61 | Bulto (variante 2) | 12 **(<15)** | 3 | 2 | 1 |
| 62 | Caca diarrea (variante 2) | 21 | 4 | 2 | 1 |
| 63 | Caca diarrea (variante 2) (1M) | 22 | 4 | 2 | 1 |
| 64 | Caca (variante 2) | 47 | 4 | 2 | 1 |
| 65 | Caca (variante 2) (L PEC) | 48 | 3 | 2 | 1 |
| 66 | Caer | 7 **(<15)** | 3 | 2 | 1 |
| 67 | Cancer | 46 | 4 | 2 | 1 |
| 68 | Cansado (variante 2) | 48 | 3 | 2 | 1 |
| 69 | Canula nasal dormirse | 7 **(<15)** | 3 | 2 | 1 |
| 70 | Celula | 15 | 4 | 2 | 1 |
| 71 | Centro vivir | 39 | 3 | 2 | 1 |
| 72 | Cerebro (variante 2) | 15 | 3 | 2 | 1 |
| 73 | Certificado | 9 **(<15)** | 3 | 2 | 1 |
| 74 | Circulación brazo menos | 17 | 4 | 2 | 1 |
| 75 | Circulacion (S) | 9 **(<15)** | 3 | 2 | 1 |
| 76 | Cita | 30 | 3 | 2 | 1 |
| 77 | Citologia | 8 **(<15)** | 4 | 2 | 1 |
| 78 | Cojera | 12 **(<15)** | 3 | 2 | 1 |
| 79 | Colesterol | 18 | 4 | 2 | 1 |
| 80 | Colesterol (variante 2) | 11 **(<15)** | 4 | 2 | 1 |
| 81 | Colonoscopia | 11 **(<15)** | 3 | 2 | 1 |
| 82 | Comer | 48 | 3 | 2 | 1 |
| 83 | Comer poder (variante 2) | 25 | 3 | 2 | 1 |
| 84 | Comer sin | 30 | 3 | 2 | 1 |
| 85 | Consulta fuera (MP) | 9 **(<15)** | 3 | 2 | 1 |
| 86 | Contra | 29 | 4 | 2 | 1 |
| 87 | Contra defender (M CI) | 7 **(<15)** | 3 | 2 | 1 |
| 88 | Control | 29 | 4 | 2 | 1 |
| 89 | Convulsion | 7 **(<15)** | 3 | 2 | 1 |
| 90 | Corazon | 31 | 4 | 2 | 1 |
| 91 | Corazon infarto | 8 **(<15)** | 4 | 2 | 1 |
| 92 | Corazon latir | 17 | 3 | 2 | 1 |
| 93 | Coronavirus | 11 **(<15)** | 3 | 2 | 1 |
| 94 | Costumbre | 15 | 4 | 2 | 1 |
| 95 | Costumbre haber no (2M) | 27 | 4 | 2 | 1 |
| 96 | Costumbre menopausia | 6 **(<15)** | 4 | 2 | 1 |
| 97 | Costumbre sin | 18 | 3 | 2 | 1 |
| 98 | Cuidar | 44 | 3 | 2 | 1 |
| 99 | Curar | 46 | 4 | 2 | 1 |
| 100 | Curar (variante 2) | 32 | 3 | 2 | 1 |
| 101 | Debil | 46 | 4 | 2 | 1 |
| 102 | Depresion | 45 | 3 | 2 | 1 |
| 103 | Desayunar poder (variante 2) | 26 | 3 | 2 | 1 |
| 104 | Desayunar sin | 28 | 3 | 2 | 1 |
| 105 | Descubrir | 13 **(<15)** | 3 | 2 | 1 |
| 106 | Diabetes | 48 | 3 | 2 | 1 |
| 107 | Diarrea | 23 | 4 | 2 | 1 |
| 108 | Diarrea (variante 2) | 29 | 4 | 2 | 1 |
| 109 | Diarrea (variante 3) | 16 | 4 | 2 | 1 |
| 110 | Dieta | 16 | 4 | 2 | 1 |
| 111 | Dolor | 47 | 4 | 2 | 1 |
| 112 | Dolor de cabeza | 30 | 4 | 2 | 1 |
| 113 | Dormir | 46 | 4 | 2 | 1 |
| 114 | Dormir insomnio | 25 | 4 | 2 | 1 |
| 115 | Droga | 46 | 4 | 2 | 1 |
| 116 | Droga alergia contra | 9 **(<15)** | 3 | 2 | 1 |
| 117 | Droga contra | 16 | 4 | 2 | 1 |
| 118 | Droga farmacia | 13 **(<15)** | 3 | 2 | 1 |
| 119 | Droga nivel bajar | 9 **(<15)** | 3 | 2 | 1 |
| 120 | Droga relajar | 16 | 4 | 2 | 1 |
| 121 | Droga terapia | 7 **(<15)** | 4 | 2 | 1 |
| 122 | Dulce circulación brazo | 10 **(<15)** | 3 | 2 | 1 |
| 123 | Ecografia | 21 | 4 | 2 | 1 |
| 124 | Ejercicio | 7 **(<15)** | 4 | 2 | 1 |
| 125 | Ejercicio (M DE) | 9 **(<15)** | 3 | 2 | 1 |
| 126 | Embarazo | 48 | 3 | 2 | 1 |
| 127 | Embarazo defender | 8 **(<15)** | 3 | 2 | 1 |
| 128 | Embarazo evitar | 8 **(<15)** | 3 | 2 | 1 |
| 129 | Endoscopia | 14 **(<15)** | 3 | 2 | 1 |
| 130 | Enfermeria | 21 | 3 | 2 | 1 |
| 131 | Enfermo | 47 | 3 | 2 | 1 |
| 132 | Enfermo casa (M AB) | 32 | 4 | 2 | 1 |
| 133 | Enfermo es asi corazon | 9 **(<15)** | 3 | 2 | 1 |
| 134 | Enfermo operar | 7 **(<15)** | 3 | 2 | 1 |
| 135 | Enfermo principal | 16 | 3 | 2 | 1 |
| 136 | Equilibrio | 10 **(<15)** | 4 | 2 | 1 |
| 137 | Escuchar (variante 2) | 9 **(<15)** | 3 | 2 | 1 |
| 138 | Escuchar (variante 2) revisar (M CI) | 16 | 4 | 2 | 1 |
| 139 | Espalda | 18 | 3 | 2 | 1 |
| 140 | Espalda (1) | 8 **(<15)** | 3 | 2 | 1 |
| 141 | Espalda (AP) | 7 **(<15)** | 3 | 2 | 1 |
| 142 | Estomago | 17 | 3 | 2 | 1 |
| 143 | Estres | 47 | 4 | 2 | 1 |
| 144 | Estres (1M) | 25 | 4 | 2 | 1 |
| 145 | Evitar | 33 | 4 | 2 | 1 |
| 146 | Explotar | 26 | 4 | 2 | 1 |
| 147 | Farmacia | 38 | 4 | 2 | 1 |
| 148 | Fiebre (variante 2) | 34 | 4 | 2 | 1 |
| 149 | Fuerte | 8 **(<15)** | 3 | 2 | 1 |
| 150 | Fumar | 31 | 3 | 2 | 1 |
| 151 | Gana | 38 | 4 | 2 | 1 |
| 152 | Garganta | 17 | 4 | 2 | 1 |
| 153 | Garganta (variante 2) | 39 | 4 | 2 | 1 |
| 154 | Gordo | 37 | 3 | 2 | 1 |
| 155 | Grasa | 10 **(<15)** | 3 | 2 | 1 |
| 156 | Grasa (OP ST) | 10 **(<15)** | 3 | 2 | 1 |
| 157 | Grasa (variante 3) | 9 **(<15)** | 3 | 2 | 1 |
| 158 | Grasa (variante 3) brazo | 9 **(<15)** | 3 | 2 | 1 |
| 159 | Gripe | 33 | 4 | 2 | 1 |
| 160 | Gripe (XC) | 16 | 3 | 2 | 1 |
| 161 | H celula | 9 **(<15)** | 3 | 2 | 1 |
| 162 | H hígado | 19 | 4 | 2 | 1 |
| 163 | Habitacion esperar | 13 **(<15)** | 3 | 2 | 1 |
| 164 | Hambre | 41 | 4 | 2 | 1 |
| 165 | Hepatitis | 17 | 4 | 2 | 1 |
| 166 | Herida | 34 | 4 | 2 | 1 |
| 167 | Herida (2MSU) | 8 **(<15)** | 3 | 2 | 1 |
| 168 | Higado | 19 | 4 | 2 | 1 |
| 169 | Hipoacusico | 21 | 4 | 2 | 1 |
| 170 | Hormona | 6 **(<15)** | 3 | 2 | 1 |
| 171 | Hospital | 41 | 4 | 2 | 1 |
| 172 | Hospital desplazar | 20 | 4 | 2 | 1 |
| 173 | Ictus | 11 **(<15)** | 3 | 2 | 1 |
| 174 | Ictus (variante 2) | 16 | 3 | 2 | 1 |
| 175 | Infarto | 19 | 4 | 2 | 1 |
| 176 | Insomnio | 29 | 4 | 2 | 1 |
| 177 | Insulina | 12 **(<15)** | 3 | 2 | 1 |
| 178 | Intestino | 10 **(<15)** | 3 | 2 | 1 |
| 179 | Inyeccion | 21 | 4 | 2 | 1 |
| 180 | Inyeccion dormirse | 9 **(<15)** | 4 | 2 | 1 |
| 181 | Lactancia | 20 | 4 | 2 | 1 |
| 182 | Lactancia (OP AB) | 9 **(<15)** | 3 | 2 | 1 |
| 183 | Lactancia (variante 2) | 8 **(<15)** | 4 | 2 | 1 |
| 184 | Lesion | 27 | 4 | 2 | 1 |
| 185 | Mancha brazo | 8 **(<15)** | 3 | 2 | 1 |
| 186 | Mancha pecho | 6 **(<15)** | 3 | 2 | 1 |
| 187 | Mareo | 46 | 4 | 2 | 1 |
| 188 | Mareo (variante 2) | 43 | 4 | 2 | 1 |
| 189 | Mareo (variante 2) caer | 10 **(<15)** | 3 | 2 | 1 |
| 190 | Mascara | 10 **(<15)** | 3 | 2 | 1 |
| 191 | Meningitis | 20 | 4 | 2 | 1 |
| 192 | Meningitis (2D) | 25 | 4 | 2 | 1 |
| 193 | Meningitis (CT) | 14 **(<15)** | 4 | 2 | 1 |
| 194 | Meningitis (CT) etc | 10 **(<15)** | 3 | 2 | 1 |
| 195 | Menos | 22 | 3 | 2 | 1 |
| 196 | Mente apagada | 10 **(<15)** | 3 | 2 | 1 |
| 197 | Mente apagar | 11 **(<15)** | 4 | 2 | 1 |
| 198 | Mente cerebro | 32 | 4 | 2 | 1 |
| 199 | Mente demencia | 22 | 4 | 2 | 1 |
| 200 | Mente explotar | 13 **(<15)** | 4 | 2 | 1 |
| 201 | Mente infarto | 19 | 4 | 2 | 1 |
| 202 | Mente olvidar | 9 **(<15)** | 3 | 2 | 1 |
| 203 | Mes | 27 | 4 | 2 | 1 |
| 204 | Mes jubilar | 9 **(<15)** | 3 | 2 | 1 |
| 205 | Mierda | 22 | 4 | 2 | 1 |
| 206 | Moco | 10 **(<15)** | 3 | 2 | 1 |
| 207 | Moco nariz | 9 **(<15)** | 3 | 2 | 1 |
| 208 | Moco (variante 2) | 8 **(<15)** | 4 | 2 | 1 |
| 209 | Musculo | 12 **(<15)** | 3 | 2 | 1 |
| 210 | Nacer | 40 | 4 | 2 | 1 |
| 211 | Nausea | 16 | 4 | 2 | 1 |
| 212 | Nervioso | 23 | 4 | 2 | 1 |
| 213 | Nivel bajar | 8 **(<15)** | 3 | 2 | 1 |
| 214 | Nivel subir | 15 | 3 | 2 | 1 |
| 215 | Normal | 11 **(<15)** | 4 | 2 | 1 |
| 216 | Oir mitad | 40 | 4 | 2 | 1 |
| 217 | Oir ruido | 17 | 4 | 2 | 1 |
| 218 | Olvidar | 11 **(<15)** | 3 | 2 | 1 |
| 219 | Operar | 36 | 4 | 2 | 1 |
| 220 | Operar (M DE) | 29 | 4 | 2 | 1 |
| 221 | Ovario | 18 | 4 | 2 | 1 |
| 222 | Paralisis | 14 **(<15)** | 3 | 2 | 1 |
| 223 | No identificada (#N/A en el original) | 9 **(<15)** | 3 | 2 | 1 |
| 224 | Pie | 14 **(<15)** | 4 | 2 | 1 |
| 225 | Piel | 8 **(<15)** | 4 | 2 | 1 |
| 226 | Piel (M RT) | 7 **(<15)** | 4 | 2 | 1 |
| 227 | Piel cuerpo | 44 | 4 | 2 | 1 |
| 228 | Piel sano | 11 **(<15)** | 3 | 2 | 1 |
| 229 | Pierna | 11 **(<15)** | 3 | 2 | 1 |
| 230 | Pierna (variante 2) | 12 **(<15)** | 3 | 2 | 1 |
| 231 | Pis | 48 | 3 | 2 | 1 |
| 232 | Pis (variante 2) | 46 | 4 | 2 | 1 |
| 233 | Pis (variante 3) | 38 | 3 | 2 | 1 |
| 234 | Pis (variante 4) | 37 | 3 | 2 | 1 |
| 235 | Placa camilla | 9 **(<15)** | 3 | 2 | 1 |
| 236 | Probar | 46 | 4 | 2 | 1 |
| 237 | Problema | 21 | 4 | 2 | 1 |
| 238 | Pulmon | 30 | 3 | 2 | 1 |
| 239 | Pulmon enfermo | 10 **(<15)** | 4 | 2 | 1 |
| 240 | Quimioterapia | 20 | 4 | 2 | 1 |
| 241 | Quimioterapia (K) | 22 | 4 | 2 | 1 |
| 242 | Radiografia | 19 | 4 | 2 | 1 |
| 243 | Radioterapia | 12 **(<15)** | 3 | 2 | 1 |
| 244 | Radioterapia (5D) | 10 **(<15)** | 3 | 2 | 1 |
| 245 | Reducir | 20 | 4 | 2 | 1 |
| 246 | Reducir (B) | 23 | 3 | 2 | 1 |
| 247 | Relajar | 30 | 3 | 2 | 1 |
| 248 | Resonancia | 7 **(<15)** | 3 | 2 | 1 |
| 249 | Respirar | 24 | 4 | 2 | 1 |
| 250 | Respirar (variante 2) | 7 **(<15)** | 4 | 2 | 1 |
| 251 | Resultado | 15 | 4 | 2 | 1 |
| 252 | Resultado (variante 2) | 6 **(<15)** | 4 | 2 | 1 |
| 253 | Revisar | 19 | 3 | 2 | 1 |
| 254 | Revisar (M CI) | 13 **(<15)** | 3 | 2 | 1 |
| 255 | Revisar (M CI) descubrir | 13 **(<15)** | 4 | 2 | 1 |
| 256 | Revisar (S) | 20 | 3 | 2 | 1 |
| 257 | Revisar (S)(M CI) | 15 | 3 | 2 | 1 |
| 258 | Revisar descubrir | 14 **(<15)** | 4 | 2 | 1 |
| 259 | Riñon | 16 | 3 | 2 | 1 |
| 260 | Saliva escupir | 8 **(<15)** | 3 | 2 | 1 |
| 261 | Saliva sustancia | 6 **(<15)** | 3 | 2 | 1 |
| 262 | Sangre (variante 2) | 18 | 4 | 2 | 1 |
| 263 | Sangre (variante 2) coaugulo contra | 9 **(<15)** | 3 | 2 | 1 |
| 264 | Sano | 24 | 3 | 2 | 1 |
| 265 | Seccion hueso | 8 **(<15)** | 3 | 2 | 1 |
| 266 | Seccion pequeño (variante 2) | 12 **(<15)** | 3 | 2 | 1 |
| 267 | Sentir | 25 | 4 | 2 | 1 |
| 268 | Sentir bloquear | 10 **(<15)** | 3 | 2 | 1 |
| 269 | Sida | 7 **(<15)** | 3 | 2 | 1 |
| 270 | Sintoma | 12 **(<15)** | 4 | 2 | 1 |
| 271 | Sufrir | 34 | 4 | 2 | 1 |
| 272 | Tac | 10 **(<15)** | 3 | 2 | 1 |
| 273 | Taquicardia | 10 **(<15)** | 3 | 2 | 1 |
| 274 | Tension (B) | 6 **(<15)** | 3 | 2 | 1 |
| 275 | Tension (CT) | 10 **(<15)** | 3 | 2 | 1 |
| 276 | Tension (CT) alta | 23 | 4 | 2 | 1 |
| 277 | Tension (variante 2) | 13 **(<15)** | 4 | 2 | 1 |
| 278 | Terapia | 26 | 4 | 2 | 1 |
| 279 | Tiroides | 19 | 3 | 2 | 1 |
| 280 | Toser | 26 | 4 | 2 | 1 |
| 281 | Triaje | 6 **(<15)** | 3 | 2 | 1 |
| 282 | Urgente | 29 | 4 | 2 | 1 |
| 283 | Vacuna | 26 | 4 | 2 | 1 |
| 284 | Vacuna (variante 2) | 11 **(<15)** | 4 | 2 | 1 |
| 285 | Vacuna (variante 2) (M RT) | 18 | 4 | 2 | 1 |
| 286 | Vagina | 7 **(<15)** | 3 | 2 | 1 |
| 287 | Varicela | 7 **(<15)** | 3 | 2 | 1 |
| 288 | Varicela (variante 2) | 9 **(<15)** | 4 | 2 | 1 |
| 289 | Varicela (variante 3) | 6 **(<15)** | 3 | 2 | 1 |
| 290 | Vena | 12 **(<15)** | 3 | 2 | 1 |
| 291 | Vena (M ON) | 12 **(<15)** | 3 | 2 | 1 |
| 292 | Ver mal | 14 **(<15)** | 4 | 2 | 1 |
| 293 | Ver control | 26 | 3 | 2 | 1 |
| 294 | Ver probar | 46 | 4 | 2 | 1 |
| 295 | Virus | 34 | 4 | 2 | 1 |
| 296 | Virus B | 8 **(<15)** | 3 | 2 | 1 |
| 297 | Virus contra | 10 **(<15)** | 3 | 2 | 1 |
| 298 | Virus defender | 8 **(<15)** | 3 | 2 | 1 |
| 299 | Virus evitar | 7 **(<15)** | 3 | 2 | 1 |
| 300 | Vomitar | 17 | 3 | 2 | 1 |

## Distribución del contrato frente a LSA-94

Proporción de cuadros con bloque entero en cero y movimiento medio por cuadro
(distancia entre cuadros consecutivos de los puntos presentes, en unidades de ancho de
imagen; con W = H = 480 las distancias son en fracción de la imagen).

| Métrica | SWL-LSE (lse-300) | LSA-94 |
|---|---|---|
| Cuadros sin mano izquierda | 57.0% | 79.6% |
| Cuadros sin mano derecha | 35.1% | 12.2% |
| Cuadros sin pose | 0.0% | 0.0% |
| Movimiento medio mano izq. | 0.06844 | 0.04208 |
| Movimiento medio mano der. | 0.07784 | 0.04479 |
| Movimiento medio pose | 0.05511 | 0.02276 |

Rangos por eje (percentiles 1 y 99 de los puntos presentes):

| Eje | SWL-LSE p1 / p99 | LSA-94 p1 / p99 |
|---|---|---|
| x | -1.0642 / 1.0315 | -1.435 / 0.9799 |
| y | -0.7727 / 1.8644 | -0.8861 / 1.825 |
| z | -4.4631 / 0.1783 | -5.4192 / 0.1129 |

Lectura: en SWL-LSE la mano izquierda (no dominante) está presente en 43.0%
de los cuadros, frente a 20.4% en LSA-94: SWL-LSE tiene más señas de dos
manos. El movimiento de la pose también es mayor. Los rangos x e y son algo más amplios;
es coherente con la escala de hombros distinta (sección 3). No es una prueba de error.

## Límites

- Sesiones por marca de tiempo: no son señantes. Fuga probable, no medida.
- Escala de señantes 6–8% por encima de la referencia; H/W = 1 sin evidencia independiente.
- 2 a 4 secuencias por clase en val y test: métricas de esas particiones muy ruidosas.
- Glosas legibles pero no revisadas por personas sordas ni por personal de salud.
  Regionalismos sin verificar.
- Clase #N/A (CLASS_ID 222 base 0 = etiqueta 223 base 1) excluida del entrenamiento;
  la celda de origen sin revisar.
