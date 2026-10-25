# Eva LSA-94 — 94 señas (eva-lsa94-v4)

## Qué cambia respecto de Eva LSA64 v3

| | LSA64 v3 (`../lsa64`, tag `eva-lsa64-v3`) | LSA-94 v4 (esta carpeta) |
|---|---|---|
| Clases | 64 | 94: las 64 + Hoy, Ayer, Mañana + abecedario (A … C, CH, D … Z) |
| Catálogo | `catalogo_lsa64.json` v2.0 | `catalogo_eva94.json` v3.0, con grupos |
| Sujetos | 001-010 (LSA64, 5 reps) | + 012 (celular, 4 reps de las 94) |
| Entrada | contrato v3, 40 × 168 | **igual**: Android no cambia el preprocesado |
| Features en el grafo | `FeaturesEva` | `FeaturesEva94`: + forma de mano (distancias entre dedos, flexión) |
| Muestreo | natural | balanceado por clase (`--balance 0.5`) |
| Augmentación | v3 | + largo de dedos, giro de mano, deriva lenta; perfil fuerte para clases de un solo señante |
| Evaluación | LOSO | LOSO (1-64 entre personas) + `--cv reps` (clases del 012) + métricas por grupo |

El orden del catálogo es el de los videos: `065` Hoy, `066` Ayer, `067`
Mañana, `068` A, `069` B, `070` C, `071` CH, `072` D … `094` Z. No editar el
catálogo de un modelo ya entrenado: modelo, catálogo, manifiesto y fixture se
regeneran juntos.

## Límite importante

Las clases 65-94 las grabó **una sola persona** (sujeto 012), con 4
repeticiones. Lo que mide `--cv reps` es "otra repetición de la misma
persona": es optimista. Para saber cómo anda con otra gente, grabar el
abecedario y Hoy/Ayer/Mañana con 1 o 2 personas más (aunque sean 2-3
repeticiones) y medir con `evaluar_videos.py`.

Además, en los videos del 012 MediaPipe **pierde la mano** en varias letras
mientras la forma está sostenida (ver `data/processed_s012/qa.json`): la mano
está en cuadro, pero queda chica, clara sobre camisa clara y delante del
pecho o la cara. Recortar el cuadro no lo arregla. Para grabaciones nuevas:
ropa oscura y lisa, más cerca de la cámara, la mano un poco separada del
cuerpo y buena luz.

## Videos

Nombres estrictos `SSS_PPP_RRR.<ext>` (seña, persona, repetición: tres
dígitos y guion bajo). Se aceptan `.mp4`, `.mov`/`.MOV`, `.m4v`, `.avi`,
`.mkv`, `.webm`. Un nombre inválido aborta el preprocesado y lo lista.

## Comandos

Todo desde esta carpeta, con el `venv` del repo:

```powershell
Set-Location -LiteralPath 'D:\2026-Politecnico\helpi\entrenamiento-modelo\lsa-94'

# 1. Pruebas
..\venv\Scripts\python.exe .\test_eva_contract.py
..\venv\Scripts\python.exe .\test_entrenamiento.py
..\venv\Scripts\python.exe .\augment.py

# 2. Keypoints de los videos nuevos (las 3200 de LSA64 se reutilizan de ..\data\processed_v3)
..\venv\Scripts\python.exe .\preprocess_tasks.py --videos .\data\raw\sujeto012 --out .\data\processed_s012 --model .\assets\holistic_landmarker.task

# 3. Dataset combinado (3576 muestras, 94 clases)
..\venv\Scripts\python.exe .\combinar_datasets.py --fuentes ..\data\processed_v3 .\data\processed_s012 --out .\data\processed_94

# 4. A/B de las mejoras (varias semillas; retoma si se corta)
..\venv\Scripts\python.exe .\ab_augment.py --data .\data\processed_94 --seeds 2 --epochs 80

# 5. Validación cruzada con la configuración elegida
..\venv\Scripts\python.exe .\train_keras_tflite.py --data .\data\processed_94 --out .\runs --cv reps --epochs 100
..\venv\Scripts\python.exe .\train_keras_tflite.py --data .\data\processed_94 --out .\runs --cv loso --epochs 100

# 6. Modelo final con todos los datos y exportación
..\venv\Scripts\python.exe .\train_keras_tflite.py --data .\data\processed_94 --out .\exports --final --epochs <N>

# 7. Comparar contra v3 con videos de otras personas
..\venv\Scripts\python.exe .\evaluar_videos.py --videos <carpeta> --modelos ..\lsa64\exports .\exports
```

Épocas del paso 6: cada CV guarda `epocaSugerida` y `curvaMedia` en su JSON.
Tomar la época donde la media de las dos curvas (reps y loso) es máxima.

Criterio de aceptación: el LOSO de las clases 1-64 no debe caer respecto de
v3 (0,989), y la configuración elegida debe tener la mejor accuracy del grupo
`abecedario` en `--cv reps`.

Para iterar rápido está el modo split (sin `--cv` ni `--final`): val =
sujeto 9 + repetición 3 del 012, test = sujeto 10 + repetición 4 del 012.

Flags para experimentos: `--features v3`, `--balance 1` (natural) o `0`
(balanceado), `--sin-perfil-fuerte`, `--sin-aug-mano`, `--ema`, `--etiqueta`.

## Artefactos para Android

```text
modelo_lsa.tflite, catalogo_senas.json, lsa-manifest.json  -> app/src/main/assets/lsa/
fixture_android.json                                        -> app/src/androidTest/assets/
```

El manifiesto declara `numClasses: 94`, `catalogVersion: "3.0"` y
`contractVersion: 3`. **No copiarlos todavía a la app**: helpi asume 64 señas
en el léxico, el CI (`verify.py`), varios tests, el texto de la cuenta y el
protocolo de evaluación con el backend. Esa integración es un trabajo aparte.
