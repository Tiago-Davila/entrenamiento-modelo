# Eva LSA64 v2 — reentrenamiento sin pose facial

Este pipeline crea una nueva versión de Eva: `40 × 168` coordenadas por
secuencia. Conserva las dos manos y Pose 11..24; descarta Pose 0..10 (nariz,
ojos, orejas y boca) y Pose 25..32 (piernas).

El catálogo oficial está en `catalogo_lsa64.json`. No editar el catálogo que
acompaña a un modelo ya entrenado: cada exportación genera de nuevo el modelo,
el catálogo, el manifiesto y el fixture como una unidad.

## Preparar el dataset

```powershell
Set-Location -LiteralPath 'C:\ruta\entrenamiento-modelo\lsa64'
& .\venv\Scripts\Activate.ps1
python .\preprocess_tasks.py --videos .\data\raw\all --out .\data\processed --model .\assets\holistic_landmarker.task
```

El resultado esperado es `X.npy` con forma `(muestras, 40, 168)`.

## Validar y entrenar

```powershell
python .\test_eva_contract.py
python .\train_keras_tflite.py --data .\data\processed --out .\exports --epochs 120 --augment-factor 2
```

`--augment-factor 2` deja las muestras originales y agrega dos copias
aumentadas solo al split de entrenamiento. Usar `0` únicamente para comparar
contra una línea base sin aumentación.

## Artefactos para Android

La exportación produce estos archivos inseparables:

```text
modelo_lsa.tflite
catalogo_senas.json
lsa-manifest.json
fixture_android.json
metricas.json
```

No copiar aún estos archivos a la aplicación actual: Eva v2 usa 168
coordenadas y la aplicación publicada acepta Eva v1 de 201. Primero debe
actualizarse el contrato Android y pasar el test instrumentado con el fixture
generado por esta misma corrida.
