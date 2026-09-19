# Entrenamiento de LSA-T en Windows

Esta guía ejecuta todo desde PowerShell en Windows. No usa SSH ni WSL.
Trabaja sobre el repositorio `entrenamiento-modelo` y deja los datos,
checkpoints y exports en la raíz del repositorio.

El entrenamiento detecta automáticamente, en este orden, Intel XPU, NVIDIA
CUDA o CPU. Para un entrenamiento completo se recomienda una GPU NVIDIA con
CUDA; CPU sirve solamente para verificar la instalación. La elección actual
de wheels de PyTorch debe confirmarse en la [página oficial de instalación
local](https://docs.pytorch.org/get-started/locally/) si la versión de CUDA de
la computadora es distinta.

## 1. Abrir PowerShell y ubicarse en el repositorio

Reemplazar la ruta por la carpeta real donde se clonó el proyecto:

```powershell
Set-Location -LiteralPath 'C:\ruta\entrenamiento-modelo'
$env:LSA_PROJECT_DIR = (Get-Location).Path
```

Comprobar que los scripts existen:

```powershell
Test-Path .\lsa\01_prepare_data.py
Test-Path .\lsa\02_train.py
Test-Path .\lsa\03_evaluate.py
Test-Path .\lsa\04_export.py
```

## 2. Crear y activar el entorno Python

Se recomienda Python 3.12. El proyecto no debe ejecutarse con Python 3.14
cuando se necesite soporte XPU/IPEX.

```powershell
py -3.12 --version
py -3.12 -m venv .venv

# Solo para esta ventana de PowerShell.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1

python --version
python -m pip install --upgrade pip setuptools wheel
```

Si PowerShell informa que no puede ejecutar scripts, cerrar y abrir otra
ventana o repetir el `Set-ExecutionPolicy` anterior. No es necesario cambiar
la política de forma permanente.

## 3. Instalar PyTorch

Elegir una sola de estas opciones dentro del entorno activado.

### Opción A: NVIDIA CUDA

El ejemplo usa CUDA 12.8. Si la página oficial de PyTorch indica otro índice
para la GPU instalada, usar el comando que genere esa página.

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Opción B: sin GPU / CPU

Esta opción permite hacer smoke tests, pero el entrenamiento completo será muy
lento.

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verificar el dispositivo

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('XPU:', hasattr(torch, 'xpu') and torch.xpu.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ninguna CUDA')"
```

En Windows no instalar los comandos Linux `sudo`, `dnf` ni el runtime Level
Zero. Si la computadora tiene Intel Arc y PyTorch no informa XPU, el script
seguirá con CUDA o CPU; no se debe confundir eso con un entrenamiento acelerado.

## 4. Instalar las dependencias del proyecto

```powershell
python -m pip install -r .\lsa\requirements.txt
```

Verificar las importaciones principales:

```powershell
python -c "import numpy, pandas, torch, onnx, onnxruntime; print('Dependencias principales: OK')"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

Si la instalación falla específicamente en `pyzes` y no se va a usar Intel XPU,
instalar el resto sin esa dependencia:

```powershell
$tmpReq = Join-Path $env:TEMP 'lsa-requirements-windows.txt'
Get-Content .\lsa\requirements.txt | Where-Object { $_ -notmatch '^\s*pyzes(\s|$)' } | Set-Content -Encoding utf8 $tmpReq
python -m pip install -r $tmpReq
Remove-Item -LiteralPath $tmpReq -Force
```

## 5. Confirmar la ubicación de datos y checkpoints

El código usa la raíz indicada por `LSA_PROJECT_DIR`. Si no se define, usa la
carpeta padre de `lsa` automáticamente.

```powershell
Write-Host "Proyecto: $env:LSA_PROJECT_DIR"
Get-ChildItem .\data\processed -Directory -ErrorAction SilentlyContinue
Get-ChildItem .\checkpoints -Filter '*.pt' -ErrorAction SilentlyContinue
```

En esta copia ya existe un dataset limpio. Sus conteos esperados son:

```powershell
@('train', 'val', 'test') | ForEach-Object {
    $n = (Get-ChildItem ".\data\processed\$_" -Filter '*.npz' -ErrorAction SilentlyContinue).Count
    Write-Host "$_ : $n muestras"
}
```

No volver a ejecutar `01_prepare_data.py` sobre esa copia limpia salvo que se
quiera regenerar el dataset desde cero: volvería a poblar muestras que fueron
puestas en cuarentena.

## 6. Preparar los datos desde cero (solo si faltan)

Usar esta sección únicamente si no existen `data\processed\train`, `val` y
`test` con archivos `.npz`.

### LSA-T

Descarga las anotaciones y poses desde Hugging Face y genera el vocabulario:

```powershell
python .\lsa\01_prepare_data.py
```

### LSA-T más LSA-X opcional

LSA-X agrega una descarga grande y se usa para el preentrenamiento opcional de
la fase A:

```powershell
python .\lsa\01_prepare_data.py --lsax
```

Después de preparar datos nuevos, auditar antes de entrenar:

```powershell
python .\lsa\05_audit_data.py --report-dir reports\phase1
Get-Content .\reports\phase1\data_audit.md
```

La limpieza es recuperable. Primero mostrar el plan sin mover archivos:

```powershell
python .\lsa\06_clean_data.py
Get-Content .\reports\phase1\cleaning_report.md
```

Solo después de revisar el plan, aplicar la cuarentena:

```powershell
python .\lsa\06_clean_data.py --apply
```

El script mueve muestras a `data\quarantine\phase1`; no las elimina. No
ejecutar `--apply` otra vez sobre el mismo reporte: las muestras ya movidas no
están en `data\processed`.

## 7. Congelar evidencia y verificar el contrato

Estos comandos no entrenan. Guardan hashes y generan la referencia compartida
entre Python y Android:

```powershell
python .\lsa\00_freeze_baseline.py
python .\lsa\07_write_sequence_fixture.py
Get-Content .\reports\phase0\baseline.md
Get-Content .\reports\phase2\sequence_contract.md
```

El contrato debe mantenerse en `75` cuadros por `126` coordenadas, con
muestreo entero y padding temporal con ceros.

## 8. Smoke test antes del entrenamiento largo

El smoke test ejecuta solo cinco pasos por época y sirve para confirmar que el
modelo, el vocabulario, los datos y el dispositivo funcionan.

```powershell
python .\lsa\02_train.py --phase b --smoke --fresh
```

Debe mostrar `device=cuda` si se instaló una GPU NVIDIA correctamente. Si
muestra `device=cpu`, detenerse y corregir la instalación antes de lanzar un
entrenamiento de muchas horas.

## 9. Entrenamiento principal — Fase B

La fase B es el entrenamiento principal sobre LSA-T. `--fresh` evita cargar un
checkpoint viejo que haya sido producido con otro preprocesamiento.

```powershell
New-Item -ItemType Directory -Force .\logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = ".\logs\phase_b_$stamp.log"
python .\lsa\02_train.py --phase b --fresh 2>&1 | Tee-Object -FilePath $log
```

El entrenamiento guarda:

```text
checkpoints\phase_b_last.pt
checkpoints\phase_b_best.pt
checkpoints\phase_b_history.json
```

Comprobar que se generaron:

```powershell
Get-ChildItem .\checkpoints\phase_b_*
```

### Reanudar una Fase B interrumpida

Si ya existe `phase_b_last.pt` de la misma configuración, reanudar sin
`--fresh`:

```powershell
python .\lsa\02_train.py --phase b
```

No usar `--fresh` para reanudar; ese parámetro empieza una corrida nueva.

## 10. Fase A opcional — preentrenamiento LSA-X

Solo ejecutar si se preparó `data\processed\pretrain` con `--lsax`:

```powershell
python .\lsa\02_train.py --phase a --epochs 25 --fresh
```

Después, la fase B puede iniciar desde `phase_a_best.pt`:

```powershell
python .\lsa\02_train.py --phase b --fresh
```

## 11. Fase C opcional — fine-tuning con augmentación

La fase C parte del mejor checkpoint de B cuando no existe un checkpoint C
previo. Usar `--fresh` evita reanudar accidentalmente una fase C vieja, pero
conserva la carga de `phase_b_best.pt` prevista por el script:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = ".\logs\phase_c_$stamp.log"
python .\lsa\02_train.py --phase c --epochs 20 --fresh 2>&1 | Tee-Object -FilePath $log
```

El resultado esperado es:

```text
checkpoints\phase_c_last.pt
checkpoints\phase_c_best.pt
checkpoints\phase_c_history.json
```

## 12. Evaluar el checkpoint

Evaluar una sola versión candidata sobre `test`:

```powershell
python .\lsa\03_evaluate.py --ckpt phase_b_best.pt --examples 20
```

Si se promovió la fase C como candidata:

```powershell
python .\lsa\03_evaluate.py --ckpt phase_c_best.pt --examples 20
```

El script muestra BLEU-4, WER, ROUGE-L y ejemplos cualitativos. Revisar el
archivo generado:

```powershell
Get-ChildItem .\checkpoints\*_eval.json
```

No reemplazar los umbrales después de ver el resultado. Si la calidad no es
suficiente, el checkpoint queda como prototipo y no debe pasar a Android.

## 13. Exportar para Android

Elegir el checkpoint que haya pasado la evaluación:

```powershell
python .\lsa\04_export.py --ckpt phase_b_best.pt
```

O, si la fase C es la candidata:

```powershell
python .\lsa\04_export.py --ckpt phase_c_best.pt
```

La exportación completa produce en `exports`:

```text
encoder.onnx
decoder.onnx
encoder_int8.onnx
decoder_int8.onnx
encoder_int8.tflite
decoder_int8.tflite
vocab.json
android-manifest.json
fixture_android.json
```

Verificar los cinco archivos que forman el paquete Android:

```powershell
$required = @(
    'encoder_int8.tflite',
    'decoder_int8.tflite',
    'vocab.json',
    'android-manifest.json',
    'fixture_android.json'
)
$required | ForEach-Object {
    $path = Join-Path (Join-Path (Get-Location) 'exports') $_
    if (Test-Path -LiteralPath $path) {
        Get-FileHash -LiteralPath $path -Algorithm SHA256
    } else {
        Write-Error "Falta exports\$_"
    }
}
```

Si solo se necesita diagnosticar ONNX y todavía no se puede convertir a
LiteRT/TFLite:

```powershell
python .\lsa\04_export.py --ckpt phase_b_best.pt --skip-tflite
```

Ese comando no genera un paquete Android completo. Para integrar en la app se
necesitan ambos `.tflite`, el vocabulario, el manifiesto y el fixture del mismo
checkpoint.

## 14. Orden recomendado completo

Para una computadora Windows nueva, el orden resumido es:

```powershell
Set-Location -LiteralPath 'C:\ruta\entrenamiento-modelo'
$env:LSA_PROJECT_DIR = (Get-Location).Path
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r .\lsa\requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python .\lsa\05_audit_data.py
python .\lsa\07_write_sequence_fixture.py
python .\lsa\02_train.py --phase b --smoke --fresh
python .\lsa\02_train.py --phase b --fresh
python .\lsa\03_evaluate.py --ckpt phase_b_best.pt --examples 20
python .\lsa\04_export.py --ckpt phase_b_best.pt
```

Si el dataset no existe, ejecutar `python .\lsa\01_prepare_data.py` antes de
la auditoría. Si se usa CPU, detenerse después del smoke test salvo que se
acepte un tiempo de entrenamiento mucho mayor.

## 15. Problemas frecuentes

### `python` no encuentra el comando

Cerrar y abrir PowerShell después de instalar Python, o usar el lanzador:

```powershell
py -3.12 --version
```

### El script entrena en CPU

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin CUDA')"
nvidia-smi
```

Si `nvidia-smi` no existe o `torch.cuda.is_available()` es `False`, revisar el
driver NVIDIA y reinstalar el wheel de PyTorch para la versión CUDA correcta.

### No encuentra `data` o `vocab.json`

```powershell
Write-Host $env:LSA_PROJECT_DIR
Get-ChildItem .\data\processed\train -Filter '*.npz'
Get-Item .\data\vocab.json
```

Si faltan, preparar los datos con `01_prepare_data.py`. No cambiar el código
para apuntar a una ruta Linux.

### Carga un checkpoint viejo o incompatible

Para iniciar una versión nueva desde cero:

```powershell
python .\lsa\02_train.py --phase b --fresh
```

Para continuar una corrida interrumpida de la misma versión, quitar `--fresh`.

### La exportación no genera `.tflite`

Ejecutar primero `--skip-tflite` para comprobar ONNX, guardar el error completo
y no copiar exports parciales a Android. La app necesita un paquete completo y
validado; no debe mezclarse encoder de una corrida con decoder de otra.
