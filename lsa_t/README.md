# LSA — Modelo de traduccion de Lengua de Señas Argentina

Pipeline completo para entrenar un modelo Encoder-Decoder Transformer
que traduce LSA a texto en español, optimizado para correr offline en Android.

**Hardware de este proyecto:**
- CPU: Intel Core Ultra 7 258V
- GPU: Intel Arc 140V (integrada, Xe2)
- RAM: 30 GB (unificada CPU/GPU)

**Datasets usados:**
- LSA-T: `pedroodb/glosl-lsat` en HuggingFace (MIT, 14.880 clips, MediaPipe)
- LSA-X: Zenodo `records/19087120` (CC-BY-NC-ND, 151h, COCO format)

---

## Estructura del proyecto

```
LSA/
├── config.py             # Todos los hiperparametros (editar antes de entrenar)
├── model.py              # Arquitectura: Encoder Transformer + CTC + Decoder
├── dataset.py            # Dataset, Vocabulario, DataLoaders, augmentacion
├── 01_prepare_data.py    # Descargar y preprocesar datasets
├── 02_train.py           # Entrenamiento con IPEX/XPU (Arc 140V)
├── 03_evaluate.py        # Evaluar BLEU-4, WER, ROUGE-L en test set
├── 04_export.py          # Exportar: PyTorch -> ONNX -> INT8 -> TFLite
├── requirements.txt
├── data/                 # Datasets descargados y procesados (generado)
├── checkpoints/          # Checkpoints del modelo (generado)
└── exports/              # Modelos ONNX y TFLite para Android (generado)
```

`data_seed/`, `checkpoints_legacy/` y `exports_legacy/` conservan una muestra
y artefactos históricos. Sirven para auditoría; los scripts activos usan solo
`data/`, `checkpoints/` y `exports/`.

---

## Instalacion

### Paso 1: Instalar dependencias del sistema (una sola vez, como root)

```bash
# Runtime de GPU Intel (Level Zero + compute runtime)
sudo dnf install -y level-zero level-zero-devel intel-compute-runtime

# Si intel-compute-runtime no esta en los repos de Fedora:
sudo dnf copr enable -y jdanecki/intel-opencl
sudo dnf install -y intel-compute-runtime
```

### Paso 2: Crear entorno virtual con Python 3.12

> **Importante:** IPEX XPU no tiene wheels para Python 3.14 todavia.
> Usar Python 3.12 que es la version soportada.

```bash
sudo dnf install -y python3.12 python3.12-venv

python3.12 -m venv venv312
source venv312/bin/activate

python --version   # debe mostrar Python 3.12.x
```

### Paso 3: Instalar PyTorch con XPU nativo

Desde PyTorch 2.7+, el soporte para Intel Arc (XPU) esta integrado directamente
en los builds oficiales de PyTorch. No hace falta intel-extension-for-pytorch.

El URL correcto es el de PyTorch oficial con sufijo /xpu, NO el servidor de Intel
(`pytorch-extension.intel.com`) que solo tiene versiones antiguas (2.5.1).

```bash
# Limpiar cualquier instalacion previa
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y

# Instalar PyTorch con XPU nativo (desde PyTorch oficial)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

# Verificar: debe mostrar 2.14.0+xpu (o la version mas nueva) y XPU: True
python -c "import torch; print(torch.__version__); print('XPU:', torch.xpu.is_available())"
```

IPEX es opcional. Si se instala, el entrenamiento usa sus optimizaciones
adicionales. Si no se instala, `torch.xpu` funciona de forma nativa.

> **Si usas Kaggle o Colab:** PyTorch con CUDA ya viene instalado. Saltar al Paso 4.

### Paso 4: Instalar el resto de dependencias

```bash
pip install -r requirements.txt
# pyzes esta incluido: es la dependencia que torch.xpu necesita para
# enumerar GPUs Intel via Level Zero Sysman API (sin el, XPU: False)
```

### Paso 5: Verificar que el Arc 140V es visible

```bash
python -c "
import torch
import intel_extension_for_pytorch as ipex
print('XPU disponible:', torch.xpu.is_available())
if torch.xpu.is_available():
    print('GPU:', torch.xpu.get_device_name(0))
"
```

Salida esperada:
```
XPU disponible: True
GPU: Intel(R) Arc(TM) 140V Graphics
```

---

## Pipeline de entrenamiento

### 1. Preparar datos

**Solo LSA-T** (recomendado para empezar, ~2h de descarga):

```bash
python 01_prepare_data.py
```

**LSA-T + LSA-X** (plan completo, descarga adicional ~5 GB):

```bash
python 01_prepare_data.py --lsax
```

Esto genera en `data/processed/`:
- `train/` — ~11.900 archivos `.npz`
- `val/`   — ~1.490 archivos `.npz`
- `test/`  — ~1.490 archivos `.npz`
- `pretrain/` — muestras LSA-X (si se uso `--lsax`)
- `../vocab.json` — vocabulario (~7.000 tokens)

### 2. Verificar el setup (smoke test, ~30 segundos)

Antes de lanzar el entrenamiento completo, verificar que todo funciona:

```bash
python 02_train.py --phase b --smoke
```

Deberia correr 5 steps sin error y mostrar la GPU (XPU/CUDA).

### 3. Entrenar

#### Opcion A: Solo en tu PC (Arc 140V) — estimado ~20-30h

```bash
# Fase B: entrenamiento principal (~20-24h en Arc 140V)
python 02_train.py --phase b

# Fase C: fine-tuning con augmentacion (~5-8h en Arc 140V)
python 02_train.py --phase c
```

#### Opcion B (RECOMENDADA): Kaggle gratuito — estimado ~10h total

1. Ir a [kaggle.com/code](https://www.kaggle.com/code) -> New Notebook
2. Settings -> Accelerator: **GPU T4 x2**
3. En el notebook:

```python
# Clonar/subir el proyecto
!pip install -r requirements.txt  # (sin el paso de XPU)

# LSA-T ya esta disponible directamente
!python 01_prepare_data.py

# Fase A (opcional, ~4h en T4): pre-entrenar en LSA-X
# !python 02_train.py --phase a --epochs 25

# Fase B (~5-6h en T4):
!python 02_train.py --phase b

# Fase C (~1-2h en T4):
!python 02_train.py --phase c

# Descargar el checkpoint
from IPython.display import FileLink
FileLink('checkpoints/phase_c_best.pt')
```

#### Opcion C: Con pre-entrenamiento LSA-X (plan completo, 3 fases)

```bash
# Si ya preparaste datos con --lsax:
python 02_train.py --phase a --epochs 25   # ~12-16h en Arc 140V
python 02_train.py --phase b               # ~20-24h en Arc 140V
python 02_train.py --phase c --epochs 20   # ~5-8h en Arc 140V
```

### 4. Evaluar el modelo

```bash
python 03_evaluate.py --ckpt phase_b_best.pt --examples 20
```

**Metricas objetivo:**

| Metrica | Aceptable | Bueno |
|---------|-----------|-------|
| BLEU-4  | > 8       | > 15  |
| WER     | < 40%     | < 25% |
| ROUGE-L | > 0.30    | > 0.45 |

### 5. Exportar para Android

```bash
# Completo: ONNX -> INT8 -> TFLite
python 04_export.py --ckpt phase_b_best.pt

# Solo ONNX + INT8 (sin TFLite, si no tenes tensorflow instalado)
python 04_export.py --skip-tflite
```

Genera en `exports/`:
- `encoder_int8.tflite`  — ~10 MB
- `decoder_int8.tflite`  — ~3 MB
- `vocab.json`           — ~150 KB

Copiar estos 3 archivos + `hand_landmarker.task` (de MediaPipe) a `app/src/main/assets/`.

---

## Tiempos estimados en tu PC (Arc 140V)

| Fase                    | Tiempo estimado | Factor vs RTX 5050 |
|-------------------------|:-:|:-:|
| 01 Preparar datos       | ~3-4 h (descarga + proceso) | — |
| 02 Fase A (LSA-X opt.)  | ~12-20 h | ~4x |
| 02 Fase B (LSA-T)       | ~20-28 h | ~4x |
| 02 Fase C (fine-tune)   | ~5-8 h   | ~4x |
| 03 Evaluar              | ~10 min  | — |
| 04 Exportar a TFLite    | ~30 min  | — |

> Para entrenar en ~10h en lugar de ~50h, usar **Kaggle gratuito** (30h/semana de GPU T4).
> El codigo funciona identico en Kaggle/Colab sin cambios.

---

## Continuar un entrenamiento interrumpido

El script guarda checkpoints automaticamente despues de cada epoca
(`checkpoints/phase_b_last.pt`). Para continuar:

```bash
python 02_train.py --phase b   # retoma automaticamente desde last.pt
```

Para empezar desde cero ignorando checkpoints:

```bash
python 02_train.py --phase b --fresh
```

---

## Modificar hiperparametros

Editar `config.py` antes de correr cualquier script:

```python
BATCH_SIZE    = 32    # reducir a 16 si hay OOM
EPOCHS        = 100
LR            = 5e-4
PATIENCE      = 15    # early stopping
MAX_FRAMES    = 75    # frames por clip (75 = 2.5s @ 30fps)
```

---

## Troubleshooting

**`torch.xpu.is_available()` retorna False:**
```bash
# 1. Verificar que pyzes esta instalado (dependencia clave de torch.xpu)
pip show pyzes || pip install pyzes

# 2. Verificar que torch fue instalado con XPU (no la version de PyPI)
python -c "import torch; print(torch.__version__)"
# Debe terminar en +xpu, ej: 2.14.0+xpu

# 3. Verificar Level Zero
ldconfig -p | grep libze_loader   # debe aparecer
# Si no: sudo dnf install level-zero intel-compute-runtime
```

**`pyzes.py: OSError: libze_loader.so.1: No such file or directory` en Fedora:**
```bash
# pyzes esta compilado para Debian/Ubuntu (busca en /usr/lib/x86_64-linux-gnu/)
# pero en Fedora la libreria vive en /usr/lib64/. Fix: crear symlink.
sudo mkdir -p /usr/lib/x86_64-linux-gnu
sudo ln -s /usr/lib64/libze_loader.so.1 /usr/lib/x86_64-linux-gnu/libze_loader.so.1

# Verificar que funciona:
python -c "import pyzes; r=pyzes.zesInit(0); print(f'zesInit: {r:#010x} (0x0=OK)')"
# Luego: python -c "import torch; print(torch.xpu.is_available())"
```

**`AttributeError: .../libze_loader.so.1: undefined symbol: zesInit`:**
```bash
# El paquete "level-zero" de Fedora viene de un COPR (jdanecki/intel-opencl)
# desactualizado: es la v1.5.0 (2021) y esa version del loader todavia no
# exportaba zesInit (el Sysman API que pyzes/torch.xpu necesitan). Fedora no
# tiene "level-zero" en sus repos oficiales, asi que hay que compilarlo:

git clone --depth 1 --branch v1.33.1 https://github.com/oneapi-src/level-zero.git /tmp/level-zero
cd /tmp/level-zero
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
make -j$(nproc)
sudo make install
sudo ldconfig

# Verificar que el simbolo ya existe y que XPU funciona:
nm -D /usr/lib64/libze_loader.so.1 | grep zesInit
python -c "import torch; print(torch.xpu.is_available())"
```

**Error "No matching distribution found for intel-extension-for-pytorch":**
```bash
# Estas usando Python 3.14 — IPEX no soporta 3.14 aun
# Crear venv con Python 3.12:
sudo dnf install python3.12 python3.12-venv
python3.12 -m venv venv312 && source venv312/bin/activate
# Luego reinstalar desde el index de Intel (Paso 3)
```

**Error "needs PyTorch 2.8.x but 2.14.0+cu130 is found" o "sympy not found":**
```bash
# El servidor pytorch-extension.intel.com esta desactualizado (solo tiene 2.5.1).
# Usar el URL oficial de PyTorch en su lugar:
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu
```

**`pose-format` da error al leer el dataset:**
El dataset `glosl-lsat` puede actualizar su formato. Ver issues en:
https://github.com/sign-language-processing/pose-format

**Error OOM (Out of Memory) en XPU:**
Reducir `BATCH_SIZE` a 16 en `config.py`.

**Entrenamiento muy lento (< 1 step/seg en XPU):**
Verificar que IPEX esta instalado y que el modelo esta en XPU:
```python
import torch, intel_extension_for_pytorch as ipex
print(torch.xpu.is_available())   # debe ser True
```

**`onnx2tf` falla en la conversion a TFLite:**
```bash
pip install onnx2tf tensorflow --upgrade
# Si sigue fallando, usar solo el ONNX INT8 con ONNX Runtime en Android
# en lugar de TFLite (latencia similar)
```

---

## Referencias

- LSA-T dataset: Ronchetti et al. IBERAMIA 2022 — https://huggingface.co/datasets/pedroodb/glosl-lsat
- LSA-X dataset: Zenodo 2026 — https://zenodo.org/records/19087120
- Intel Extension for PyTorch (IPEX): https://intel.github.io/intel-extension-for-pytorch/
- PyTorch XPU: https://pytorch.org/blog/pytorch-2-7-intel-gpus/
- MediaPipe Tasks Android: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- LiteRT (antes TFLite): https://ai.google.dev/edge/litert
