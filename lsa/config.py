"""
config.py — Configuracion centralizada del proyecto LSA
Modificar aqui antes de correr cualquier script.
"""
from pathlib import Path

# ── Directorios ───────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
EXPORTS_DIR     = BASE_DIR / "exports"

for d in [DATA_DIR, CHECKPOINTS_DIR, EXPORTS_DIR]:
    d.mkdir(exist_ok=True)

# ── Vocabulario ───────────────────────────────────────────────────────────────
VOCAB_PATH  = DATA_DIR / "vocab.json"
PAD_ID      = 0
BOS_ID      = 1
EOS_ID      = 2
UNK_ID      = 3
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
MIN_FREQ    = 2     # filtrar palabras con frecuencia < MIN_FREQ (singletons)

# ── Preprocesamiento de keypoints ─────────────────────────────────────────────
MAX_FRAMES  = 75    # frames por clip (75 @ 30fps = 2.5 seg)
INPUT_DIM   = 126   # 21 landmarks mano izq * 3 coords + 21 mano der * 3 coords
# Indices de landmarks de manos en MediaPipe Holistic (543 total)
# Pose: 0-32 (33), Face: 33-500 (468), Left hand: 501-521 (21), Right: 522-542 (21)
MP_LEFT_HAND_START  = 501
MP_RIGHT_HAND_START = 522
MP_HAND_LANDMARKS   = 21

# ── Arquitectura del modelo ───────────────────────────────────────────────────
D_MODEL         = 256
NHEAD           = 8
NUM_ENC_LAYERS  = 6
NUM_DEC_LAYERS  = 3
DIM_FFN         = 1024
DROPOUT         = 0.1
MAX_DEC_LEN     = 64    # longitud maxima de la traduccion generada

# ── Entrenamiento ─────────────────────────────────────────────────────────────
# Recomendado para Intel Arc 140V (menor BW que NVIDIA)
BATCH_SIZE      = 32
EPOCHS          = 100
LR              = 5e-4         # fase A (LSA-X) y B (LSA-T): ajustar en cada fase
LR_WARMUP_RATIO = 0.10         # 10% de los steps totales como warmup
WEIGHT_DECAY    = 0.01
GRAD_CLIP       = 1.0
PATIENCE        = 15           # early stopping sobre BLEU-4 en validacion
LABEL_SMOOTHING = 0.1

# ── Cuantizacion ─────────────────────────────────────────────────────────────
CALIB_SAMPLES   = 400          # muestras del val set para calibracion INT8

# ── Dataset LSA-X (opcional, pre-entrenamiento) ───────────────────────────────
LSAX_ZENODO_BASE = "https://zenodo.org/api/records/19087120/files"
LSAX_SUBSETS = [
    "CNSordos_labeled.h5",
    "Videolibros(private)_labeled.h5",
]
