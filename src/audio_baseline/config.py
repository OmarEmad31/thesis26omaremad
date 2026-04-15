import os
from pathlib import Path

IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    # Search for the splits CSV directory
    split_search_paths = [
        Path("/content/Thesis Project/data/processed/splits/audio_eligible"),
        Path("/content/data/processed/splits/audio_eligible"),
        Path("/content/drive/MyDrive/Thesis Project/data/processed/splits/audio_eligible")
    ]
    SPLIT_CSV_DIR = None
    for p in split_search_paths:
        if (p / "train.csv").exists():
            SPLIT_CSV_DIR = p
            break
    if not SPLIT_CSV_DIR:
        SPLIT_CSV_DIR = Path("/content/drive/MyDrive/Thesis Project/data/processed/splits/audio_eligible")
    print(f"✅ SPLIT_CSV_DIR: {SPLIT_CSV_DIR}")

    # Search for the audio dataset root
    search_roots = [
        Path("/content/drive/MyDrive/Thesis Project/dataset"),
        Path("/content/drive/MyDrive/Thesis Project"),
        Path("/content/Thesis Project"),
        Path("/content")
    ]
    DATA_ROOT = None
    for root in search_roots:
        if root.exists():
            wavs = list(root.glob("**/*.wav"))
            if wavs:
                for w in wavs:
                    if "videoplayback" in str(w):
                        p = w.parent
                        while p.name and "videoplayback" not in p.name:
                            p = p.parent
                        DATA_ROOT = p.parent; break
                if DATA_ROOT: break
    if not DATA_ROOT:
        DATA_ROOT = Path("/content/drive/MyDrive/Thesis Project")
    print(f"✅ DATA_ROOT: {DATA_ROOT}")

    CHECKPOINT_DIR  = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_v6")
    # Save cache directly to Drive so it survives Colab disconnections!
    EMBEDDING_CACHE = Path("/content/drive/MyDrive/Thesis Project/audio_emb_v6.npz")
else:
    DATA_ROOT       = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR   = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR  = Path("D:/thesis_checkpoints/audio_v6")
    EMBEDDING_CACHE = Path("audio_emb_v6.npz")

# ---------------------------------------------------------------------------
# BACKBONE
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_base"   # proven better than large on this data

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLING_RATE     = 16_000
MAX_DURATION_SEC  = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# EMBEDDING STRATEGY  ← KEY INSIGHT
# SVM+RF+GBM+KNN  →  utterance-level 768-dim, CLEAN data (proven ~42%)
#                     Augmented duplicates + high dim BOTH hurt sklearn models
# MLP             →  frame-level 2304-dim (mean+std+max), AUGMENTED data
#                     Richer temporal features + more training samples
# ---------------------------------------------------------------------------
SVM_GRANULARITY   = "utterance"   # 768-dim  for all sklearn models
MLP_GRANULARITY   = "frame"       # 2304-dim for MLP only

AUGMENT_MLP_TRAIN = True
AUG_SPEED_RATES   = [0.85, 1.15]
AUG_NOISE_STD     = 0.003

# ---------------------------------------------------------------------------
# PHASE 5 — SMOTE  (targets minority classes: Happiness, Surprise, Fear)
USE_SMOTE       = True
SMOTE_TARGET    = 80   # synthesise up to this many samples per class (MLP training)
SMOTE_K         = 5    # KNN neighbours for interpolation

# ---------------------------------------------------------------------------
# MLP TRAINING
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
EPOCHS        = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 5e-3
LOG_EVERY     = 10
MAX_PATIENCE  = 40

# SWA — stochastic weight averaging, starts at SWA_START_FRAC of training
USE_SWA         = True
SWA_START_FRAC  = 0.70   # start averaging at 70% of epochs

# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------
USE_SCL      = True
SCL_WEIGHT   = 0.3
SCL_TEMP     = 0.2
FOCAL_GAMMA  = 1.5

USE_MIXUP    = True
MIXUP_ALPHA  = 0.2
MIXUP_PROB   = 0.35

# ---------------------------------------------------------------------------
# TEST-TIME AUGMENTATION
# ---------------------------------------------------------------------------
USE_TTA    = True
TTA_PASSES = 8
TTA_NOISE  = 0.003

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------
DEVICE = "cuda"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
