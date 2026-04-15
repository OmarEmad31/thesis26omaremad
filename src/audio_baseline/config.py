import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    print("🕵️ Hunting for data folders...")
    search_roots = [Path("/content/Thesis Project"), Path("/content")]
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
                        DATA_ROOT = p.parent
                        break
                if DATA_ROOT:
                    break
    if not DATA_ROOT:
        DATA_ROOT = Path("/content/Thesis Project")
    print(f"✅ DATA_ROOT Locked: {DATA_ROOT}")

    SPLIT_CSV_DIR  = Path("/content/data/processed/splits/audio_eligible")
    CHECKPOINT_DIR = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_power_mode")
    # Dual-extraction cache (stores both SVM + MLP embeddings)
    EMBEDDING_CACHE = Path("/content/audio_embeddings_dual.npz")

else:
    DATA_ROOT       = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR   = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR  = Path("D:/thesis_checkpoints/audio_power_mode")
    EMBEDDING_CACHE = Path("audio_embeddings_dual.npz")

# ---------------------------------------------------------------------------
# BACKBONE
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_base"

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# EMBEDDING STRATEGY  (key insight from experiments)
#
# SVM  → utterance-level (768-dim), CLEAN original data only
#         Reason: SVM needs few, high-quality, diverse samples.
#                 Augmented duplicates & high-dim features both hurt SVM.
#
# MLP  → frame-level concat(mean,std,max) = 2304-dim, AUGMENTED training
#         Reason: MLP benefits from temporal info and more training samples.
# ---------------------------------------------------------------------------
SVM_GRANULARITY = "utterance"   # 768-dim — proven best for SVM on this data
MLP_GRANULARITY = "frame"       # 2304-dim (after mean+std+max aggregation)

AUGMENT_MLP_TRAIN = True        # 4× training samples for MLP only
AUG_SPEED_RATES   = [0.85, 1.15]
AUG_NOISE_STD     = 0.003

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
EPOCHS        = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 5e-3
LOG_EVERY     = 10
MAX_PATIENCE  = 40

# ---------------------------------------------------------------------------
# LOSS / AUGMENTATION IN TRAINING LOOP
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
