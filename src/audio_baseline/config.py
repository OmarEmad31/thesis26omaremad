import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    # --- UNIVERSAL DATA FINDER ---
    # Searches /content/ for the folder that contains 'videoplayback' subfolders
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
        DATA_ROOT = Path("/content/Thesis Project")  # Final Fallback
    print(f"✅ DATA_ROOT Locked: {DATA_ROOT}")

    SPLIT_CSV_DIR = Path("/content/data/processed/splits/audio_eligible")
    CHECKPOINT_DIR = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_power_mode")

else:
    # Standard Local Paths
    DATA_ROOT = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR = Path("D:/thesis_checkpoints/audio_power_mode")

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS  (single source of truth — no duplicates)
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 5e-5           # Backbone gets 10% of this (5e-6) via optimizer groups
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 8 * 4 = 32 samples per update

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_base"

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
SAMPLING_RATE = 16000
MAX_DURATION_SEC = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# SCL (Supervised Contrastive Learning)
# ---------------------------------------------------------------------------
USE_SCL = True
SCL_WEIGHT = 0.3       # Reduced slightly so CE dominates early training
SCL_TEMP = 0.07        # Tighter temperature → sharper emotion clusters

# ---------------------------------------------------------------------------
# FINE-TUNING SCHEDULE
# ---------------------------------------------------------------------------
UNFREEZE_EPOCH = 2     # Unfreeze backbone at Epoch 2

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------
NUM_FOLDS = 1
DEVICE = "cuda"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
