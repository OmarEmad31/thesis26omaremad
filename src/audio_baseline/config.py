import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
# Check if we are running in Google Colab
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    # Check if we unzipped to the local SSD (/content/)
    if os.path.exists("/content/data") and os.path.exists("/content/Thesis Project"):
        print("🚀 [POWER MODE] Using LOCAL SSD with FULL DATASET!")
        DRIVE_BASE = Path("/content/Thesis Project")
        DATA_ROOT = DRIVE_BASE / "dataset/Final Modalink Dataset MERGED"
        # SWITCH TO THE FULL ELIGIBLE DATASET (5,700 CLIPS)
        SPLIT_CSV_DIR = Path("/content/data/processed/splits/audio_eligible")
        # SAVE TO DRIVE FOR SAFETY
        CHECKPOINT_DIR = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_power_mode")
    else:
        print("📁 Using Google Drive (Slow mode). Tip: Unzip your dataset to /content/ for 10x speed.")
        DRIVE_BASE = Path("/content/drive/MyDrive/Thesis Project")
        DATA_ROOT = DRIVE_BASE / "dataset/Final Modalink Dataset MERGED"
        SPLIT_CSV_DIR = DRIVE_BASE / "data/processed/splits/audio_eligible"
        CHECKPOINT_DIR = DRIVE_BASE / "checkpoints/audio_power_mode"
else:
    # Standard Local Paths
    DATA_ROOT = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR = Path("D:/thesis_checkpoints/audio_power_mode")

# --- POWER MODE CONFIG ---
USE_SCL = True
SCL_WEIGHT = 0.5         # How much to focus on clustering emotions
UNFREEZE_EPOCH = 5       # We start deep fine-tuning at this epoch
SCL_TEMP = 0.1           # Temperature for contrastive loss

# --- MODEL CONFIG ---
MODEL_NAME = "emotion2vec/emotion2vec_plus_base"

# --- DATA CONFIG ---
SAMPLING_RATE = 16000
MAX_DURATION_SEC = 10 
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# --- TRAINING CONFIG ---
BATCH_SIZE = 8       
GRADIENT_ACCUMULATION_STEPS = 4 # Increased for more stability with SCL
EPOCHS = 20                     # Increased for Colab Pro to allow deeper mastery
LEARNING_RATE = 1e-5           # Lower LR for more stable fine-tuning
WEIGHT_DECAY = 0.01

# --- FOLD CONFIG ---
NUM_FOLDS = 1 # We use a single split for faster iteration in Power Mode
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- INFERENCE ---
DEVICE = "cuda" 
