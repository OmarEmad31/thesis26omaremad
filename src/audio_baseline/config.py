import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
# Check if we are running in Google Colab
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    # Check if we unzipped to the local SSD (/content/)
    if os.path.exists("/content/data") and os.path.exists("/content/Thesis Project"):
        print("🚀 Using LOCAL SSD for maximum speed!")
        DRIVE_BASE = Path("/content/Thesis Project")
        DATA_ROOT = DRIVE_BASE / "dataset/Final Modalink Dataset MERGED"
        SPLIT_CSV_DIR = Path("/content/data/processed/splits/text_hc")
        # SAVE TO DRIVE FOR SAFETY
        CHECKPOINT_DIR = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_baseline_emotion2vec")
    else:
        print("📁 Using Google Drive (Slow mode). Tip: Unzip your dataset to /content/ for 10x speed.")
        DRIVE_BASE = Path("/content/drive/MyDrive/Thesis Project")
        DATA_ROOT = DRIVE_BASE / "dataset/Final Modalink Dataset MERGED"
        SPLIT_CSV_DIR = DRIVE_BASE / "data/processed/splits/text_hc"
        CHECKPOINT_DIR = DRIVE_BASE / "checkpoints/audio_baseline_emotion2vec"
else:
    # Standard Local Paths
    DATA_ROOT = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR = Path("data/processed/splits/text_hc")
    CHECKPOINT_DIR = Path("D:/thesis_checkpoints/audio_baseline_emotion2vec")

# --- MODEL CONFIG ---
# Official Alibaba ModelScope/HuggingFace Path
MODEL_NAME = "emotion2vec/emotion2vec_plus_base"

# --- DATA CONFIG ---
# (Paths are now defined above)
SAMPLING_RATE = 16000
MAX_DURATION_SEC = 10  # Truncate clips longer than this
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# --- TRAINING CONFIG ---
BATCH_SIZE = 8       # Audio transformers are heavy; keep batch size small
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# --- FOLD CONFIG ---
NUM_FOLDS = 5
CHECKPOINT_DIR = Path("D:/thesis_checkpoints/audio_baseline_emotion2vec")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- INFERENCE ---
DEVICE = "cuda"  # Will fallback in code if not available
