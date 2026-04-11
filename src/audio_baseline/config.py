import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
# Check if we are running in Google Colab
IS_COLAB = "COLAB_GPU" in os.environ or "COLAB_JUPYTER_IP" in os.environ

if IS_COLAB:
    # Path in Google Drive after mounting
    # We assume you put your folders in "My Drive/Thesis Project"
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
