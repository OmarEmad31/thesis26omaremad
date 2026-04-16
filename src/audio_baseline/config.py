import os
from pathlib import Path

IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
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

    CHECKPOINT_DIR     = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_wavlm")
    OFFLINE_FEATURES_DIR= Path("/content/drive/MyDrive/Thesis Project/data/processed/whisper_offline")
    SER_FEATURES_DIR    = Path("/content/drive/MyDrive/Thesis Project/data/processed/ser_offline")
else:
    DATA_ROOT           = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR       = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR      = Path("D:/thesis_checkpoints/audio_wavlm")
    OFFLINE_FEATURES_DIR= Path("data/processed/whisper_offline")
    SER_FEATURES_DIR    = Path("data/processed/ser_offline")

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION (SOTA TRANSFORMER)
# ---------------------------------------------------------------------------
# We use OpenAI's Whisper Small. Because it was trained broadly on YouTube, it natively clusters Egyptian Arabic cadences organically without needing gated access.
MODEL_NAME        = "openai/whisper-small"
N_MELS            = 128
HOP_LENGTH        = 512

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS (Fast PyTorch MLP)
# ---------------------------------------------------------------------------
SEED = 42

SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10   
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

BATCH_SIZE        = 32    
NUM_EPOCHS        = 100    
LEARNING_RATE     = 3e-3  
EARLY_STOP_PATIENCE = 15  
WEIGHT_DECAY      = 0.05  
WARMUP_RATIO      = 0.1

# SCL Settings
USE_SCL           = True
SCL_TEMP          = 0.1
SCL_WEIGHT        = 0.1

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OFFLINE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
SER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
