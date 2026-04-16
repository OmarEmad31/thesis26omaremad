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

    CHECKPOINT_DIR  = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_hf")
else:
    DATA_ROOT       = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR   = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR  = Path("D:/thesis_checkpoints/audio_hf")

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION (PURE HUGGINGFACE)
# ---------------------------------------------------------------------------
# We are using OpenAI's Whisper model. Specifically, we will extract just its Acoustic Encoder.
# Since it was trained on millions of hours of dialectal Arabic content, it inherently
# clusters Egyptian colloquial tones and cadences perfectly mapped to underlying semantics.
MODEL_NAME = "openai/whisper-small"

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS (Matched to Text Baseline)
# ---------------------------------------------------------------------------
SEED = 42

SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10   # Max seconds of audio per sample (Whisper handles padding intrinsically)
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

BATCH_SIZE        = 8     # Small memory footprint since base is frozen
GRAD_ACCUM_STEPS  = 4     # 8 * 4 = 32 effective batch size
NUM_EPOCHS        = 40
LEARNING_RATE     = 1e-3  # High learning rate to rapidly train the fresh classification head
EARLY_STOP_PATIENCE = 5   # Stop if F1 doesn't improve for 5 epochs
WEIGHT_DECAY      = 0.05
WARMUP_RATIO      = 0.1

# SCL Settings
USE_SCL           = True
SCL_TEMP          = 0.1
SCL_WEIGHT        = 0.1

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
