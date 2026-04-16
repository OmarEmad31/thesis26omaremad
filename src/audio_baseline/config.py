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
# We revert back to the pristine Emotion backbone. It intrinsically understands emotion clusters,
# and we will use LoRA to specifically translate those boundary mappings into Egyptian Arabic rhythm.
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS (Matched to Text Baseline)
# ---------------------------------------------------------------------------
SEED = 42

SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10   # Max seconds of audio per sample
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

BATCH_SIZE        = 8     # Safe for LoRA computation
GRAD_ACCUM_STEPS  = 4     # 8 * 4 = 32 effective batch size
NUM_EPOCHS        = 40
LEARNING_RATE     = 3e-4  # Optimal PEFT adapter learning rate
EARLY_STOP_PATIENCE = 5   # Stop if F1 doesn't improve for 5 epochs
WEIGHT_DECAY      = 0.05
WARMUP_RATIO      = 0.1

# LoRA / PEFT Settings for Egyptian Arabic Dialectal Shift
LORA_R            = 16    # Rank of the adapter injected into the attention layers
LORA_ALPHA        = 32    # Scaling factor
LORA_DROPOUT      = 0.1

# SCL Settings
USE_SCL           = True
SCL_TEMP          = 0.1
SCL_WEIGHT        = 0.1

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
