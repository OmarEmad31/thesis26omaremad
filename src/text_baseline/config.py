"""Paths and hyperparameters for the MARBERT text baseline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits" / "trackA_cleaned"
TRAIN_CSV = SPLITS_DIR / "trackA_train_clean.csv"
VAL_CSV = SPLITS_DIR / "trackA_val_clean.csv"
TEST_CSV = SPLITS_DIR / "trackA_val_clean.csv" # We pool these for K-Fold in train.py

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "text_sanitized_ensemble"

MODEL_NAME = "UBC-NLP/MARBERT"

TEXT_COLUMN = "transcript"
LABEL_COLUMN = "emotion_final"

MAX_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 15
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 1  # Physical batch is 32 (needed for SCL contrast)
WARMUP_RATIO = 0.3    # Massive 30% warmup to prevent catastrophic forgetting
SEED = 42
