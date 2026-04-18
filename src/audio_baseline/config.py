import os
from pathlib import Path

# Base Paths (Configured for root project directory)
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_ROOT = ROOT_DIR / "dataset" / "Final Modalink Dataset MERGED"
SPLIT_CSV_DIR = ROOT_DIR / "data" / "processed" / "splits" / "audio_eligible"

# Results & Checkpoints
RESULTS_DIR = ROOT_DIR / "results" / "audio_baseline"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "audio_sota"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Dataset properties
SR = 16000
MAX_DURATION_SEC = 10
NUM_CLASSES = 8

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
LABEL2ID = {name: i for i, name in enumerate(EMOTIONS)}
ID2LABEL = {i: name for i, name in enumerate(EMOTIONS)}
