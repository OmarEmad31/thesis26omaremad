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

    SPLIT_CSV_DIR   = Path("/content/data/processed/splits/audio_eligible")
    CHECKPOINT_DIR  = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_power_mode")
    EMBEDDING_CACHE = Path("/content/audio_embeddings.npz")   # Fast local SSD

else:
    DATA_ROOT        = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR    = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR   = Path("D:/thesis_checkpoints/audio_power_mode")
    EMBEDDING_CACHE  = Path("audio_embeddings.npz")

# ---------------------------------------------------------------------------
# BACKBONE  (used for embedding extraction only, always FROZEN)
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_base"

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLING_RATE    = 16000
MAX_DURATION_SEC = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# MLP CLASSIFIER TRAINING  (after embeddings are pre-computed)
# ---------------------------------------------------------------------------
EMBEDDING_DIM  = 768      # emotion2vec+ transformer hidden size
BATCH_SIZE     = 64       # Large batch — tensors are tiny (no audio in memory)
EPOCHS         = 100      # Fast epochs since backbone is frozen
LEARNING_RATE  = 5e-4
WEIGHT_DECAY   = 1e-4

# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------
USE_SCL    = True
SCL_WEIGHT = 0.5          # Equal emphasis on CE and SCL
SCL_TEMP   = 0.07         # Tight clusters → sharper discrimination

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------
DEVICE = "cuda"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
