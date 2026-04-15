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
# emotion2vec_plus_large gives ~3-5% better accuracy than base
# Switch to large model if you have enough RAM/VRAM on Colab
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_large"   # upgraded: base → large

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# MLP CLASSIFIER TRAINING  (after embeddings are pre-computed)
# ---------------------------------------------------------------------------
EMBEDDING_DIM  = None      # Detected automatically from first extracted sample
BATCH_SIZE     = 128       # Larger batch → more SCL positive pairs per step
EPOCHS         = 200       # More epochs; early stopping prevents overfit
LEARNING_RATE  = 2e-4      # Slightly lower than before for stable convergence
WEIGHT_DECAY   = 1e-2      # Stronger L2 for deeper network
LOG_EVERY      = 20        # Log every N epochs

# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------
USE_SCL    = True
SCL_WEIGHT = 0.4           # Raised from 0.1 → 0.4: SCL now a real contributor
SCL_TEMP   = 0.1           # Lowered from 0.3 → 0.1: tighter cluster separation

# ---------------------------------------------------------------------------
# MIXUP augmentation (embedding-space)
# ---------------------------------------------------------------------------
USE_MIXUP    = True
MIXUP_ALPHA  = 0.3         # Beta dist param; 0.3 gives mild mixing

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------
DEVICE = "cuda"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
