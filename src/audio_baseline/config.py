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
    EMBEDDING_CACHE = Path("/content/audio_embeddings.npz")

else:
    DATA_ROOT        = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR    = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR   = Path("D:/thesis_checkpoints/audio_power_mode")
    EMBEDDING_CACHE  = Path("audio_embeddings.npz")

# ---------------------------------------------------------------------------
# BACKBONE  (frozen feature extractor — swap to base if VRAM is tight)
# emotion2vec_plus_large → ~1024-dim embeddings, significantly stronger
# emotion2vec_plus_base  → ~768-dim, lighter, use if large OOMs
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_large"

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# MLP CLASSIFIER TRAINING
# ---------------------------------------------------------------------------
EMBEDDING_DIM  = None      # Auto-detected
BATCH_SIZE     = 128       # Large batch → more SCL positive pairs
EPOCHS         = 300       # More epochs; early stopping guards against overfit
LEARNING_RATE  = 2e-4
WEIGHT_DECAY   = 1e-2
LOG_EVERY      = 25

# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------
USE_SCL    = True
SCL_WEIGHT = 0.4           # Real contributor (was 0.1)
SCL_TEMP   = 0.07          # Standard SupCon paper value — tight cluster separation
FOCAL_GAMMA = 2.0          # Focal loss γ: down-weights easy/majority-class predictions

# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------
USE_MIXUP        = True
MIXUP_ALPHA      = 0.3
MIXUP_PROB       = 0.7     # Apply MixUp to 70% of batches
NOISE_STD        = 0.005   # Gaussian noise injection during training
USE_OVERSAMPLING = True    # Random oversample minority classes → balanced training

# ---------------------------------------------------------------------------
# TEST-TIME AUGMENTATION (TTA)
# ---------------------------------------------------------------------------
USE_TTA      = True
TTA_PASSES   = 8           # Average predictions over N noisy forward passes
TTA_NOISE    = 0.005       # Noise std during TTA

# ---------------------------------------------------------------------------
# TRAINING CONTROL
# ---------------------------------------------------------------------------
MAX_PATIENCE = 60          # Much more patience than before (was 25)
DEVICE       = "cuda"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
