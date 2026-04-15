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
    # NEW cache name (frame-level + augmented) — old utterance cache won't be loaded
    EMBEDDING_CACHE = Path("/content/audio_embeddings_frame_aug.npz")

else:
    DATA_ROOT       = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR   = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR  = Path("D:/thesis_checkpoints/audio_power_mode")
    EMBEDDING_CACHE = Path("audio_embeddings_frame_aug.npz")

# ---------------------------------------------------------------------------
# BACKBONE — emotion2vec_plus_base proven better than large on this dataset.
# (large gives 1024-dim embeddings that are harder to fit with small N)
# ---------------------------------------------------------------------------
MODEL_NAME = "iic/emotion2vec_plus_base"

# ---------------------------------------------------------------------------
# EMBEDDING EXTRACTION
# ---------------------------------------------------------------------------
# Phase 3: use frame-level embeddings aggregated as concat(mean, std, max)
# This gives 768×3 = 2304-dim — richer than single utterance-level vector
EMBED_GRANULARITY = "frame"   # "frame" or "utterance"

# Phase 2: augment training audio before embedding extraction
# Each train file gets 4 versions → ~2200 training samples instead of 648
AUGMENT_TRAIN    = True
AUG_SPEED_RATES  = [0.85, 1.15]   # time-stretch rates (slow, fast)
AUG_NOISE_STD    = 0.003          # Gaussian noise std multiplier (of signal std)

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLING_RATE     = 16000
MAX_DURATION_SEC  = 10
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# ---------------------------------------------------------------------------
# TRAINING  (tuned for the actual dataset size after manifest rebuild)
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
EPOCHS        = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 5e-3
LOG_EVERY     = 10
MAX_PATIENCE  = 40    # absolute patience; restarts reset this

# ---------------------------------------------------------------------------
# LOSS / REGULARIZATION
# ---------------------------------------------------------------------------
USE_SCL      = True
SCL_WEIGHT   = 0.3
SCL_TEMP     = 0.2    # works well for batches of 64 and 7 classes
FOCAL_GAMMA  = 1.5    # lighter than 2.0; combines with class weights

USE_MIXUP    = True
MIXUP_ALPHA  = 0.2
MIXUP_PROB   = 0.35   # apply to ~35% of batches; rest see clean data

# ---------------------------------------------------------------------------
# TEST-TIME AUGMENTATION
# ---------------------------------------------------------------------------
USE_TTA   = True
TTA_PASSES = 8
TTA_NOISE  = 0.003

# ---------------------------------------------------------------------------
# MISC
# ---------------------------------------------------------------------------
DEVICE = "cuda"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
