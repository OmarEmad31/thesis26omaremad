import os
from pathlib import Path

# --- ENVIRONMENT DETECTION ---
# Check if we are running in Google Colab
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    # --- UNIVERSAL DATA FINDER ---
    # We look for where the audio is actually hiding on the SSD
    print("🕵️‍♂️ Hunting for data folders...")
    search_roots = [Path("/content/Thesis Project"), Path("/content")]
    DATA_ROOT = None
    
    for root in search_roots:
        if root.exists():
            # Does this root or its subfolders contain a .wav?
            wavs = list(root.glob("**/*.wav"))
            if wavs:
                # We found audio! The root is the parent of the first wav's folder structure
                # Actually, the best DATA_ROOT is the folder that contains 'videoplayback' folders
                for w in wavs:
                    if "videoplayback" in str(w):
                        # The parent that contains 'videoplayback (X)' is our root
                        p = w.parent
                        while p.name and "videoplayback" not in p.name:
                            p = p.parent
                        DATA_ROOT = p.parent
                        break
                if DATA_ROOT: break

    if not DATA_ROOT: DATA_ROOT = Path("/content/Thesis Project") # Final Fallback
    print(f"✅ DATA_ROOT Locked: {DATA_ROOT}")
    
    SPLIT_CSV_DIR = Path("/content/data/processed/splits/audio_eligible")
    CHECKPOINT_DIR = Path("/content/drive/MyDrive/Thesis Project/checkpoints/audio_power_mode")
else:
    # Standard Local Paths
    DATA_ROOT = Path("dataset/Final Modalink Dataset MERGED")
    SPLIT_CSV_DIR = Path("data/processed/splits/audio_eligible")
    CHECKPOINT_DIR = Path("D:/thesis_checkpoints/audio_power_mode")

# --- POWER MODE CONFIG ---
USE_SCL = True
SCL_WEIGHT = 0.5         # How much to focus on clustering emotions
UNFREEZE_EPOCH = 2       # Unfreeze earlier for more training time 🚀
SCL_TEMP = 0.1           # Temperature for contrastive loss

# --- MODEL CONFIG ---
class config:
    # Model Architecture
    MODEL_NAME = "iic/emotion2vec_plus_base"
    
    # Training Loop
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 5e-5      # Faster engine for better convergence 🌪️

# --- DATA CONFIG ---
SAMPLING_RATE = 16000
MAX_DURATION_SEC = 10 
MAX_AUDIO_SAMPLES = SAMPLING_RATE * MAX_DURATION_SEC

# --- TRAINING CONFIG ---
BATCH_SIZE = 8       
GRADIENT_ACCUMULATION_STEPS = 4 # Increased for more stability with SCL
EPOCHS = 20                     # Increased for Colab Pro to allow deeper mastery
LEARNING_RATE = 1e-5           # Lower LR for more stable fine-tuning
WEIGHT_DECAY = 0.01

# --- FOLD CONFIG ---
NUM_FOLDS = 1 # We use a single split for faster iteration in Power Mode
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- INFERENCE ---
DEVICE = "cuda" 
