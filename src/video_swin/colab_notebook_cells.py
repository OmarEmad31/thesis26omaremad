# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        VIDEO SWIN TRANSFORMER — COLAB NOTEBOOK (T4 GPU OPTIMISED)          ║
# ║                  Egyptian Arabic Emotion Recognition                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HOW TO USE
# ----------
# 1. Locally, run:
#      python "D:\Thesis Project\scripts\zip_video_for_colab.py"
#    -> Produces  D:\ThesisWork\Thesis_Video_Colab.zip
#
# 2. Upload  Thesis_Video_Colab.zip  to the ROOT of your Google Drive
#    (not inside any subfolder -- just My Drive/).
#
# 3. In Colab: Runtime -> Change runtime type -> T4 GPU
#
# 4. Paste the SINGLE CELL below into a Colab code cell and run it.
#    It handles everything: mount, install, unzip, train.
#
# ================================================================================
# SINGLE ALL-IN-ONE CELL  (copy everything between the triple-quotes)
# ================================================================================
"""
import os, sys, subprocess, zipfile, time, shutil

# ── 0. Mount Google Drive ──────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── 1. Install dependencies ────────────────────────────────────────────────────
print("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "timm==1.0.26", "opencv-python-headless", "scikit-learn"], check=True)
print("Packages ready.")

# ── 2. Unzip to local SSD (fast, skips if already done) ───────────────────────
ZIP_PATH    = "/content/drive/MyDrive/Thesis_Video_Colab.zip"
EXTRACT_DIR = "/content/thesis_video"
MARKER      = os.path.join(EXTRACT_DIR, ".extracted")

os.makedirs(EXTRACT_DIR, exist_ok=True)

if os.path.exists(MARKER):
    print("Dataset already extracted -- skipping unzip.")
else:
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Zip not found at {ZIP_PATH}\n"
            "Upload Thesis_Video_Colab.zip to the ROOT of your Google Drive (My Drive/).")
    print(f"Extracting {ZIP_PATH} to local SSD...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    open(MARKER, 'w').close()
    print(f"Extraction done in {(time.time()-t0)/60:.1f} min")

# Verify
import glob
videos = glob.glob(f"{EXTRACT_DIR}/Thesis_Video/dataset/**/*.mp4", recursive=True)
print(f"Videos found: {len(videos)}")
assert len(videos) > 0, "No .mp4 files found! Check the zip structure."

# ── 3. Copy training script ────────────────────────────────────────────────────
BASE         = f"{EXTRACT_DIR}/Thesis_Video"
DATASET_ROOT = f"{BASE}/dataset"
SPLITS_DIR   = f"{BASE}/splits"
SCRIPT_SRC   = f"{BASE}/colab_train.py"
SCRIPT_DST   = "/content/colab_train.py"
CKPT_DIR     = "/content/drive/MyDrive/thesis_video_checkpoints"

shutil.copy(SCRIPT_SRC, SCRIPT_DST)
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Script copied. Checkpoints -> {CKPT_DIR}")

# ── 4. GPU info ────────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Change runtime type to T4 GPU.")

# ── 5. Train ───────────────────────────────────────────────────────────────────
RESUME_CKPT = f"{CKPT_DIR}/last_checkpoint.pt"
resume_flag = ["--resume", RESUME_CKPT] if os.path.exists(RESUME_CKPT) else []

if resume_flag:
    print(f"Resuming from {RESUME_CKPT}")
else:
    print("Starting fresh training run...")

cmd = [
    sys.executable, "/content/colab_train.py",
    "--train_csv",       f"{SPLITS_DIR}/train.csv",
    "--val_csv",         f"{SPLITS_DIR}/val.csv",
    "--test_csv",        f"{SPLITS_DIR}/test.csv",
    "--dataset_root",    DATASET_ROOT,
    "--checkpoint_dir",  CKPT_DIR,
    "--backbone",        "swin_base_patch4_window7_224",
    "--num_frames",      "16",
    "--batch_size",      "16",
    "--epochs",          "40",
    "--freeze_stages",   "2",
    "--num_workers",     "4",
] + resume_flag

subprocess.run(cmd, check=True)
"""
#
# 4. Create a new notebook and paste each cell below (separated by ---CELL---)
#    into its own Colab code cell, then run them in order.
#
# ════════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Drive & verify GPU
# ───────────────────────────────────────────────────────────────────────────────
"""
from google.colab import drive
drive.mount('/content/drive')

import subprocess, torch

gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                      '--format=csv,noheader'], capture_output=True, text=True)
print("GPU:", gpu.stdout.strip())
print("PyTorch CUDA:", torch.cuda.is_available(),
      "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
"""

# ───────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install dependencies
# ───────────────────────────────────────────────────────────────────────────────
"""
%%capture
!pip install timm==1.0.26 opencv-python-headless scikit-learn
"""

# ───────────────────────────────────────────────────────────────────────────────
# CELL 3 — Unzip dataset to local SSD  (/content  — fast NVMe, not Drive)
# ───────────────────────────────────────────────────────────────────────────────
"""
import os, zipfile, time

ZIP_PATH    = "/content/drive/MyDrive/Thesis_Video_Colab.zip"
EXTRACT_DIR = "/content/thesis_video"           # local SSD — FAST reads

os.makedirs(EXTRACT_DIR, exist_ok=True)

# Check if already extracted (useful if cell is re-run)
marker = os.path.join(EXTRACT_DIR, ".extracted")
if os.path.exists(marker):
    print("Already extracted — skipping unzip.")
else:
    print("Extracting zip to local SSD (this takes ~5-15 min depending on dataset size)...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    elapsed = time.time() - t0
    open(marker, 'w').close()
    print(f"Done in {elapsed/60:.1f} min")

# Verify structure
import glob
videos = glob.glob(f"{EXTRACT_DIR}/Thesis_Video/dataset/**/*.mp4", recursive=True)
print(f"\\nTotal .mp4 files found: {len(videos)}")
print("Sample:", videos[0] if videos else "NONE — check path!")
"""

# ───────────────────────────────────────────────────────────────────────────────
# CELL 4 — Copy training script & set paths
# ───────────────────────────────────────────────────────────────────────────────
"""
import shutil, os

BASE         = "/content/thesis_video/Thesis_Video"
DATASET_ROOT = f"{BASE}/dataset"
SPLITS_DIR   = f"{BASE}/splits"
SCRIPT_SRC   = f"{BASE}/colab_train.py"
SCRIPT_DST   = "/content/colab_train.py"
CKPT_DIR     = "/content/drive/MyDrive/thesis_video_checkpoints"   # saved to Drive

shutil.copy(SCRIPT_SRC, SCRIPT_DST)
os.makedirs(CKPT_DIR, exist_ok=True)

print("Dataset root :", DATASET_ROOT)
print("Splits dir   :", SPLITS_DIR)
print("Checkpoint dir:", CKPT_DIR)

# Quick sanity check on splits
import csv
for split in ["train", "val"]:
    with open(f"{SPLITS_DIR}/{split}.csv") as f:
        n = sum(1 for _ in csv.DictReader(f))
    print(f"  {split.upper()}: {n} rows")
"""

# ───────────────────────────────────────────────────────────────────────────────
# CELL 5 — TRAIN  (T4-optimised settings)
# ───────────────────────────────────────────────────────────────────────────────
#
# T4 has 16 GB VRAM.  Tuning rationale:
#   --batch_size 16      : T4 can handle 16 frames × 224² × batch 16 with AMP
#   --num_frames 16      : Standard for temporal coverage
#   --backbone swin_base : Best accuracy vs cost; fits T4 in fp16
#   --freeze_stages 1    : Unfreeze more stages than local (more VRAM headroom)
#   --lr 8e-5            : Slightly higher peak LR — T4 trains faster per epoch
#   --epochs 50          : ~1.5h on T4 for this dataset size
#   --num_workers 4      : Local SSD (not Drive) → safe to use workers
#   --label_smoothing 0.1: Helps generalisation on small imbalanced set
#   --weight_decay 0.05  : AdamW regularisation
#   --warmup_epochs 5    : Stable warm-up
#
"""
!python /content/colab_train.py \
    --train_csv      /content/thesis_video/Thesis_Video/splits/train.csv \
    --val_csv        /content/thesis_video/Thesis_Video/splits/val.csv   \
    --test_csv       /content/thesis_video/Thesis_Video/splits/test.csv  \
    --dataset_root   /content/thesis_video/Thesis_Video/dataset          \
    --checkpoint_dir /content/drive/MyDrive/thesis_video_checkpoints     \
    --backbone       swin_base_patch4_window7_224                        \
    --num_frames     16                                                   \
    --batch_size     16                                                   \
    --epochs         40                                                   \
    --freeze_stages  2                                                    \
    --num_workers    4
"""

# ───────────────────────────────────────────────────────────────────────────────
# CELL 6 — (OPTIONAL) Resume from last checkpoint after session expires
# ───────────────────────────────────────────────────────────────────────────────
"""
# Re-run CELL 1 (mount) and CELL 3 (unzip — will skip if already done)
# then run this cell to resume:

!python /content/colab_train.py \
    --train_csv      /content/thesis_video/Thesis_Video/splits/train.csv \
    --val_csv        /content/thesis_video/Thesis_Video/splits/val.csv   \
    --test_csv       /content/thesis_video/Thesis_Video/splits/test.csv  \
    --dataset_root   /content/thesis_video/Thesis_Video/dataset          \
    --checkpoint_dir /content/drive/MyDrive/thesis_video_checkpoints     \
    --backbone       swin_base_patch4_window7_224                        \
    --num_frames     16                                                   \
    --batch_size     16                                                   \
    --epochs         40                                                   \
    --freeze_stages  2                                                    \
    --num_workers    4                                                    \
    --resume /content/drive/MyDrive/thesis_video_checkpoints/last_checkpoint.pt
"""

# ───────────────────────────────────────────────────────────────────────────────
# CELL 7 — Inspect training history (run any time after training starts)
# ───────────────────────────────────────────────────────────────────────────────
"""
import json, pandas as pd

history_path = "/content/drive/MyDrive/thesis_video_checkpoints/history.json"

with open(history_path) as f:
    history = json.load(f)

df = pd.DataFrame(history)
print(df[["epoch", "train_loss", "val_loss", "val_acc", "val_f1_macro"]].to_string(index=False))

best = df.loc[df["val_acc"].idxmax()]
print(f"\\n★ Best epoch: {int(best['epoch'])}  val_acc={best['val_acc']:.4f}  f1_macro={best['val_f1_macro']:.4f}")
"""
