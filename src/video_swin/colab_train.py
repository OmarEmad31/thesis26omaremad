"""
=======================================================================
  VIDEO SWIN TRANSFORMER — EMOTION RECOGNITION  (Google Colab Script)
=======================================================================

INSTRUCTIONS
------------
1. Upload this file and your video splits to your Google Drive:
     /MyDrive/thesis/
       ├── colab_train.py          ← this file
       ├── splits/
       │    ├── train.csv
       │    └── val.csv
       └── dataset/               ← your video files (mirroring local structure)

2. In Colab: Runtime → Change runtime type → GPU (T4 recommended)

3. Run each cell in order.

CELL 1 — Mount Drive + Install deps
------------------------------------
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install -q timm scikit-learn

CELL 2 — Run training
----------------------
  !python /content/drive/MyDrive/thesis/colab_train.py \
      --train_csv  /content/drive/MyDrive/thesis/splits/train.csv \
      --val_csv    /content/drive/MyDrive/thesis/splits/val.csv \
      --dataset_root /content/drive/MyDrive/thesis/dataset \
      --checkpoint_dir /content/drive/MyDrive/thesis/checkpoints \
      --epochs 40 \
      --batch_size 8 \
      --num_frames 16 \
      --lr 5e-5 \
      --freeze_stages 2
=======================================================================
"""

import os, sys, csv, json, time, logging, random, argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Globals (overridden by CLI) ───────────────────────────────────────────────
DATASET_ROOT: Path = None   # set from --dataset_root arg

EMOTION_LABELS = {
    "Anger": 0, "Disgust": 1, "Fear": 2, "Happiness": 3,
    "Neutral": 4, "Sadness": 5, "Surprise": 6,
}
ID_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}
NUM_CLASSES = len(EMOTION_LABELS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_video_path(row: dict) -> Path | None:
    """Find the video file under DATASET_ROOT using folder + relpath."""
    # Strategy 1: direct join (works when folder/relpath columns are populated)
    folder  = row.get("folder", "").strip()
    relpath = row.get("video_relpath", "").strip()
    if folder and relpath:
        candidate = DATASET_ROOT / folder / relpath
        if candidate.exists():
            return candidate

    # Strategy 2: parse the raw video_path column.
    # On Colab (Linux), Windows paths like C:\Users\...\file.mp4 are a single
    # string — we must split on backslash manually, not use Path.parts.
    raw = row.get("video_path", "").strip()
    if raw:
        # Normalise: replace backslashes with forward slashes then split
        parts = raw.replace("\\", "/").split("/")
        try:
            idx = next(i for i, pt in enumerate(parts)
                       if "Final Modalink Dataset MERGED" in pt)
            # Everything AFTER "Final Modalink Dataset MERGED"
            tail_parts = parts[idx + 1:]
            if tail_parts:
                tail = Path(*tail_parts)
                candidate = DATASET_ROOT / tail
                if candidate.exists():
                    return candidate
        except StopIteration:
            pass
    return None


def check_dataset_root():
    """Print diagnostics to help find the correct --dataset_root."""
    print(f"\n[DIAGNOSTIC] DATASET_ROOT = {DATASET_ROOT}")
    print(f"[DIAGNOSTIC] DATASET_ROOT exists = {DATASET_ROOT.exists()}")
    if not DATASET_ROOT.exists():
        parent = DATASET_ROOT.parent
        print(f"[DIAGNOSTIC] Parent '{parent}' exists = {parent.exists()}")
        if parent.exists():
            children = [p.name for p in sorted(parent.iterdir())[:20]]
            print(f"[DIAGNOSTIC] Contents of parent:\n  " + "\n  ".join(children))
    else:
        children = [p.name for p in sorted(DATASET_ROOT.iterdir())[:5]]
        print(f"[DIAGNOSTIC] First 5 subfolders: {children}")


def load_csv_split(csv_path: str) -> list[dict]:
    rows, failed = [], 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            emotion = row.get("emotion_final", "").strip()
            if emotion not in EMOTION_LABELS:
                continue
            if row.get("elig_video", "0").strip() != "1":
                continue
            path = resolve_video_path(row)
            if path is None:
                failed += 1
                continue
            row["_resolved_path"] = str(path)
            row["_label"] = EMOTION_LABELS[emotion]
            rows.append(row)
    if failed > 0:
        logger.warning("%d rows could not be resolved — check --dataset_root", failed)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save_frames(video_path: str, out_dir: Path, num_frames: int) -> bool:
    """Decode num_frames from video and save as JPEGs. Returns True on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"frame_{num_frames-1:04d}.jpg").exists():
        return True  # already done
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
    cap.release()
    # Pad missing frames by copying last
    if 0 < saved < num_frames:
        last = out_dir / f"frame_{saved-1:04d}.jpg"
        for i in range(saved, num_frames):
            import shutil
            shutil.copy(str(last), str(out_dir / f"frame_{i:04d}.jpg"))
    return saved > 0


def preextract_frames(csv_paths: list, frames_root: str, num_frames: int):
    """Phase 1: decode all videos to JPEG frames on local SSD (one-time)."""
    frames_root = Path(frames_root)
    frames_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    seen_sids = set()
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("elig_video", "0").strip() != "1":
                    continue
                if row.get("emotion_final", "").strip() not in EMOTION_LABELS:
                    continue
                path = resolve_video_path(row)
                if path is None:
                    continue
                sid = (row.get("sample_id", "").strip()
                       .replace("/", "_").replace("\\", "_")
                       .replace(" ", "_").replace(":", "_"))
                if sid in seen_sids:
                    continue
                seen_sids.add(sid)
                all_rows.append({"path": path, "sid": sid})

    logger.info("Phase 1: extracting frames for %d videos to %s...", len(all_rows), frames_root)
    done, skipped = 0, 0
    for r in tqdm(all_rows, desc="Extracting frames"):
        out_dir = frames_root / r["sid"]
        if extract_and_save_frames(r["path"], out_dir, num_frames):
            if (out_dir / f"frame_{num_frames-1:04d}.jpg").exists():
                skipped += 1 if done == 0 else 0
            done += 1
        else:
            logger.warning("Failed to extract: %s", r["path"])
    logger.info("Phase 1 done: %d videos -> %s", done, frames_root)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class VideoAugment:
    def __init__(self, train: bool = True, size: int = 224):
        self.train = train
        self.size  = size
        self.norm  = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        T_len, H, W, _ = frames.shape
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(
                torch.zeros(H, W), scale=(0.6, 1.0), ratio=(0.75, 1.333))
            do_flip = random.random() < 0.5
            brightness  = random.uniform(0.8, 1.2)
            contrast    = random.uniform(0.8, 1.2)
            saturation  = random.uniform(0.8, 1.2)
            angle       = random.uniform(-10, 10)
        processed = []
        for frame in frames:
            img = TF.to_tensor(frame)
            if self.train:
                img = TF.resized_crop(img, i, j, h, w, [self.size, self.size])
                if do_flip:        img = TF.hflip(img)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.rotate(img, angle)
            else:
                img = TF.resize(img, [self.size + 32, self.size + 32])
                img = TF.center_crop(img, [self.size, self.size])
            processed.append(self.norm(img))
        stacked = torch.stack(processed, dim=0)   # (T, 3, H, W)
        return stacked.permute(1, 0, 2, 3)        # (3, T, H, W)


class FrameDataset(Dataset):
    """Loads pre-extracted JPEG frames from SSD — fast end-to-end training."""
    def __init__(self, csv_path: str, frames_root: str,
                 num_frames: int = 16, train: bool = True):
        self.frames_root = Path(frames_root)
        self.num_frames  = num_frames
        self.transform   = VideoAugment(train=train)
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                emotion = row.get("emotion_final", "").strip()
                if emotion not in EMOTION_LABELS:
                    continue
                if row.get("elig_video", "0").strip() != "1":
                    continue
                sid = (row.get("sample_id", "").strip()
                       .replace("/", "_").replace("\\", "_")
                       .replace(" ", "_").replace(":", "_"))
                frame_dir = self.frames_root / sid
                if frame_dir.exists():
                    rows.append({"frame_dir": frame_dir,
                                 "_label": EMOTION_LABELS[emotion]})
        self.samples = rows
        counts = [0] * NUM_CLASSES
        for r in rows:
            counts[r["_label"]] += 1
        self.class_counts = counts
        logger.info("FrameDataset: %d samples (train=%s)", len(rows), train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        frames = []
        for i in range(self.num_frames):
            img = cv2.imread(str(row["frame_dir"] / f"frame_{i:04d}.jpg"))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not frames:
            return self.__getitem__((idx + 1) % len(self.samples))
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        clip = self.transform(np.stack(frames[:self.num_frames]))
        return clip, row["_label"]





# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CACHE DATASET  (fast path — loads pre-extracted .pt files)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """Loads pre-extracted (T, D) feature tensors instead of raw video."""
    def __init__(self, csv_path: str, cache_dir: str, augment: bool = False):
        self.cache_dir = Path(cache_dir)
        self.augment   = augment
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                emotion = row.get("emotion_final", "").strip()
                if emotion not in EMOTION_LABELS:
                    continue
                if row.get("elig_video", "0").strip() != "1":
                    continue
                sid = (row.get("sample_id", "").strip()
                       .replace("/", "_").replace("\\", "_")
                       .replace(" ", "_").replace(":", "_"))
                feat_path = self.cache_dir / f"{sid}.pt"
                if feat_path.exists():
                    rows.append({"feat_path": feat_path,
                                 "_label": EMOTION_LABELS[emotion]})
        self.samples = rows
        counts = [0] * NUM_CLASSES
        for r in rows:
            counts[r["_label"]] += 1
        self.class_counts = counts
        logger.info("FeatureDataset: %d cached samples from %s", len(rows), csv_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row  = self.samples[idx]
        feat = torch.load(row["feat_path"], map_location="cpu", weights_only=True)
        if self.augment and random.random() < 0.5:
            feat = feat[torch.randperm(feat.size(0))]  # temporal shuffle augment
        return feat, row["_label"]


def preextract_features(backbone, csv_paths: list, cache_dir: str,
                        num_frames: int, device: torch.device):
    """Run all videos through frozen Swin once, save (T, D) tensors to disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone.eval()
    transform = VideoAugment(train=False)

    all_rows = []
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("elig_video", "0").strip() != "1":
                    continue
                if row.get("emotion_final", "").strip() not in EMOTION_LABELS:
                    continue
                path = resolve_video_path(row)
                if path is None:
                    continue
                sid = (row.get("sample_id", "").strip()
                       .replace("/", "_").replace("\\", "_")
                       .replace(" ", "_").replace(":", "_"))
                all_rows.append({"path": path, "sid": sid})

    logger.info("Pre-extracting features for %d videos...", len(all_rows))
    done, skipped = 0, 0
    with torch.no_grad():
        for r in tqdm(all_rows, desc="Extracting features"):
            feat_path = cache_dir / f"{r['sid']}.pt"
            if feat_path.exists():
                skipped += 1
                continue
            frames = extract_frames(str(r["path"]), num_frames)
            if frames is None:
                continue
            clip = transform(frames).unsqueeze(0).to(device)  # (1, 3, T, H, W)
            _, C, T, H, W = clip.shape
            clip = clip.squeeze(0).permute(1, 0, 2, 3)        # (T, C, H, W)
            with autocast("cuda"):
                feat = backbone(clip)                          # (T, D)
            torch.save(feat.cpu(), feat_path)
            done += 1

    logger.info("Done: %d new features, %d skipped.", done, skipped)



# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Lightweight self-attention over T frame tokens before classification."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):          # x: (B, T, D)
        out, _ = self.attn(x, x, x)
        return self.norm(x + out).mean(dim=1)   # (B, D)


class SwinVideoModel(nn.Module):
    def __init__(self, backbone="swin_base_patch4_window7_224",
                 pretrained=True, num_classes=NUM_CLASSES,
                 dropout=0.4, freeze_stages=2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        # Freeze patch embed + first N stages
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "absolute_pos_embed"):
            self.backbone.absolute_pos_embed.requires_grad = False
        if hasattr(self.backbone, "layers"):
            for i, layer in enumerate(self.backbone.layers):
                if i < freeze_stages:
                    for p in layer.parameters():
                        p.requires_grad = False

        self.temporal_attn = TemporalAttention(feat_dim, num_heads=4, dropout=0.5)

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            # Pre-extracted features: (B, T, D) — skip backbone
            feat = self.temporal_attn(x)     # (B, D)
        else:
            # Raw video: (B, 3, T, H, W)
            B, C, T, H, W = x.shape
            x    = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            feat = self.backbone(x).reshape(B, T, -1)
            feat = self.temporal_attn(feat)  # (B, D)
        return self.head(feat), feat



# ─────────────────────────────────────────────────────────────────────────────
# LOSS ENGINES (SCL + FOCAL)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def scl_loss(hidden, labels, temp=0.1):
    features = F.normalize(hidden, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(hidden.device)
    mask *= (1 - torch.eye(labels.size(0), device=hidden.device))
    valid = mask.sum(1) > 0
    if not valid.any(): return torch.tensor(0.0).to(hidden.device)
    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True) + 1e-8)
    loss = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_sampler(dataset):
    total = sum(dataset.class_counts)
    cw    = [total / (c + 1e-6) for c in dataset.class_counts]
    sw    = [cw[row["_label"]] for row in dataset.samples]
    return WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)


def make_criterion(dataset, device, label_smoothing=0.15):
    counts  = np.array(dataset.class_counts, dtype=np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * NUM_CLASSES
    logger.info("Class weights: %s",
                {ID_TO_EMOTION[i]: f"{w:.3f}" for i, w in enumerate(weights)})
    return FocalLoss(
        weight=torch.tensor(weights).to(device),
        gamma=2.0,
        label_smoothing=label_smoothing,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip, grad_accum=1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for step, (videos, labels) in enumerate(bar):
        videos, labels = videos.to(device), labels.to(device)
        with autocast("cuda"):
            logits, pooled = model(videos)
            loss   = (criterion(logits, labels) + 0.1 * scl_loss(pooled, labels)) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * grad_accum * labels.size(0)
        correct    += (logits.detach().argmax(1) == labels).sum().item()
        total      += labels.size(0)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labs, total_loss = [], [], 0.0
    bar = tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True)
    for videos, labels in bar:
        videos, labels = videos.to(device), labels.to(device)
        with autocast("cuda"):
            logits, _ = model(videos)
            total_loss += criterion(logits, labels).item() * labels.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labs.extend(labels.cpu().numpy())
    return {
        "loss":        total_loss / len(loader.dataset),
        "accuracy":    accuracy_score(labs, preds),
        "f1_macro":    f1_score(labs, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labs, preds, average="weighted", zero_division=0),
        "preds": preds, "labels": labs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",       required=True)
    p.add_argument("--val_csv",         required=True)
    p.add_argument("--test_csv",        default=None)
    p.add_argument("--dataset_root",    required=True,
                   help="Root folder containing 'videoplayback (N)' subdirs")
    p.add_argument("--checkpoint_dir",  default="checkpoints/video_swin")
    p.add_argument("--backbone",        default="swin_base_patch4_window7_224")
    p.add_argument("--num_frames",      type=int,   default=16)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--epochs",          type=int,   default=40)
    p.add_argument("--lr",              type=float, default=5e-4, help="Learning rate for head")
    p.add_argument("--backbone_lr",     type=float, default=5e-5, help="Learning rate for backbone")
    p.add_argument("--min_lr",          type=float, default=1e-7)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--weight_decay",    type=float, default=0.05)
    p.add_argument("--dropout",         type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.15)
    p.add_argument("--freeze_stages",   type=int,   default=3)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--grad_accum",      type=int,   default=2,
                   help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    p.add_argument("--patience",        type=int,   default=10,
                   help="Early stopping patience (epochs without val_acc improvement)")
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--resume",          default=None)
    p.add_argument("--cache_dir",       default="/content/feat_cache",
                   help="Where to store pre-extracted backbone features")
    return p.parse_args()


def main():
    args = parse_args()

    # Set global dataset root
    global DATASET_ROOT
    DATASET_ROOT = Path(args.dataset_root)

    # Reproducibility
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: extract frames to JPEG on local SSD (one-time, skips if done) ──
    preextract_frames(
        csv_paths   = [args.train_csv, args.val_csv] + ([args.test_csv] if args.test_csv else []),
        frames_root = args.cache_dir,
        num_frames  = args.num_frames,
    )

    # ── Phase 2: end-to-end fine-tuning loading from JPEG (fast I/O) ──
    train_ds = FrameDataset(args.train_csv, args.cache_dir, args.num_frames, train=True)
    val_ds   = FrameDataset(args.val_csv,   args.cache_dir, args.num_frames, train=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No frame directories found — Phase 1 may have failed.")

    # ── Model ──
    logger.info("Building model: %s", args.backbone)
    model = SwinVideoModel(args.backbone, pretrained=True,
                           dropout=args.dropout,
                           freeze_stages=args.freeze_stages).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %dM total | %dM trainable", total//1_000_000, trainable//1_000_000)

    # ── Loss / Optim — discriminative LRs ──
    criterion = make_criterion(train_ds, device, args.label_smoothing)

    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params,     "lr": args.lr,          "weight_decay": args.weight_decay},
    ])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=make_sampler(train_ds),
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.min_lr),
    ], milestones=[warmup_steps])

    scaler = GradScaler("cuda")



    # ── Resume ──
    start_epoch, best_acc, best_f1, history = 1, 0.0, 0.0, []
    if args.resume and Path(args.resume).exists():
        try:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc    = ckpt.get("best_val_acc", 0.0)
            history     = ckpt.get("history", [])
            logger.info("Resumed from epoch %d (best_acc=%.4f)", start_epoch-1, best_acc)
        except Exception as e:
            logger.warning("Failed to resume from %s: %s. Starting fresh.", args.resume, e)
            start_epoch, best_acc, best_f1, history = 1, 0.0, 0.0, []

    logger.info("=" * 60)
    logger.info("Training %d epochs  |  train=%d  val=%d  effective_batch=%d",
                args.epochs, len(train_ds), len(val_ds),
                args.batch_size * args.grad_accum)
    logger.info("=" * 60)

    no_improve = 0  # early stopping counter

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n── Epoch {epoch}/{args.epochs} ──")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            args.grad_clip, args.grad_accum)
        scheduler.step()

        val = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.3f} "
              f"| val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.3f} "
              f"val_f1_macro={val['f1_macro']:.3f}  val_f1_w={val['f1_weighted']:.3f} "
              f"| {elapsed:.0f}s")

        if val["accuracy"] > best_acc:
            best_acc   = val["accuracy"]
            no_improve = 0
            best_f1  = val["f1_macro"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("★ New best  acc=%.4f  f1_macro=%.4f  → saved!", best_acc, best_f1)
            label_names = [ID_TO_EMOTION[i] for i in range(NUM_CLASSES)]
            print(classification_report(val["labels"], val["preds"],
                                        target_names=label_names, zero_division=0))

        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping triggered (no improvement for {args.patience} epochs).")
                break

        # Full checkpoint (for resume)
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_acc, "best_val_f1": best_f1, "history": history,
        }, ckpt_dir / "last_checkpoint.pt")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val["loss"], "val_acc": val["accuracy"],
            "val_f1_macro": val["f1_macro"], "val_f1_weighted": val["f1_weighted"],
        })
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best val_acc={best_acc:.4f} ({best_acc*100:.1f}%)  f1_macro={best_f1:.4f}")
    print(f"Best model -> {ckpt_dir / 'best_model.pt'}")

    # ── Final test evaluation on best checkpoint ──
    if args.test_csv:
        print("\n" + "=" * 60)
        print("FINAL TEST SET EVALUATION (best_model.pt)")
        print("=" * 60)
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
        test_ds     = FrameDataset(args.test_csv, args.cache_dir, args.num_frames, train=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test = evaluate(model, test_loader, criterion, device)
        label_names = [ID_TO_EMOTION[i] for i in range(NUM_CLASSES)]
        print(f"  test_acc     = {test['accuracy']:.4f} ({test['accuracy']*100:.1f}%)")
        print(f"  f1_macro     = {test['f1_macro']:.4f}")
        print(f"  f1_weighted  = {test['f1_weighted']:.4f}")
        print()
        print(classification_report(test["labels"], test["preds"],
                                    target_names=label_names, zero_division=0))
        print("=" * 60)


if __name__ == "__main__":
    main()
