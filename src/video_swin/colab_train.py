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
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
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
    folder  = row.get("folder", "")
    relpath = row.get("video_relpath", "")
    candidate = DATASET_ROOT / folder / relpath
    if candidate.exists():
        return candidate
    # Fallback: try stripping old absolute prefix from video_path column
    raw = row.get("video_path", "")
    if raw:
        p = Path(raw)
        parts = p.parts
        try:
            idx = next(i for i, pt in enumerate(parts)
                       if "Final Modalink Dataset MERGED" in pt)
            tail = Path(*parts[idx + 1:])      # everything after the root dir
            c2 = DATASET_ROOT / tail
            if c2.exists():
                return c2
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

def extract_frames(video_path: str, num_frames: int = 16) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames])   # (T, H, W, 3)


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


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class VideoEmotionDataset(Dataset):
    def __init__(self, csv_path: str, num_frames: int = 16, train: bool = True):
        self.num_frames = num_frames
        self.transform  = VideoAugment(train=train)
        self.samples    = load_csv_split(csv_path)
        if not self.samples:
            check_dataset_root()
            raise RuntimeError(
                f"No valid samples found in {csv_path}.\n"
                f"DATASET_ROOT={DATASET_ROOT} — does it contain 'videoplayback (N)' subfolders?\n"
                f"Pass the correct path via --dataset_root"
            )
        counts = [0] * NUM_CLASSES
        for row in self.samples:
            counts[row["_label"]] += 1
        self.class_counts = counts
        logger.info("Loaded %d samples | dist: %s", len(self.samples),
                    {ID_TO_EMOTION[i]: c for i, c in enumerate(counts)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Iterative fallback — avoids RecursionError on bad/slow Drive files
        for attempt in range(len(self.samples)):
            current_idx = (idx + attempt) % len(self.samples)
            row = self.samples[current_idx]
            frames = extract_frames(row["_resolved_path"], self.num_frames)
            if frames is not None:
                return self.transform(frames), row["_label"]
        raise RuntimeError("Could not load any video from the dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

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

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):                     # x: (B, 3, T, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feat = self.backbone(x).reshape(B, T, -1).mean(dim=1)
        return self.head(feat)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_sampler(dataset):
    total = sum(dataset.class_counts)
    cw    = [total / (c + 1e-6) for c in dataset.class_counts]
    sw    = [cw[row["_label"]] for row in dataset.samples]
    return WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)


def make_criterion(dataset, device, label_smoothing=0.1):
    counts  = np.array(dataset.class_counts, dtype=np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * NUM_CLASSES
    logger.info("Class weights: %s",
                {ID_TO_EMOTION[i]: f"{w:.3f}" for i, w in enumerate(weights)})
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights).to(device),
        label_smoothing=label_smoothing,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (videos, labels) in enumerate(loader):
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            loss = criterion(model(videos), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct    += (model(videos).detach().argmax(1) == labels).sum().item() if False else 0
        total      += labels.size(0)
        if (batch_idx + 1) % 15 == 0:
            logger.info("  [%d/%d] loss=%.4f", batch_idx + 1, len(loader),
                        total_loss / total)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labs, total_loss = [], [], 0.0
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        with autocast("cuda"):
            logits = model(videos)
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
    p.add_argument("--train_csv",      required=True)
    p.add_argument("--val_csv",        required=True)
    p.add_argument("--dataset_root",   required=True,
                   help="Root folder containing 'videoplayback (N)' subdirs")
    p.add_argument("--checkpoint_dir", default="checkpoints/video_swin")
    p.add_argument("--backbone",       default="swin_base_patch4_window7_224")
    p.add_argument("--num_frames",     type=int,   default=16)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--min_lr",         type=float, default=1e-7)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--dropout",        type=float, default=0.4)
    p.add_argument("--label_smoothing",type=float, default=0.1)
    p.add_argument("--freeze_stages",  type=int,   default=2)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--resume",         default=None)
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

    # ── Data ──
    train_ds = VideoEmotionDataset(args.train_csv, args.num_frames, train=True)
    val_ds   = VideoEmotionDataset(args.val_csv,   args.num_frames, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=make_sampler(train_ds),
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model ──
    model = SwinVideoModel(args.backbone, pretrained=True,
                           dropout=args.dropout,
                           freeze_stages=args.freeze_stages).to(device)
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %dM total | %dM trainable", total//1_000_000, trainable//1_000_000)

    # ── Loss / Optim ──
    criterion = make_criterion(train_ds, device, args.label_smoothing)

    backbone_params = [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params     = [p for n,p in model.named_parameters() if "head"     in n and p.requires_grad]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay},
        {"params": head_params,     "lr": args.lr,       "weight_decay": args.weight_decay * 0.1},
    ])

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
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_val_acc", 0.0)
        history     = ckpt.get("history", [])
        logger.info("Resumed from epoch %d (best_acc=%.4f)", start_epoch-1, best_acc)

    # ── Training Loop ──
    logger.info("=" * 60)
    logger.info("Training %d epochs  |  train=%d  val=%d",
                args.epochs, len(train_ds), len(val_ds))
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        logger.info("\n── Epoch %d/%d ──", epoch, args.epochs)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_clip)
        scheduler.step()

        val = evaluate(model, val_loader, criterion, device)
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f val_acc=%.4f "
            "val_f1_macro=%.4f val_f1_w=%.4f | %.0fs",
            epoch, train_loss,
            val["loss"], val["accuracy"], val["f1_macro"], val["f1_weighted"],
            time.time() - t0,
        )

        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_f1  = val["f1_macro"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("★ New best  acc=%.4f  f1_macro=%.4f  → saved!", best_acc, best_f1)
            label_names = [ID_TO_EMOTION[i] for i in range(NUM_CLASSES)]
            print(classification_report(val["labels"], val["preds"],
                                        target_names=label_names, zero_division=0))

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

    logger.info("\n" + "=" * 60)
    logger.info("Done!  Best val_acc=%.4f (%.1f%%)  f1_macro=%.4f",
                best_acc, best_acc * 100, best_f1)
    logger.info("Best model → %s", ckpt_dir / "best_model.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
