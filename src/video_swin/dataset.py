"""
Video Dataset for Emotion Recognition
Uses OpenCV for frame extraction with robust fallback strategies.
"""

import os
import csv
import random
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

# ── Label mapping ──────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happiness": 3,
    "Neutral": 4,
    "Sadness": 5,
    "Surprise": 6,
}
ID_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}
NUM_CLASSES = len(EMOTION_LABELS)

# ── Path resolution ────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"D:\Thesis Project\dataset\Final Modalink Dataset MERGED")


def resolve_video_path(row: dict) -> Path | None:
    """
    Try multiple strategies to find the actual video file on disk.
    The manifests may reference old Desktop paths; we re-anchor to DATASET_ROOT.
    """
    folder = row.get("folder", "")
    relpath = row.get("video_relpath", "")

    # Strategy 1: re-anchor using folder + relative path
    candidate = DATASET_ROOT / folder / relpath
    if candidate.exists():
        return candidate

    # Strategy 2: absolute path stored in video_path column (may have old prefix)
    raw = row.get("video_path", "")
    if raw:
        p = Path(raw)
        # Try as-is first
        if p.exists():
            return p
        # Re-anchor: strip everything up to "Final Modalink Dataset MERGED"
        parts = p.parts
        try:
            idx = next(
                i
                for i, part in enumerate(parts)
                if "Final Modalink Dataset MERGED" in part
            )
            tail = Path(*parts[idx:])
            candidate2 = Path("D:/") / tail
            if candidate2.exists():
                return candidate2
            candidate3 = DATASET_ROOT.parent / Path(*parts[idx + 1:])
            if candidate3.exists():
                return candidate3
        except StopIteration:
            pass

    return None


def load_csv_split(csv_path: str) -> list[dict]:
    """Load a split CSV and filter to video-eligible, existent files."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Must have a valid emotion label
            emotion = row.get("emotion_final", "").strip()
            if emotion not in EMOTION_LABELS:
                continue
            # Must be video eligible
            if row.get("elig_video", "0").strip() != "1":
                continue
            path = resolve_video_path(row)
            if path is None:
                logger.debug("File not found: %s", row.get("video_path", ""))
                continue
            row["_resolved_path"] = str(path)
            row["_label"] = EMOTION_LABELS[emotion]
            rows.append(row)
    return rows


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 16) -> np.ndarray | None:
    """
    Extract `num_frames` frames uniformly from the video.
    Returns (T, H, W, 3) uint8 array or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Uniformly sample frame indices
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)  # repeat frames

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Try sequential read as fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx) - 1))
            ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # Pad if we got fewer frames than expected
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames[:num_frames])  # (T, H, W, 3)


# ── Transforms ────────────────────────────────────────────────────────────────

# ImageNet mean/std – used by Swin pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_SIZE = 224
CROP_SIZE = 224


class VideoAugment:
    """Consistent spatial augmentation applied to all frames of a clip."""

    def __init__(self, train: bool = True, size: int = CROP_SIZE):
        self.train = train
        self.size = size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: (T, H, W, 3) uint8 numpy
        Returns:
            tensor: (3, T, H, W) float32 normalised
        """
        T_len, H, W, C = frames.shape

        if self.train:
            # Random resized crop (consistent across frames)
            i, j, h, w = T.RandomResizedCrop.get_params(
                torch.zeros(H, W),
                scale=(0.6, 1.0),
                ratio=(0.75, 1.333),
            )
            # Random horizontal flip
            do_flip = random.random() < 0.5
            # Random colour jitter params
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            # Mild rotation
            angle = random.uniform(-10, 10)
        else:
            i, j, h, w = None, None, None, None
            do_flip = False

        processed = []
        for frame in frames:
            img = TF.to_tensor(frame)  # (3, H, W), float [0,1]

            if self.train:
                img = TF.resized_crop(img, i, j, h, w, [self.size, self.size])
                if do_flip:
                    img = TF.hflip(img)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.rotate(img, angle)
            else:
                # Centre crop
                img = TF.resize(img, [self.size + 32, self.size + 32])
                img = TF.center_crop(img, [self.size, self.size])

            img = self.normalize(img)
            processed.append(img)

        # Stack: (T, 3, H, W) → (3, T, H, W)
        stacked = torch.stack(processed, dim=0)  # (T, 3, H, W)
        stacked = stacked.permute(1, 0, 2, 3)     # (3, T, H, W)
        return stacked


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoEmotionDataset(Dataset):
    """
    PyTorch Dataset for video-based emotion recognition.
    Loads short clips, extracts frames, applies augmentation.
    """

    def __init__(
        self,
        csv_path: str,
        num_frames: int = 16,
        train: bool = True,
    ):
        self.num_frames = num_frames
        self.transform = VideoAugment(train=train)
        self.samples = load_csv_split(csv_path)
        logger.info(
            "Loaded %d samples from %s (train=%s)",
            len(self.samples), csv_path, train,
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {csv_path}")

        # Compute class weights for balanced sampling
        counts = [0] * NUM_CLASSES
        for row in self.samples:
            counts[row["_label"]] += 1
        self.class_counts = counts
        logger.info("Class distribution: %s", {ID_TO_EMOTION[i]: c for i, c in enumerate(counts)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        path = row["_resolved_path"]
        label = row["_label"]

        frames = extract_frames(path, self.num_frames)

        if frames is None:
            # Fall back to random valid sample
            logger.warning("Could not decode %s, using random fallback", path)
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)

        video_tensor = self.transform(frames)  # (3, T, H, W)
        return video_tensor, label
