"""Audio data utilities.

Two datasets:
1. AudioEmotionDataset  — used during embedding extraction (loads raw audio)
2. EmbeddingDataset     — used during MLP training (loads pre-computed tensors)
"""

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------
def resolve_audio_path(data_root: Path, folder: str, rel_path: str) -> Path:
    """Try direct path, then scan one subfolder level as fallback."""
    candidate = data_root / folder / rel_path
    if candidate.exists():
        return candidate
    try:
        for sub in data_root.iterdir():
            if sub.is_dir():
                alt = sub / folder / rel_path
                if alt.exists():
                    return alt
    except Exception:
        pass
    return candidate  # will fail .exists(), triggers skip


# ---------------------------------------------------------------------------
# 1. Raw-audio dataset (embedding extraction only)
# ---------------------------------------------------------------------------
class AudioEmotionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        label2id: dict,
        max_samples: int = 160000,
        sampling_rate: int = 16000,
        feature_extractor=None,   # kept for backward-compat, NOT used
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.label2id = label2id
        self.max_samples = max_samples
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label2id[row["emotion_final"]]

        folder   = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        path     = resolve_audio_path(self.data_root, folder, rel_path)

        try:
            if not path.exists():
                return None
            audio, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
            if len(audio) > self.max_samples:
                audio = audio[: self.max_samples]
            return {
                "audio": audio.astype(np.float32),   # raw numpy waveform
                "label": label,
                "path": str(path),
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 2. Embedding dataset  (MLP training)
# ---------------------------------------------------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels     = torch.tensor(labels,     dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Collate for AudioEmotionDataset (used only during extraction)
# ---------------------------------------------------------------------------
def collate_audio_fn(batch):
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    return valid   # list of dicts — processed one-by-one in extraction loop
