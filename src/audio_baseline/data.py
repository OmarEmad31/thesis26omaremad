"""Audio Dataset — raw waveform loading only.

The emotion2vec backbone has its OWN internal feature extractor.
We must NOT pre-process with Wav2Vec2FeatureExtractor before passing to it.
We just load, resample, normalize, and return the raw float32 waveform.
"""

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
from pathlib import Path


def collate_audio_fn(batch):
    """
    Synchronized Collate: drops None samples and pads the rest.
    Returns a dict with input_values [B, T], attention_mask [B, T], labels [B].
    """
    valid_batch = [b for b in batch if b is not None]

    if len(valid_batch) == 0:
        return None

    input_values = [b["input_values"] for b in valid_batch]
    labels = [b["labels"] for b in valid_batch]

    # Pad to the longest waveform in the batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )

    # Attention mask: 1 for real signal, 0 for padding
    attention_mask = (inputs_padded != 0).long()

    return {
        "input_values": inputs_padded,
        "attention_mask": attention_mask,
        "labels": torch.stack(labels),
    }


class AudioEmotionDataset(Dataset):
    def __init__(
        self,
        csv_path,          # Path or None (set .df manually after init)
        data_root: Path,
        label2id: dict,
        max_samples: int = 160000,
        sampling_rate: int = 16000,
        # feature_extractor param kept for backward compat but IGNORED
        feature_extractor=None,
    ):
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
        self.data_root = data_root
        self.label2id = label2id
        self.max_samples = max_samples
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, folder_name: str, rel_path: str) -> Path:
        """Try the direct path, then scan one level of subfolders as fallback."""
        candidate = self.data_root / folder_name / rel_path
        if candidate.exists():
            return candidate
        # Fallback: maybe the zip added an extra wrapper folder
        try:
            for sub in self.data_root.iterdir():
                if sub.is_dir():
                    alt = sub / folder_name / rel_path
                    if alt.exists():
                        return alt
        except Exception:
            pass
        return candidate  # return original (will fail exists() check, triggers None)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Label
        label_str = row["emotion_final"]
        label = self.label2id[label_str]

        # Path
        folder_name = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        audio_path = self._resolve_path(folder_name, rel_path)

        # Load
        try:
            if not audio_path.exists():
                return None

            # Load and resample to 16 kHz
            speech, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)

            # Truncate to max length
            if len(speech) > self.max_samples:
                speech = speech[: self.max_samples]

            # Normalize to zero mean, unit variance
            # This is what emotion2vec expects as its raw input
            if speech.std() > 1e-6:
                speech = (speech - speech.mean()) / speech.std()
            else:
                speech = speech - speech.mean()

            waveform = torch.tensor(speech, dtype=torch.float32)

            return {
                "input_values": waveform,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        except Exception:
            return None
