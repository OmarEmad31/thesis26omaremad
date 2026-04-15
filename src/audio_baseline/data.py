"""Audio Dataset and Preprocessing (librosa + transformers)."""

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

def collate_audio_fn(batch):
    """
    Synchronized Collate: Ensures labels and inputs are matched perfectly.
    If a sample failed to load (is None), it is dropped from both lists.
    """
    # Filter out None samples (failed loads) to keep batch size in sync
    valid_batch = [b for b in batch if b is not None]
    
    if len(valid_batch) == 0:
        return None
        
    input_values = [b["input_values"] for b in valid_batch]
    labels = [b["labels"] for b in valid_batch]
    
    # Pad waveforms to the longest in the batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )
    
    # Create attention mask
    attention_mask = (inputs_padded != 0).long()
    
    return {
        "input_values": inputs_padded,
        "attention_mask": attention_mask,
        "labels": torch.stack(labels)
    }

class AudioEmotionDataset(Dataset):
    def __init__(
        self, 
        csv_path: Path | None, 
        data_root: Path, 
        feature_extractor: Wav2Vec2FeatureExtractor, 
        label2id: dict[str, int],
        max_samples: int = 160000,
        sampling_rate: int = 16000
    ):
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
        self.data_root = data_root
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.max_samples = max_samples
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Map labels
        label_str = row["emotion_final"]
        label = self.label2id[label_str]
        
        # 2. Optimized Path Reconstruction
        folder_name = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        
        # The true path on SSD
        audio_path = self.data_root / folder_name / rel_path
        
        # 3. 🛡️ SMART PATH FALLBACK:
        # Avoids skipping files if unzipping created a subfolder
        if not audio_path.exists():
            try:
                subfolders = [d for d in self.data_root.iterdir() if d.is_dir()]
                for sub in subfolders:
                    alt_path = sub / folder_name / rel_path
                    if alt_path.exists():
                        audio_path = alt_path
                        break
            except Exception:
                pass

        # 4. Load audio with librosa
        try:
            if not audio_path.exists():
                return None # Return None for collate_fn to handle
            
            speech, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            if len(speech) > self.max_samples:
                speech = speech[:self.max_samples]
            
            inputs = self.feature_extractor(
                speech, 
                sampling_rate=self.sampling_rate, 
                return_tensors="pt",
                padding=False
            )
            
            return {
                "input_values": inputs.input_values.squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        except Exception:
            return None
