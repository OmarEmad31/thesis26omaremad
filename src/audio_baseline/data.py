"""Audio Dataset and Preprocessing (librosa + transformers)."""

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

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
            self.df = None # To be set by fold logic
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
        # Ensure we are using strings and healthy slashes
        folder_name = str(row["folder"]).strip()
        rel_path = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        
        # The true path on SSD
        audio_path = self.data_root / folder_name / rel_path
        
        # 3. Load audio with librosa
        try:
            if not audio_path.exists():
                # Try a fallback in case there's an extra 'dataset' folder in the way
                alt_path = self.data_root.parent / folder_name / rel_path
                if alt_path.exists():
                    audio_path = alt_path
            
            speech, sr = librosa.load(audio_path, sr=self.sampling_rate)
        except Exception as e:
            # Reverted: Show skipping message as per user request
            print(f"⚠️ Skipping {folder_name}/{rel_path} (File not found)")
            speech = np.zeros(self.max_samples, dtype=np.float32)
        # 4. Truncate / Pad
        if len(speech) > self.max_samples:
            speech = speech[:self.max_samples]
        
        # 5. Extract features
        # Returns a dict with 'input_values' (and possibly 'attention_mask')
        inputs = self.feature_extractor(
            speech, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt",
            padding=False
        )
        
        # Remove the batch dimension added by the extractor
        input_values = inputs.input_values.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def collate_audio_fn(batch):
    """Pad a batch of waveforms to the same length."""
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad to the longest in the batch
    # (batch, longest_seq)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )
    
    # Create attention mask (1 for real, 0 for padding)
    # Most Wav2Vec2/WavLM models don't strictly need masks for short clips, 
    # but it's safer to have.
    attention_mask = (inputs_padded != 0).long()
    
    return {
        "input_values": inputs_padded,
        "attention_mask": attention_mask,
        "labels": torch.stack(labels)
    }
