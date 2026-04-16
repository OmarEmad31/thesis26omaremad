import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import librosa
from tqdm import tqdm
from pathlib import Path

# Fix relative imports
import sys
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
from src.text_baseline.train import SupervisedContrastiveLoss

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# SPECAUGMENT: The Acoustic Generalizer
# ==============================================================================
def apply_spec_augment(mel_spectrogram, freq_mask_param=15, time_mask_param=30, num_masks=2):
    """
    Applies SpecAugment: Randomly masks blocks of frequency and time.
    Forces the model to learn underlying emotional texture instead of specific pitches.
    mel_spectrogram shape expected: (N_MELS, Time_Frames)
    """
    n_mels, n_steps = mel_spectrogram.shape
    augmented = mel_spectrogram.copy()
    
    for _ in range(num_masks):
        # Frequency Masking
        f_mask = random.randint(0, freq_mask_param)
        f0 = random.randint(0, n_mels - f_mask)
        augmented[f0:f0 + f_mask, :] = 0
        
        # Time Masking
        if n_steps > time_mask_param:
            t_mask = random.randint(0, time_mask_param)
            t0 = random.randint(0, n_steps - t_mask)
            augmented[:, t0:t0 + t_mask] = 0
            
    return augmented

# ==============================================================================
# PYTORCH DATASET: Raw Audio to Native Spectrogram
# ==============================================================================
class AcousticEmotionDataset(Dataset):
    def __init__(self, df, label2id, audio_map, is_train=True):
        self.df = df
        self.label2id = label2id
        self.is_train = is_train
        self.audio_map = audio_map
        
        # Fixed temporal dimension based on max duration
        self.max_time_frames = config.MAX_AUDIO_SAMPLES // config.HOP_LENGTH + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
        basename = Path(rel).name
        
        # Secure Pathing Bypass
        if basename in self.audio_map:
            audio_path = self.audio_map[basename]
        else:
            # Fallback for silent testing
            audio_path = config.DATA_ROOT / str(row["folder"]).strip() / rel

        # 1. Load Audio
        try:
            audio, _ = librosa.load(audio_path, sr=config.SAMPLING_RATE, mono=True)
        except Exception:
            # If completely corrupted, return empty safe tensor to avoid crashing loop
            audio = np.zeros(config.MAX_AUDIO_SAMPLES, dtype=np.float32)
            
        # 2. Pad or Truncate sequentially
        if len(audio) > config.MAX_AUDIO_SAMPLES:
            audio = audio[:config.MAX_AUDIO_SAMPLES]
        else:
            # Pad with zeros
            audio = np.pad(audio, (0, config.MAX_AUDIO_SAMPLES - len(audio)))
            
        # 3. Extract Acoustic Mathematical Features
        melspec = librosa.feature.melspectrogram(
            y=audio, 
            sr=config.SAMPLING_RATE, 
            n_mels=config.N_MELS,
            hop_length=config.HOP_LENGTH
        )
        
        # 4. Convert power to Decibels (Log scaling mimics logarithmic human hearing)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # 5. Improvement 1: Dynamic Data Augmentation (Only during Training)
        if self.is_train:
            log_melspec = apply_spec_augment(log_melspec)
            
        # 6. Normalize Feature Scales (Mean 0, Std 1)
        mean = np.mean(log_melspec)
        std = np.std(log_melspec) + 1e-6
        log_melspec = (log_melspec - mean) / std

        label = self.label2id[row["emotion_final"]]
        
        # Expected shape: [128, Time_Frames]
        return torch.tensor(log_melspec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# MODEL: Hybrid Conv1D + Bi-LSTM (768 Dim Output for Multimodal Parity)
# ==============================================================================
class AcousticHybridNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        # BLOCK 1: Spatial Frequency Extractor (Ears)
        # Input shape: [Batch, 128 (Channels=Mels), Time]
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        
        # BLOCK 2: Temporal Sequence Engine (Brain)
        # Setting hidden_size = 384. Bidirectional makes output 384 * 2 = 768.
        # This GUARANTEES parity with BERT/VideoMAE outputs for late fusion!
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=384, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        # BLOCK 3: Classification Head
        self.fc_classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # x is [B, 128, Time]
        
        # Convolutional Block
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.dropout(out)
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout(out)
        
        # Prepare for LSTM -> Needs [Batch, Time, Features]
        # We swap dimensions: from [B, Features, Time] -> [B, Time, Features]
        out = out.permute(0, 2, 1)
        
        # LSTM Block
        lstm_out, _ = self.lstm(out)  # lstm_out shape: [Batch, Time, 768]
        
        # Global Average Pooling across time to crush into a 1D Vector per audio
        # Shape: [Batch, 768]
        embeddings = lstm_out.mean(dim=1)
        
        # Classification
        logits = self.fc_classifier(embeddings)
        
        # We return the 768-D embeddings specifically for the SCL Loss calculation!
        return logits, embeddings

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def main():
    print("🚀 Initializing Acoustic CNN-LSTM Pipeline...")
    
    # 0. Global Fallback Drive Mapper (Fix for Missing Colab zip files)
    print("Mapping physical .wav files across Google Drive securely...")
    audio_map = {}
    search_dir = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in search_dir.rglob("*.wav"):
        audio_map[p.name] = p
    print(f"Mapped {len(audio_map)} absolute .wav file tracks.")

    # 1. Data Prep
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    label2id = {lbl: i for i, lbl in enumerate(sorted(all_df["emotion_final"].unique()))}
    num_labels = len(label2id)
    
    X_indices = np.arange(len(all_df))
    y_labels = np.array([label2id[lbl] for lbl in all_df["emotion_final"]])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n======================================================================")
    print(f"🔥 ARCHITECTURE B: Conv1D-BiLSTM | Features: Mel-Spectrogram ({device.upper()})")
    print(f"======================================================================")

    all_fold_f1 = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_indices, y_labels)):
        print(f"\n[FOLD {fold_idx + 1} / 5]")
        
        train_fold_df = all_df.iloc[train_index].reset_index(drop=True)
        val_fold_df = all_df.iloc[val_index].reset_index(drop=True)
        
        train_dataset = AcousticEmotionDataset(train_fold_df, label2id, audio_map, is_train=True)
        val_dataset = AcousticEmotionDataset(val_fold_df, label2id, audio_map, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Improvement 3: Stratified Class Weights natively injected into CrossEntropy
        class_w = compute_class_weight("balanced", classes=np.unique(y_labels[train_index]), y=y_labels[train_index])
        class_w_tensor = torch.tensor(class_w, dtype=torch.float32).to(device)
        
        criterion_ce = nn.CrossEntropyLoss(weight=class_w_tensor)
        criterion_scl = SupervisedContrastiveLoss(temperature=config.SCL_TEMP)
        
        model = AcousticHybridNet(num_classes=num_labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
        
        best_f1 = 0
        patience = 0
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            model.train()
            total_loss = 0
            
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits, embeddings = model(inputs)
                
                loss_ce = criterion_ce(logits, labels)
                # Improvement 2: Supervised Contrastive Loss active on the 768-D embeddings
                loss_scl = criterion_scl(embeddings, labels)
                
                loss = (1 - config.SCL_WEIGHT) * loss_ce + config.SCL_WEIGHT * loss_scl
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            scheduler.step()
            
            # Validation
            model.eval()
            all_preds, all_labels_val = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    logits, _ = model(inputs)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels_val.extend(labels.numpy())
                    
            val_acc = accuracy_score(all_labels_val, all_preds)
            val_f1 = f1_score(all_labels_val, all_preds, average="macro")
            
            print(f"Epoch {epoch:2d}/{config.NUM_EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                
                best_model_path = config.CHECKPOINT_DIR / f"fold_{fold_idx}"
                best_model_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path / "best_acoustic_model.pt")
            else:
                patience += 1
                
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping triggered. Best Macro F1: {best_f1:.4f}")
                break
                
        all_fold_f1.append(best_f1)
        print(f"✅ Fold {fold_idx + 1} Best Marco F1: {best_f1 * 100:.2f}%")

    print(f"\n🏆 ALL FOLDS COMPLETE. Mean F1 Macro: {np.mean(all_fold_f1) * 100:.2f}%")

if __name__ == "__main__":
    
    # Initialize deterministic behavior
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    main()
