"""
Egyptian Arabic SER — Ultimate Neural Suite (v79)
==============================================
1. Layer-Weighted Fusion: Learns to mix all 12 WavLM layers.
2. Progressive Unfreezing: Head-only warmup then full-backbone.
3. Weighted Sampling: Balances rare classes (Fear/Surprise).
4. Forensic Logging: Accurate tracking of Train/Val/Test metrics.
"""

import os, random, warnings, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from transformers import WavLMModel, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
class UltimateConfig:
    SR = 16000
    MAX_LEN = 80000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    BATCH_SIZE = 8
    EPOCHS = 15
    BASE_LR = 4e-5
    HEAD_LR = 1e-3
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────
# MODEL (Layer-Weighted Architecture)
# ─────────────────────────────────────────────────────────
class LayerWeightedClassifier(nn.Module):
    def __init__(self, num_layers=13): # 12 + 1 input layer
        super().__init__()
        self.config = AutoConfig.from_pretrained("microsoft/wavlm-base-plus")
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
        
        # Learnable layer weights (The "Secret Sauce")
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # Project 768 to 1536 (Mean + Std Pooling)
        self.head = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 7)
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        hidden_states = outputs.hidden_states # List of 13 tensors
        
        # Weighted sum of all layers
        stacked = torch.stack(hidden_states, dim=0) # [13, B, T, 768]
        norm_weights = F.softmax(self.layer_weights, dim=0)
        weighted_out = torch.sum(stacked * norm_weights.view(-1, 1, 1, 1), dim=0) # [B, T, 768]
        
        # Pooling: Mean + Std
        mean = torch.mean(weighted_out, dim=1)
        std = torch.std(weighted_out, dim=1)
        pooled = torch.cat([mean, std], dim=1) # [B, 1536]
        
        return self.head(pooled)

# ─────────────────────────────────────────────────────────
# DATA & SAMPLING
# ─────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            y, _ = librosa.load(row['resolved_path'], sr=UltimateConfig.SR)
            y, _ = librosa.effects.trim(y, top_db=25)
            if len(y) > UltimateConfig.MAX_LEN:
                y = y[:UltimateConfig.MAX_LEN]
            else:
                y = np.pad(y, (0, max(0, UltimateConfig.MAX_LEN - len(y))))
            return {"input": torch.from_numpy(y).float(), "label": torch.tensor(row['label_id'], dtype=torch.long)}
        except: return self.__getitem__((idx + 1) % len(self.df))

def run_eval(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input'].to(UltimateConfig.DEVICE))
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            targets.extend(batch['label'].cpu().numpy())
    return {
        "acc": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, average='macro'),
        "report": classification_report(targets, preds, target_names=UltimateConfig.EMOTIONS, digits=4, zero_division=0)
    }

# ─────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────
def main():
    seed_everything(UltimateConfig.SEED)
    root = Path("/content/drive/MyDrive/Thesis Project")
    clean_p = root / "data/processed/splits/trackA_cleaned"
    
    # 1. Load Sanitized Data
    tr_df = pd.read_csv(clean_p / "trackA_train_clean.csv")
    full_va_df = pd.read_csv(clean_p / "trackA_val_clean.csv")
    
    # Create 3-Way Split (Val / Test)
    va_df, te_df = train_test_split(full_va_df, test_size=0.4, random_state=42, stratify=full_va_df['emotion_final'])
    
    for df in [tr_df, va_df, te_df]:
        df['label_id'] = df['emotion_final'].map(UltimateConfig.LID)
    
    # 2. Weighted Random Sampler (Balances the Train Set)
    class_counts = tr_df['label_id'].value_counts().to_dict()
    weights = [1.0 / class_counts[i] for i in tr_df['label_id']]
    sampler = WeightedRandomSampler(weights, num_samples=len(tr_df), replacement=True)
    
    train_loader = DataLoader(AudioDataset(tr_df), batch_size=UltimateConfig.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(AudioDataset(va_df), batch_size=UltimateConfig.BATCH_SIZE)
    test_loader = DataLoader(AudioDataset(te_df), batch_size=UltimateConfig.BATCH_SIZE)

    model = LayerWeightedClassifier().to(UltimateConfig.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=UltimateConfig.BASE_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    print(f"\n📊 Split: Train={len(tr_df)} | Val={len(va_df)} | Test={len(te_df)}")
    print("🚀 Initializing Progressive Fine-Tuning Strategy...")

    best_v_f1 = 0
    for epoch in range(1, UltimateConfig.EPOCHS + 1):
        # Technique: Progressive Unfreezing
        if epoch == 1:
            print(f"\n[EPOCH {epoch}] Strategy: Warmup (Backbone Frozen, Head Only)")
            for param in model.backbone.parameters(): param.requires_grad = False
        elif epoch == 3:
            print(f"\n[EPOCH {epoch}] Strategy: Settle (Unfreezing All Layers)")
            for param in model.backbone.parameters(): param.requires_grad = True
        
        model.train()
        t_loss = 0
        for batch in tqdm(train_loader, desc=f"   Training"):
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits = model(batch['input'].to(UltimateConfig.DEVICE))
                loss = criterion(logits, batch['label'].to(UltimateConfig.DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            t_loss += loss.item()

        # Evaluation
        v_res = run_eval(model, val_loader)
        print(f"📈 [VAL] Acc: {v_res['acc']:.4f} | Macro F1: {v_res['f1']:.4f}")
        
        if v_res['f1'] > best_v_f1:
            best_v_f1 = v_res['f1']
            torch.save(model.state_dict(), "ultimate_audio_model.pt")
            print("⭐ New Best Model Saved!")

    # Final Final Check on UNSEEN TEST
    print("\n" + "="*50)
    print("🏁 FINAL UNBIASED TEST EVALUATION")
    print("="*50)
    model.load_state_dict(torch.load("ultimate_audio_model.pt"))
    te_res = run_eval(model, test_loader)
    print(f"TEST ACCURACY  : {te_res['acc']:.4f}")
    print(f"TEST MACRO F1  : {te_res['f1']:.4f}")
    print("\nDetailed Test Report:")
    print(te_res['report'])

if __name__ == "__main__": main()
