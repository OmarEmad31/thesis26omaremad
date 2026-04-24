"""
Egyptian Arabic SER — Track A Neural Fine-Tuning (v78)
=====================================================
Target: Maximize accuracy on the CLEANED Track A split.
Method: Neural Fine-Tuning (Epochs) of WavLM.
"""

import os, random, pandas as pd, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import librosa
from transformers import WavLMModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

class NeuralConfig:
    SR = 16000
    MAX_LEN = 80000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    BATCH_SIZE = 8
    EPOCHS = 12
    LR = 5e-5
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            y, _ = librosa.load(row['resolved_path'], sr=NeuralConfig.SR)
            if len(y) > NeuralConfig.MAX_LEN:
                y = y[(len(y)-NeuralConfig.MAX_LEN)//2 : (len(y)-NeuralConfig.MAX_LEN)//2 + NeuralConfig.MAX_LEN]
            else:
                y = np.pad(y, (0, max(0, NeuralConfig.MAX_LEN - len(y))))
            return {"input": torch.from_numpy(y).float(), "label": torch.tensor(row['label_id'], dtype=torch.long)}
        except: return self.__getitem__((idx + 1) % len(self.df))

class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        # Freeze bottom layers
        for layer in self.backbone.encoder.layers[:6]:
            for p in layer.parameters(): p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 7))
    def forward(self, x):
        out = self.backbone(x).last_hidden_state
        return self.head(torch.mean(out, dim=1))

def main():
    seed_everything(NeuralConfig.SEED)
    root = Path("/content/drive/MyDrive/Thesis Project")
    clean_p = root / "data/processed/splits/trackA_cleaned"
    
    if not (clean_p / "trackA_val_clean.csv").exists():
        print("❌ Error: Run leakage audit first!")
        return

    tr_df = pd.read_csv(clean_p / "trackA_train_clean.csv")
    va_df = pd.read_csv(clean_p / "trackA_val_clean.csv")
    tr_df['label_id'] = tr_df['emotion_final'].map(NeuralConfig.LID)
    va_df['label_id'] = va_df['emotion_final'].map(NeuralConfig.LID)

    train_loader = DataLoader(AudioDataset(tr_df), batch_size=NeuralConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioDataset(va_df), batch_size=NeuralConfig.BATCH_SIZE)

    model = EmotionClassifier().to(NeuralConfig.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=NeuralConfig.LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_acc = 0
    for epoch in range(1, NeuralConfig.EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits = model(batch['input'].to(NeuralConfig.DEVICE))
                loss = criterion(logits, batch['label'].to(NeuralConfig.DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input'].to(NeuralConfig.DEVICE))
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                targets.extend(batch['label'].cpu().numpy())
        
        acc = accuracy_score(targets, preds)
        print(f"   Validation Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_audio_neural.pt")

    print(f"\n🏆 Final Neural Accuracy: {best_acc:.4f}")

if __name__ == "__main__": main()
