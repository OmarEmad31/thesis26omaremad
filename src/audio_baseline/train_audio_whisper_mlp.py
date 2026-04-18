"""
Method 19: Whisper-MLP "The Last Stand".
Strategy: 
1. Use OpenAI's Whisper-Small (Speech-native intelligence).
2. Freeze the Encoder.
3. Extract 768-dim embeddings for all 915 samples.
4. Train a fast MLP head on the frozen features.
Goal: >60% Accuracy in 15 minutes.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_NAME    = "openai/whisper-small"
BATCH_SIZE    = 4
LR            = 1e-3
NUM_EPOCHS    = 50 # MLP trains in seconds
SR            = 16000
MAX_DURATION  = 10 # Whisper likes 30s but we clip to 10s for speed

# ---------------------------------------------------------------------------
# MODEL: Whisper Feature Extractor + MLP
# ---------------------------------------------------------------------------
class WhisperMLP(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(MODEL_NAME)
        # Freeze Encoder
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        hidden_dim = self.whisper.config.d_model # 768 for small
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_features):
        # input_features shape: [BS, 80, 3000]
        outputs = self.whisper.encoder(input_features)
        last_hidden = outputs.last_hidden_state # [BS, SeqLen, 768]
        # Global Average Pooling
        pooled = last_hidden.mean(dim=1)
        return self.head(pooled)

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class WhisperDataset(Dataset):
    def __init__(self, df, label2id, audio_map, processor):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.audio_map = audio_map
        self.processor = processor

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except:
            audio = np.zeros(SR * 30) # Whisper default length is 30s
        
        # Whisper Preprocessing
        input_features = self.processor(audio, sampling_rate=SR, return_tensors="pt").input_features
        label_id = self.label2id.get(row["emotion_final"], -1)
        
        return {
            "input_features": input_features.squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🛸 WHISPER-MLP FINAL STAND")
    
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    audio_map = {}
    src = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}; num_labels = len(label2id)

    # Weights for F1 stability
    y_tr = [label2id[l] for l in train_df["emotion_final"]]
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    weights = torch.tensor(cw, dtype=torch.float32).to(device)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    train_ds = WhisperDataset(train_df, label2id, audio_map, processor)
    val_ds   = WhisperDataset(val_df, label2id, audio_map, processor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = WhisperMLP(num_labels).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR)
    ce_fn = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0
    checkpoint = config.CHECKPOINT_DIR / "whisper_mlp_final.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); t_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep{epoch}", leave=False)
        for batch in pbar:
            feats, lbs = batch["input_features"].to(device), batch["label"].to(device)
            logits = model(feats)
            loss = ce_fn(logits, lbs)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        model.eval(); preds, truth = [] , []
        with torch.no_grad():
            for b in val_loader:
                ls = model(b["input_features"].to(device))
                preds.extend(torch.argmax(ls, 1).cpu().numpy()); truth.extend(b["label"].numpy())
        
        acc, f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
        print(f"   Ep {epoch:2d}/{NUM_EPOCHS} | Loss {t_loss/len(train_loader):.3f} | Acc {acc:.3f} | F1 {f1:.3f}")
        
        if acc > best_acc:
            best_acc = acc; torch.save(model.state_dict(), checkpoint)
            print("   🌟 New Best")
    
    print(f"\n🏆 TITAN RESULT: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
