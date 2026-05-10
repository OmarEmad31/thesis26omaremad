import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path

EPOCHS = 35
BATCH_SIZE = 64
K_PER_CLASS = 4
LR = 5e-4
WEIGHT_DECAY = 0.01

LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class VideoSequenceDataset(Dataset):
    def __init__(self, df, feat_dir, suffix="_clip_seq.npy"):
        self.df = df.reset_index(drop=True)
        self.feat_dir = feat_dir
        self.suffix = suffix
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['sample_id']
        fid = sid.replace("::", "__").replace("/", "_").replace(".mp4", "")
        fpath = self.feat_dir / f"{fid}{self.suffix}"
        
        feat = np.load(fpath) # Expected shape: [16, D]
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes = sorted(list(self.class_indices.keys()))
        self.n_batches = max(1, min(len(v) for v in self.class_indices.values()) // k)

    def __iter__(self):
        pools = {c: random.sample(idxs, len(idxs)) for c, idxs in self.class_indices.items()}
        ptrs = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                if ptrs[c] + self.k > len(pools[c]):
                    pools[c] = random.sample(self.class_indices[c], len(self.class_indices[c]))
                    ptrs[c] = 0
                batch.extend(pools[c][ptrs[c] : ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch)
            yield batch

    def __len__(self): return self.n_batches

class VideoBiLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch, 16, 768]
        lstm_out, (hn, cn) = self.lstm(x)
        # hn shape: [num_layers*2, batch, hidden_dim]
        # Concat the final forward and backward hidden states from the last layer
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) # [batch, hidden_dim*2]
        logits = self.fc(hidden)
        return logits

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    root = Path(r"d:\Thesis Project")
    manifest_path = root / "video_manifest_trackA.csv"
    feat_dir = root / "data" / "processed" / "features" / "video_sequences_v1"
    suffix = "_clip_seq.npy"
    
    df = pd.read_csv(manifest_path)
    df['exists'] = df['sample_id'].apply(lambda sid: (feat_dir / f"{sid.replace('::', '__').replace('/', '_').replace('.mp4', '')}{suffix}").exists())
    df = df[(df['resolution_status'] == 'resolved') & (df['exists'] == True)]
    
    if len(df) == 0:
        print("No sequence features found. Run extraction script first!")
        return

    # 5-Fold pooling (Train + Val)
    tr_df = df[df['split'] == 'train']
    va_df = df[df['split'] == 'val']
    te_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    print(f"Loaded Sequence Features: Pool={len(pool_df)} | Test={len(te_df)}", flush=True)
    
    pool_labels = np.array([LID[e] for e in pool_df['emotion_final']])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    te_ds = VideoSequenceDataset(te_df, feat_dir, suffix)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model_paths = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(pool_labels)), pool_labels)):
        print("\n" + "="*50, flush=True)
        print(f"FOLD {fold+1}/5 - TRAINING SEQUENCE MODEL (BiLSTM)", flush=True)
        print("="*50, flush=True)
        
        fold_tr_df = pool_df.iloc[train_idx].reset_index(drop=True)
        fold_va_df = pool_df.iloc[val_idx].reset_index(drop=True)
        
        tr_ds = VideoSequenceDataset(fold_tr_df, feat_dir, suffix)
        va_ds = VideoSequenceDataset(fold_va_df, feat_dir, suffix)
        
        tr_labels_fold = [LID[e] for e in fold_tr_df['emotion_final']]
        bal_sampler = BalancedBatchSampler(tr_labels_fold, k=K_PER_CLASS)
        
        tr_loader = DataLoader(tr_ds, batch_sampler=bal_sampler, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        counts = pd.Series(tr_labels_fold).value_counts().sort_index()
        weights = torch.tensor([1.0/counts.get(i, 1) for i in range(7)], dtype=torch.float32).to(device)
        weights = weights / weights.sum() * 7.0
        
        model = VideoBiLSTM(input_dim=768).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(tr_loader))
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        best_f1 = 0
        save_path = root / f"best_video_seq_fold_{fold}.pt"
        model_paths.append(save_path)
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            ep_loss = 0
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                logits = model(x)
                loss = criterion(logits, y)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                ep_loss += loss.item()
                
            model.eval()
            p_val, t_val = [], []
            with torch.no_grad():
                for x, y in va_loader:
                    x = x.to(device)
                    logits = model(x)
                    p_val.extend(torch.argmax(logits, 1).cpu().numpy())
                    t_val.extend(y.numpy())
                    
            val_acc = accuracy_score(t_val, p_val)
            val_f1 = f1_score(t_val, p_val, average='macro')
            
            star = ""
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                star = "BEST"
                
            if epoch % 5 == 0 or star == "BEST":
                print(f"Fold {fold+1} | Epoch {epoch:02d} | Loss: {ep_loss/len(tr_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} {star}", flush=True)

    print("\n" + "="*50, flush=True)
    print("FINAL SEQUENCE MODEL EVALUATION (5-FOLD SOFT-VOTING ENSEMBLE)", flush=True)
    print("="*50, flush=True)
    
    ensemble_probs = []
    t_test = []
    
    for path in model_paths:
        model = VideoBiLSTM(input_dim=768).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        
        probs = []
        t_test_fold = []
        with torch.no_grad():
            for x, y in te_loader:
                x = x.to(device)
                logits = model(x)
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())
                t_test_fold.extend(y.numpy())
        
        ensemble_probs.append(np.vstack(probs))
        if len(t_test) == 0:
            t_test = t_test_fold
            
    avg_probs = np.mean(ensemble_probs, axis=0)
    p_test = np.argmax(avg_probs, axis=1)
            
    print(f"Final Test Accuracy:   {accuracy_score(t_test, p_test):.4f}")
    print(f"Final Test UAR:        {balanced_accuracy_score(t_test, p_test):.4f}")
    print(f"Final Test Macro F1:   {f1_score(t_test, p_test, average='macro'):.4f}")
    print(f"Final Test Weighted F1:{f1_score(t_test, p_test, average='weighted'):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(t_test, p_test, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__":
    main()
