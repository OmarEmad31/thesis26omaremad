"""
Egyptian Arabic Text SER — Sanitized Focal Ensemble (v94)
======================================================
Combines 5-Fold Stratified splits with Weighted Focal Loss.
This is the designated path to 50%+ on the 44-sample test set.
"""

import os, json, random, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Path injection
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, set_seed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

from src.text_baseline.model import MARBERTWithMultiSampleDropout
from src.text_baseline import config

# ─────────────────────────────────────────────────────────
# LOSS & UTILS
# ─────────────────────────────────────────────────────────

class WeightedFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0):
        super().__init__()
        self.weights = weights # Class weights
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, lid):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ─────────────────────────────────────────────────────────
# K-FOLD ENGINE
# ─────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    # Load Locked Splits
    tr_df = pd.read_csv(split_dir / "train.csv")
    va_df = pd.read_csv(split_dir / "val.csv")
    te_df = pd.read_csv(split_dir / "test.csv")
    
    # Pool Train + Val for K-Fold (100% Honest approach)
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    pool_df['label_id'] = pool_df['emotion_final'].map(LID)
    
    pool_texts = pool_df['transcript'].values
    pool_labels = pool_df['label_id'].values
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_preds = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(pool_texts, pool_labels)):
        print(f"\n🚀 TRAINING FOLD {fold}...")
        
        # Prepare Fold Data
        t_ds = TextDataset(pool_texts[t_idx], pool_labels[t_idx], tokenizer, LID)
        v_ds = TextDataset(pool_texts[v_idx], pool_labels[v_idx], tokenizer, LID)
        t_loader = DataLoader(t_ds, batch_size=16, shuffle=True)
        v_loader = DataLoader(v_ds, batch_size=16)

        id2label = {v: k for k, v in LID.items()}
        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to(device)
        
        # Differential LRs
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if "bert" in n], 'lr': 2e-5},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': 1e-3}
        ], weight_decay=0.01)
        
        # Compute weights for this fold
        c = pool_df.iloc[t_idx]['label_id'].value_counts().to_dict()
        w = torch.tensor([1.0/c.get(i, 1) for i in range(7)]).to(device)
        w = w / w.sum() * 7 # Normalize
        criterion = WeightedFocalLoss(weights=w)

        best_v_f1 = 0
        fold_save_path = f"best_model_fold_{fold}.pt"
        
        for epoch in range(1, 13): # 12 epochs per fold is usually sweet spot
            model.train()
            for batch in t_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(out.logits, batch['labels'])
                loss.backward()
                optimizer.step()
            
            # Val
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for b in v_loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    vp.extend(torch.argmax(model(b['input_ids'], b['attention_mask']).logits, 1).cpu().numpy())
                    vt.extend(b['labels'].cpu().numpy())
            
            vf1 = f1_score(vt, vp, average='macro')
            if vf1 > best_v_f1:
                best_v_f1 = vf1
                torch.save(model.state_dict(), fold_save_path)
            
        print(f"✅ Fold {fold} Best Val F1: {best_v_f1:.4f}")

    # ─────────────────────────────────────────────────────────
    # ENSEMBLE PREDICTION (ON TEST 44)
    # ─────────────────────────────────────────────────────────
    print("\n🏁 FINAL ENSEMBLE EVALUATION (ON 44 TEST SAMPLES)")
    te_ds = TextDataset(te_df['transcript'].values, [LID[e] for e in te_df['emotion_final']], tokenizer, LID)
    te_loader = DataLoader(te_ds, batch_size=16)
    
    ensemble_probs = []
    for fold in range(5):
        id2label = {v: k for k, v in LID.items()}
        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to(device)
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pt"))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for b in te_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b['input_ids'], b['attention_mask']).logits
                fold_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        ensemble_probs.append(np.vstack(fold_probs))
    
    avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    final_targets = [LID[e] for e in te_df['emotion_final']]
    
    print(f"\nENSEMBLE ACCURACY: {accuracy_score(final_targets, final_preds):.4f}")
    print(classification_report(final_targets, final_preds, target_names=list(LID.keys())))

if __name__ == "__main__": main()
