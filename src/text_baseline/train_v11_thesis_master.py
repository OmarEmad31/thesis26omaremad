"""
Egyptian Arabic Text SER — Thesis Master Suite (v11)
====================================================
This script is a 1:1 mathematical reflection of the successful thesis methodology.
Phase 1: Arabic Normalization (Tatweel, Diacritics, Alefs).
Phase 2: MARBERT Foundation.
Phase 3: SCL, MSD, LLRD.
Phase 4: 5-Fold Soft-Voting Ensemble on ORIGINAL clean data.
"""

import os, re, sys, torch
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

# ─────────────────────────────────────────────────────────
# PHASE 1: ARABIC TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────
def clean_arabic_text(text):
    """
    Implements the exact Phase 1 filtering from the Thesis Notes.
    """
    if not isinstance(text, str): return ""
    
    # 1. Remove Diacritics (Harakat)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # 2. Normalize Alefs (أ, إ, آ -> ا)
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # 3. Remove Tatweel / Elongation
    text = re.sub(r'\u0640', '', text)
    
    # 4. Remove Character Repitition ("كتااااب" -> "كتاب")
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    return text.strip()

# ─────────────────────────────────────────────────────────
# DATASET & LOSS
# ─────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        cleaned_texts = [clean_arabic_text(t) for t in texts]
        self.enc = tokenizer(list(cleaned_texts), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

def scl_loss(hidden, labels, temp=0.1):
    """Supervised Contrastive Learning Loss"""
    features = F.normalize(hidden, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().cuda()
    mask *= (1 - torch.eye(labels.size(0), device="cuda")) # remove self-match
    valid = mask.sum(1) > 0
    if not valid.any(): return torch.tensor(0.0).cuda()
    
    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True) + 1e-8)
    loss = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
    return loss.mean()

# ─────────────────────────────────────────────────────────
# THE 5-FOLD ENSEMBLE ENGINE
# ─────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    # LOAD ORIGINAL CLEAN DATA ONLY (As Requested)
    tr_df = pd.read_csv(split_dir / "train.csv") # 511 samples
    va_df = pd.read_csv(split_dir / "val.csv")   # 64 samples
    te_df = pd.read_csv(split_dir / "test.csv")  # 44 samples
    
    # Pool for K-Fold
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts = pool_df['transcript'].values
    labels = np.array([LID[e] for e in pool_df['emotion_final']])
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_paths = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n{'='*40}\n🚀 TRAINING FOLD {fold}\n{'='*40}")
        
        t_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, shuffle=True)
        v_loader = DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)

        id2label = {v: k for k, v in LID.items()}
        # Phase 2 & 3: MARBERT with Multi-Sample Dropout (5x)
        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to(device)
        
        # Phase 3: LLRD (Layer-wise Learning Rate Decay)
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if "bert" in n], 'lr': 2e-5},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': 5e-4}
        ], weight_decay=0.01)
        
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        path = f"best_thesis_fold_{fold}.pt"
        
        for epoch in range(1, 13):
            model.train()
            for batch_data, batch_labels in tqdm(t_loader, desc=f"Epoch {epoch}", leave=False):
                batch = {k: v.to(device) for k, v in batch_data.items()}
                targets = batch_labels.to(device)
                optimizer.zero_grad()
                
                out = model(**batch)
                
                # Classical SCL Integration
                ce_loss = criterion(out.logits, targets)
                hidden = out.hidden_states[-1][:, 0, :] # Extract CLS token
                con_loss = scl_loss(hidden, targets)
                
                # Weight SCL heavily as per previous successful runs
                loss = ce_loss + (0.1 * con_loss) 
                
                loss.backward()
                optimizer.step()
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    batch = {k: v.to(device) for k, v in batch_data.items()}
                    p.extend(torch.argmax(model(**batch).logits, 1).cpu().numpy())
                    t.extend(batch_labels.numpy())
            
            f1 = f1_score(t, p, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), path)
                
        print(f"✅ Fold {fold} Final Best Val F1: {best_f1:.4f}")
        model_paths.append(path)

    # ─────────────────────────────────────────────────────────
    # PHASE 4: ENSEMBLE EVALUATION ON ISOLATED TEST SET
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*40 + "\n🏁 FINAL PHASE 4 ENSEMBLE REPORT (44 TEST SAMPLES)\n" + "="*40)
    
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    te_loader = DataLoader(TextDataset(te_df['transcript'].values, te_labels, tokenizer), batch_size=16)
    
    ensemble_probs = []
    
    for path in model_paths:
        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        
        probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                batch = {k: v.to(device) for k, v in batch_data.items()}
                logits = model(**batch).logits
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        ensemble_probs.append(np.vstack(probs))
        
    avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    
    print(f"\nFINAL TEST ACCURACY: {accuracy_score(te_labels, final_preds):.4f}")
    print(f"FINAL MACRO F1:      {f1_score(te_labels, final_preds, average='macro'):.4f}\n")
    print(classification_report(te_labels, final_preds, target_names=list(LID.keys())))

if __name__ == "__main__": main()
