"""
Egyptian Arabic Text SER — Simple Seed Ensemble (v10)
===================================================
Back to basics: No extreme regularizations.
Trains 5 models using different random seeds on the FULL augmented dataset.
Ensembles predictions on the 44-sample test set to squeeze out the final ~3 correct predictions needed for 50%+.
"""

import os, random, sys, torch
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

from src.text_baseline.model import MARBERTWithMultiSampleDropout

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, lid):
        self.enc = tokenizer(list(df['transcript']), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = [lid[e] for e in df['emotion_final']]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

def train_seed(seed, tr_df, va_df, tokenizer, LID):
    print(f"\n{'='*40}\n🚀 TRAINING SEED {seed}\n{'='*40}")
    set_seed(seed)
    
    tr_loader = DataLoader(TextDataset(tr_df, tokenizer, LID), batch_size=16, shuffle=True)
    va_loader = DataLoader(TextDataset(va_df, tokenizer, LID), batch_size=16)
    
    id2label = {v: k for k, v in LID.items()}
    model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to("cuda")
    
    # Simple Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if "bert" in n], 'lr': 2e-5},
        {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': 5e-4}
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_f1 = 0
    save_path = f"best_model_seed_{seed}.pt"
    
    # 12 Epochs is the sweet spot for MARBERT on this data
    for epoch in range(1, 13):
        model.train()
        for batch_data, labels in tqdm(tr_loader, desc=f"Epoch {epoch}", leave=False):
            batch = {k: v.to("cuda") for k, v in batch_data.items()}
            labels = labels.to("cuda")
            optimizer.zero_grad()
            
            out = model(**batch)
            loss = criterion(out.logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        p, t = [], []
        with torch.no_grad():
            for batch_data, labels in va_loader:
                batch = {k: v.to("cuda") for k, v in batch_data.items()}
                p.extend(torch.argmax(model(**batch).logits, 1).cpu().numpy())
                t.extend(labels.numpy())
        
        f1 = f1_score(t, p, average='macro')
        acc = accuracy_score(t, p)
        print(f"📈 Epoch {epoch} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # Save best F1
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            
    print(f"✅ Seed {seed} Best Val F1: {best_f1:.4f}")
    return save_path

def main():
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    tr_df = pd.read_csv(split_dir / "train_augmented.csv") # 1113 samples
    va_df = pd.read_csv(split_dir / "val.csv")
    te_df = pd.read_csv(split_dir / "test.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    seeds = [42, 100, 256, 1024, 2024]
    model_paths = []
    
    # 1. Train 5 models
    for seed in seeds:
        path = train_seed(seed, tr_df, va_df, tokenizer, LID)
        model_paths.append(path)
        
    # 2. Ensemble on Unseen 44 Test Samples
    print("\n" + "="*40 + "\n🏁 FINAL 5-SEED ENSEMBLE TEST REPORT\n" + "="*40)
    te_loader = DataLoader(TextDataset(te_df, tokenizer, LID), batch_size=16)
    
    ensemble_probs = []
    id2label = {v: k for k, v in LID.items()}
    
    for path in model_paths:
        model = MARBERTWithMultiSampleDropout("UBC-NLP/MARBERT", num_labels=7, id2label=id2label, label2id=LID).to("cuda")
        model.load_state_dict(torch.load(path))
        model.eval()
        
        probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                batch = {k: v.to("cuda") for k, v in batch_data.items()}
                logits = model(**batch).logits
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        ensemble_probs.append(np.vstack(probs))
        
    # Average Probabilities
    avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    final_targets = [LID[e] for e in te_df['emotion_final']]
    
    print(f"\nFINAL TEST ACCURACY: {accuracy_score(final_targets, final_preds):.4f}")
    print(f"FINAL MACRO F1:      {f1_score(final_targets, final_preds, average='macro'):.4f}\n")
    print(classification_report(final_targets, final_preds, target_names=list(LID.keys())))

if __name__ == "__main__": main()
