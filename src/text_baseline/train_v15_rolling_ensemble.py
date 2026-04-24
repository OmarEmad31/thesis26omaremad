"""
Egyptian Arabic Text SER — Rolling Ensemble Rebirth (v15)
=========================================================
Features the v14 Rebirth Architecture (LoRA, Mean Pooling, Class Weights,
Dialect NLP cleaner) wrapped in a 5-Fold Cross Validation.
NEW FEATURE: "Live Rolling Evaluation" - It evaluates the True Test set
at the end of EVERY fold, showing you step-by-step how the Ensemble builds
and stabilizes the accuracy up to the 50%+ threshold.
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
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────
# PREPROCESSING (Advanced Egyptian NLP)
# ─────────────────────────────────────────────────────────
def clean_egyptian_dialect(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)    # Drop Tashkeel
    text = re.sub(r'[أإآ]', 'ا', text)                  # Alef Normalization
    text = re.sub(r'\u0640', '', text)                  # Drop Tatweel
    fillers = [r'\bاه\b', r'\bيعني\b', r'\bبص\b', r'\bطيب\b', r'\bامم\b', r'\bكده\b', r'\bطب\b']
    for f in fillers: text = re.sub(f, '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)             # Deduplicate extensions
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        cleaned_texts = [clean_egyptian_dialect(t) for t in texts]
        self.enc = tokenizer(list(cleaned_texts), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = labels
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (LoRA + Mean Pooling)
# ─────────────────────────────────────────────────────────
class MeanPoolingLoRAClassifier(nn.Module):
    def __init__(self, model_name="UBC-NLP/MARBERT", num_labels=7):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # MEAN POOLING 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return self.head(mean_pooled)

# ─────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    # 1. LOAD ORIGINAL SANITIZED DATA (511 Train/Val + 44 Test)
    tr_df = pd.read_csv(split_dir / "train.csv")
    va_df = pd.read_csv(split_dir / "val.csv")
    te_df = pd.read_csv(split_dir / "test.csv")
    
    # Combine Train and Val for the K-Fold pool (total 575 samples)
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts = pool_df['transcript'].values
    labels = np.array([LID[e] for e in pool_df['emotion_final']])
    
    # Test Data setup
    te_texts = te_df['transcript'].values
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    te_loader = DataLoader(TextDataset(te_texts, te_labels, tokenizer), batch_size=16)
    
    # 2. CLASS WEIGHTS (Computed over all 575 pool samples for stability)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Master tracker arrays for our "Rolling Test Evaluation"
    rolling_probs = []

    print("\n" + "="*50 + "\n🚀 LIVE ROLLING ENSEMBLE ENGINE\n" + "="*50)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n📂 FOLD {fold} PREPARATION...")
        
        t_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, shuffle=True)
        v_loader = DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)
        
        # Build Fresh Model for the Fold
        model = MeanPoolingLoRAClassifier().to(device)
        
        # 10x DIFFERENTIAL LRs
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad], 'lr': 5e-5},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad], 'lr': 5e-4}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        best_f1 = 0
        patience = 5
        patience_counter = 0
        path = f"best_rolling_fold_{fold}.pt"
        
        for epoch in range(1, 20):
            model.train()
            train_loss = 0.0
            tr_p, tr_t = [], []
            
            for batch_data, batch_labels in tqdm(t_loader, desc=f"   Epoch {epoch}", leave=False):
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                targets = batch_labels.to(device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tr_p.extend(torch.argmax(logits, 1).cpu().numpy())
                tr_t.extend(targets.cpu().numpy())
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    input_ids = batch_data['input_ids'].to(device)
                    attention_mask = batch_data['attention_mask'].to(device)
                    logits = model(input_ids, attention_mask)
                    p.extend(torch.argmax(logits, 1).cpu().numpy())
                    t.extend(batch_labels.numpy())
            
            v_acc = accuracy_score(t, p)
            v_f1 = f1_score(t, p, average='macro')
            tr_acc = accuracy_score(tr_t, tr_p)
            avg_loss = train_loss / len(t_loader)
            
            print(f"   📈 E{epoch} | TrL: {avg_loss:.4f} | TrAcc: {tr_acc:.4f} | VAcc: {v_acc:.4f} | VF1: {v_f1:.4f}")
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(model.state_dict(), path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   🛑 Early stop. Best Val F1: {best_f1:.4f}")
                    break
        
        # ─────────────────────────────────────────────────────────
        # LIVE STEP-BY-STEP TEST EVALUATION (THE ROLLING ENSEMBLE)
        # ─────────────────────────────────────────────────────────
        print(f"\n🔍 Evaluating Fold {fold} on UNSEEN 44 Test Samples...")
        model.load_state_dict(torch.load(path))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                logits = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                fold_probs.append(probs)
                
        fold_probs_matrix = np.vstack(fold_probs)
        rolling_probs.append(fold_probs_matrix)
        
        # 1. Fold's Individual Score
        fold_preds = np.argmax(fold_probs_matrix, axis=1)
        fold_test_acc = accuracy_score(te_labels, fold_preds)
        
        # 2. Cumulative Rolling Average Score
        avg_rolling_matrix = np.mean(rolling_probs, axis=0) # Averages all folds trained so far
        rolling_preds = np.argmax(avg_rolling_matrix, axis=1)
        rolling_test_acc = accuracy_score(te_labels, rolling_preds)
        rolling_test_f1 = f1_score(te_labels, rolling_preds, average='macro')
        
        print(f"   • Fold {fold} Individual Test Acc : {fold_test_acc:.4f}")
        print(f"   🔥 CURRENT ROLLING ENSEMBLE TEST ACC : {rolling_test_acc:.4f} (from 1 to {fold+1} folds)")
        
        # Print classification report if it hit our 50% target
        if rolling_test_acc >= 0.50:
            print(f"   🏆 TARGET ACQUIRED! CURRENT MACRO F1: {rolling_test_f1:.4f}")
            print(classification_report(te_labels, rolling_preds, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__": main()
