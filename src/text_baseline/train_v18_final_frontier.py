"""
Egyptian Arabic Text SER — The Final Frontier (v18)
===================================================
The nuclear option to solve the 'Impossible Triangle':
1. Validation Accuracy -> Target 60%+
2. Test Accuracy -> Target 50%+
3. Minority Emotions (Fear/Surprise) -> F1 > 0

Mechanisms:
- WeightedRandomSampler: Massive oversampling of Fear/Surprise in training.
- Supervised Contrastive Learning (SCL): Forcing emotional clustering.
- Focal Loss (gamma=2.0): Squaring the penalty for minority errors.
- Aggressive Regularization: Dropout 0.5, Weight Decay 0.05, Label Smoothing 0.15.
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────
# PREPROCESSING
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
# LOSS ENGINES (SCL + FOCAL)
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def scl_loss(hidden, labels, temp=0.1):
    features = F.normalize(hidden, p=2, dim=1)
    sim = torch.matmul(features, features.T) / temp
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().cuda()
    mask *= (1 - torch.eye(labels.size(0), device="cuda"))
    valid = mask.sum(1) > 0
    if not valid.any(): return torch.tensor(0.0).cuda()
    log_p = (sim - torch.max(sim, 1, True)[0].detach()) - torch.log(torch.exp(sim-torch.max(sim,1,True)[0].detach()).sum(1, True) + 1e-8)
    loss = - (mask[valid] * log_p[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
    return loss.mean()

# ─────────────────────────────────────────────────────────
# MODEL (LoRA + Mean Pooling + High Reg)
# ─────────────────────────────────────────────────────────
class FinalFrontierModel(nn.Module):
    def __init__(self, model_name="UBC-NLP/MARBERT", num_labels=7):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1, bias="none")
        self.bert = get_peft_model(self.bert, lora_config)
        
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5), # AGGRESSIVE DROPOUT
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return self.head(mean_pooled), mean_pooled

# ─────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    tr_df = pd.read_csv(split_dir / "train.csv")
    va_df = pd.read_csv(split_dir / "val.csv")
    te_df = pd.read_csv(split_dir / "test.csv")
    pool_df = pd.concat([tr_df, va_df]).reset_index(drop=True)
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    
    texts = pool_df['transcript'].values
    labels = np.array([LID[e] for e in pool_df['emotion_final']])
    te_labels = np.array([LID[e] for e in te_df['emotion_final']])
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    te_loader = DataLoader(TextDataset(te_df['transcript'].values, te_labels, tokenizer), batch_size=16)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rolling_probs = []

    print("\n" + "="*50 + "\n🚀 V18: THE FINAL FRONTIER (NUCLEAR OPTION)\n" + "="*50)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n📂 FOLD {fold} PREPARATION...")
        
        # OVERSAMPLING LOGIC
        t_labels = labels[t_idx]
        class_sample_count = np.array([len(np.where(t_labels == t)[0]) for t in np.unique(t_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in t_labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        t_loader = DataLoader(TextDataset(texts[t_idx], labels[t_idx], tokenizer), batch_size=16, sampler=sampler)
        v_loader = DataLoader(TextDataset(texts[v_idx], labels[v_idx], tokenizer), batch_size=16)
        
        model = FinalFrontierModel().to(device)
        
        # AGGRESSIVE WEIGHT DECAY (0.05)
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if "bert" in n], 'lr': 5e-5},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': 5e-4}
        ], weight_decay=0.05)
        
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.15)
        
        best_f1 = 0
        patience = 6 # Slightly more patience for oversampling noise
        path = f"best_v18_fold_{fold}.pt"
        
        for epoch in range(1, 30):
            model.train()
            train_loss, tr_p, tr_t = 0.0, [], []
            for batch_data, batch_labels in tqdm(t_loader, desc=f"   E{epoch}", leave=False):
                optimizer.zero_grad()
                logits, pooled = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                targets = batch_labels.to(device)
                
                loss = criterion(logits, targets) + (0.1 * scl_loss(pooled, targets))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                tr_p.extend(torch.argmax(logits, 1).cpu().numpy())
                tr_t.extend(targets.cpu().numpy())
                
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for batch_data, batch_labels in v_loader:
                    logits, _ = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                    p.extend(torch.argmax(logits, 1).cpu().numpy())
                    t.extend(batch_labels.numpy())
            
            v_acc, v_f1 = accuracy_score(t, p), f1_score(t, p, average='macro')
            print(f"   📈 E{epoch} | TrAcc: {accuracy_score(tr_t, tr_p):.4f} | VAcc: {v_acc:.4f} | VF1: {v_f1:.4f}")
            
            if v_f1 > best_f1:
                best_f1 = v_f1
                torch.save(model.state_dict(), path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: break
        
        model.load_state_dict(torch.load(path))
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch_data, _ in te_loader:
                logits, _ = model(batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device))
                fold_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        
        rolling_probs.append(np.vstack(fold_probs))
        ensemble_preds = np.argmax(np.mean(rolling_probs, axis=0), axis=1)
        print(f"   🔥 ROLLING ENSEMBLE TEST ACC: {accuracy_score(te_labels, ensemble_preds):.4f}")
        print(classification_report(te_labels, ensemble_preds, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__": main()
