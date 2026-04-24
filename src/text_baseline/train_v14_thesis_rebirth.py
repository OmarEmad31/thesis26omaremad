"""
Egyptian Arabic Text SER — Thesis Rebirth (v14)
=============================================
Fast-iteration single-run script replacing 5-Fold with rapid evaluation.
Implements the 5 explicitly requested Thesis Notes:
1. Advanced Eqyptian Preprocessing (Strip Fillers/Tashkeel, Ali Normalization)
2. Mean Pooling (Replacing [CLS] token)
3. Dynamic Class Weights directly injected into Loss
4. CrossEntropy Label Smoothing (0.1)
5. 10x Differential LRs + LoRA Architecture
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
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────
# PREPROCESSING (Advanced Egyptian NLP)
# ─────────────────────────────────────────────────────────
def clean_egyptian_dialect(text):
    if not isinstance(text, str): return ""
    
    # 1. Strip Tashkeel (Diacritics - causes severe noise in dialect)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # 2. Alef Normalization
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # 3. Strip Tatweel
    text = re.sub(r'\u0640', '', text)
    
    # 4. Remove Common Egyptian Fillers (Noise)
    fillers = [r'\bاه\b', r'\bيعني\b', r'\bبص\b', r'\bطيب\b', r'\bامم\b', r'\bكده\b', r'\bطب\b']
    for f in fillers:
        text = re.sub(f, '', text)
        
    # 5. Deduplicate repeating characters (e.g. كتااااب -> كتاب)
    text = re.sub(r'(.)\1+', r'\1\1', text) 
    
    # Extra clean
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        cleaned_texts = [clean_egyptian_dialect(t) for t in df['transcript'].values]
        self.enc = tokenizer(list(cleaned_texts), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        # Ensure mapping
        LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
        self.labels = [LID[e] for e in df['emotion_final']]
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (LoRA + Mean Pooling)
# ─────────────────────────────────────────────────────────
class MeanPoolingLoRAClassifier(nn.Module):
    def __init__(self, model_name="UBC-NLP/MARBERT", num_labels=7):
        super().__init__()
        # Load raw MARBERT backbone
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Inject LoRA specifically to Attention mechanisms
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        # Custom Classification Head (Using 512 intermediate layer + Heavy Dropout)
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # last_hidden_state shape: [batch, sequence_length, 768]
        last_hidden_state = outputs.last_hidden_state
        
        # MEAN POOLING LOGIC (masking out padding tokens)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Pass mean pooled vector to Head
        logits = self.head(mean_pooled)
        return logits

# ─────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/content/drive/MyDrive/Thesis Project")
    split_dir = root / "data/processed/splits/final_sanitized"
    
    # 1. LOAD STRICT RAW DATA (NO LEAKAGE, NO AUGMENTATION)
    tr_df = pd.read_csv(split_dir / "train.csv")
    va_df = pd.read_csv(split_dir / "val.csv")
    te_df = pd.read_csv(split_dir / "test.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    
    train_loader = DataLoader(TextDataset(tr_df, tokenizer), batch_size=16, shuffle=True)
    val_loader = DataLoader(TextDataset(va_df, tokenizer), batch_size=16)
    test_loader = DataLoader(TextDataset(te_df, tokenizer), batch_size=16)
    
    # Compute the Class Weights dynamically based on the Training Set distribution
    LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
    train_labels_raw = [LID[e] for e in tr_df['emotion_final']]
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_raw), y=train_labels_raw)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print("\n⚖️ Computed Class Weights for CrossEntropy:")
    for i, w in enumerate(weights):
        name = list(LID.keys())[i]
        print(f"   {name}: {w:.4f}")

    # Build Model
    model = MeanPoolingLoRAClassifier().to(device)
    
    # 10x DIFFERENTIAL LEARNING RATES
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad], 'lr': 5e-5},
        {'params': [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad], 'lr': 5e-4}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    # LABEL SMOOTHING + CLASS WEIGHTS
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    print("\n" + "="*40 + "\n🚀 INIT RAPID SINGLE-SHOT TRAINING\n" + "="*40)
    best_f1 = 0
    patience = 5
    patience_counter = 0
    path = "best_rapid_text.pt"
    
    for epoch in range(1, 20):
        model.train()
        train_loss = 0.0
        tr_p, tr_t = [], []
        
        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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
            for batch_data, batch_labels in val_loader:
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                logits = model(input_ids, attention_mask)
                p.extend(torch.argmax(logits, 1).cpu().numpy())
                t.extend(batch_labels.numpy())
        
        v_acc = accuracy_score(t, p)
        v_f1 = f1_score(t, p, average='macro')
        avg_loss = train_loss / len(train_loader)
        tr_acc = accuracy_score(tr_t, tr_p)
        
        print(f"📈 Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping kicked in. Best Val F1: {best_f1:.4f}")
                break
                
    # ─────────────────────────────────────────────────────────
    # IMMEDIATE TEST EVALUATION
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*40 + "\n🏁 FINAL TEST REPORT (SPEED RUN)\n" + "="*40)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    te_preds, te_targets = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            te_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            te_targets.extend(batch_labels.numpy())
            
    print(f"FINAL TEST ACCURACY: {accuracy_score(te_targets, te_preds):.4f}")
    print(f"FINAL TEST MACRO F1: {f1_score(te_targets, te_preds, average='macro'):.4f}\n")
    print(classification_report(te_targets, te_preds, target_names=list(LID.keys()), zero_division=0))

if __name__ == "__main__": main()
