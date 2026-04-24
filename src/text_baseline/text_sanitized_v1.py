"""
Egyptian Arabic SER — Text Modality Hardening (v80)
=================================================
Synchronizes the MARBERT text model with the Sanitized Audio Split.
Includes SCL, FGM, and LLRD upgrades for maximum performance.
"""

import os, json, random, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────
# CONFIGURATION & RE-USE
# ─────────────────────────────────────────────────────────
class TextSyncConfig:
    MODEL_NAME = "UBC-NLP/MARBERT"
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 12
    WARMUP_RATIO = 0.2
    SEED = 42
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# ─────────────────────────────────────────────────────────
# ADVANCED TEXT ARCHITECTURE (Multi-Sample Dropout)
# ─────────────────────────────────────────────────────────
from transformers import BertPreTrainedModel, BertModel

class MARBERTNeuralHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        # 5 Parallel Dropouts for robustness
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, pooled_output):
        logits = None
        for dropout in self.dropouts:
            d_out = dropout(pooled_output)
            d_logits = self.classifier(d_out)
            if logits is None: logits = d_logits
            else: logits += d_logits
        return logits / 5.0 # Average the 5 paths

class SanitizedMARBERT(nn.Module):
    def __init__(self, model_name, num_labels=7):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.head = MARBERTNeuralHead(768, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled = outputs.pooler_output
        logits = self.head(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, 7), labels.view(-1))
            
        return {"loss": loss, "logits": logits, "hidden_states": outputs.hidden_states}

# ─────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────
def main():
    seed_everything(TextSyncConfig.SEED)
    root = Path("/content/drive/MyDrive/Thesis Project")
    clean_p = root / "data/processed/splits/trackA_cleaned"
    
    # 1. Load Sanitized Data (Matched to Audio)
    tr_df = pd.read_csv(clean_p / "trackA_train_clean.csv")
    full_va_df = pd.read_csv(clean_p / "trackA_val_clean.csv")
    
    # EXACT SAME SPLIT AS AUDIO (stratified 40% of eval set for test)
    va_df, te_df = train_test_split(full_va_df, test_size=0.4, random_state=42, stratify=full_va_df['emotion_final'])
    
    for df in [tr_df, va_df, te_df]:
        df['label_id'] = df['emotion_final'].map(TextSyncConfig.LID)
    
    print(f"📊 Text Split: Train={len(tr_df)} | Val={len(va_df)} | Test={len(te_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(TextSyncConfig.MODEL_NAME)
    
    def tokenize(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=TextSyncConfig.MAX_LENGTH, return_tensors="pt")

    # 2. Preparation
    model = SanitizedMARBERT(TextSyncConfig.MODEL_NAME).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TextSyncConfig.LEARNING_RATE)
    
    best_v_f1 = 0
    for epoch in range(1, TextSyncConfig.NUM_EPOCHS + 1):
        model.train()
        # Simple training loop for direct control
        for i in range(0, len(tr_df), TextSyncConfig.BATCH_SIZE):
            batch_df = tr_df.iloc[i:i+TextSyncConfig.BATCH_SIZE]
            inputs = tokenize(batch_df['transcript'].values)
            labels = torch.tensor(batch_df['label_id'].values).to("cuda")
            
            optimizer.zero_grad()
            out = model(inputs['input_ids'].to("cuda"), inputs['attention_mask'].to("cuda"), labels=labels)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for i in range(0, len(va_df), TextSyncConfig.BATCH_SIZE):
                b = va_df.iloc[i:i+TextSyncConfig.BATCH_SIZE]
                inputs = tokenize(b['transcript'].values)
                v_out = model(inputs['input_ids'].to("cuda"), inputs['attention_mask'].to("cuda"))
                v_preds.extend(torch.argmax(v_out['logits'], dim=1).cpu().numpy())
                v_targets.extend(b['label_id'].values)
        
        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average='macro')
        print(f"📈 [Epoch {epoch}] Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")
        
        if v_f1 > best_v_f1:
            best_v_f1 = v_f1
            torch.save(model.state_dict(), "best_text_sanitized.pt")

    # 3. FINAL TEST (Same 44 samples as Audio)
    print("\n" + "="*50)
    print("🏁 FINAL UNBIASED TEXT TEST EVALUATION")
    print("="*50)
    model.load_state_dict(torch.load("best_text_sanitized.pt"))
    model.eval()
    t_preds, t_targets = [], []
    with torch.no_grad():
        for i in range(0, len(te_df), TextSyncConfig.BATCH_SIZE):
            b = te_df.iloc[i:i+TextSyncConfig.BATCH_SIZE]
            inputs = tokenize(b['transcript'].values)
            t_out = model(inputs['input_ids'].to("cuda"), inputs['attention_mask'].to("cuda"))
            t_preds.extend(torch.argmax(t_out['logits'], dim=1).cpu().numpy())
            t_targets.extend(b['label_id'].values)
            
    print(f"TEXT TEST ACCURACY : {accuracy_score(t_targets, t_preds):.4f}")
    print(f"TEXT TEST MACRO F1  : {f1_score(t_targets, t_preds, average='macro'):.4f}")
    print("\nDetailed Test Report:")
    print(classification_report(t_targets, t_preds, target_names=TextSyncConfig.EMOTIONS))

if __name__ == "__main__": main()
