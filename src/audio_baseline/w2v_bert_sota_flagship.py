"""
HArnESS-W2V-BERT-SOTA: Egyptian Arabic SER (The 50% Milestone Push).
The 600-Million Parameter Paradigm Shift.

Backbone: facebook/w2v-bert-2.0 (The Conformer King).
Innovation: 160-dim Log-Mel Filterbanks + High-Resolution Acoustic Conformers.
Memory: Gradient Checkpointing + Accumulation (Simulated Batch Size: 64).
Target: 50% Milestone.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2BertModel, 
    AutoFeatureExtractor,
    get_cosine_schedule_with_warmup
)
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
# LOSS: SUPCON (Geometric Resolution)
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]; labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        features = F.normalize(features, dim=1)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# ARCHITECTURE: W2V-BERT 2.0 (THE GIANT)
# ---------------------------------------------------------------------------
class W2VBertGiant(nn.Module):
    def __init__(self, num_labels, model_name="facebook/w2v-bert-2.0"):
        super().__init__()
        self.bert = Wav2Vec2BertModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable() # MANDATORY for 600M
        
        # W2V-BERT output hidden dim is 1024
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, features, mode="train"):
        # W2V-BERT takes log-mel features directly
        outputs = self.bert(features).last_hidden_state
        
        # Mean Pooling for Context
        pooled = outputs.mean(dim=1)
            
        if mode == "contrast": return self.projector(pooled)
        elif mode == "classify": return self.classifier(pooled)
        else: return self.projector(pooled), self.classifier(pooled)

# ---------------------------------------------------------------------------
# DATASET: LOG-MEL FILTERBANKS
# ---------------------------------------------------------------------------
class BertAcousticDataset(Dataset):
    def __init__(self, df, audio_map, extractor):
        self.df = df; self.audio_map = audio_map; self.extractor = extractor; self.max_sec = 10

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # Fixed 10s Window
        if len(audio) > 160000: audio = audio[:160000]
        else: audio = np.pad(audio, (0, 160000 - len(audio)))
            
        # W2V-Bert Extractor generates 160-dim log-mel features
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        return {"features": inputs.input_features.squeeze(0), 
                "label": torch.tensor(row["label_id"], dtype=torch.long)}

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["features"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("🏗️ Initializing W2V-BERT 2.0 Giant (600M Parameters)...")
    extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    classes = sorted(train_df["emotion_final"].unique()); lid = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)

    model = W2VBertGiant(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.07); l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    # ACCUMULATION to simulate Batch Size 64 with 600M params
    accum_steps = 4
    batch_size = 16 # Adjust if OOM
    tr_loader = DataLoader(BertAcousticDataset(train_df, audio_map, extractor), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(BertAcousticDataset(val_df, audio_map, extractor), batch_size=batch_size)
    te_loader = DataLoader(BertAcousticDataset(test_df, audio_map, extractor), batch_size=batch_size)

    # Full fine-tuning
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*30)
    
    best_va = 0
    for epoch in range(1, 41):
        model.train(); pbar = tqdm(tr_loader, desc=f"Giant Ep{epoch}")
        for i, b in enumerate(pbar):
            f, l = b["features"].to(device), b["label"].to(device)
            p, c = model(f, mode="both"); loss = (1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)) / accum_steps
            loss.backward()
            if (i+1) % accum_steps == 0:
                opt.step(); opt.zero_grad(); sch.step()
        
        va, vf = evaluate(model, va_loader, device); ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} F1: {vf:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "w2v_bert_best.pt")

if __name__ == "__main__": main()
