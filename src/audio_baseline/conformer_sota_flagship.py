"""
HArnESS-Conformer-Masterclass: Egyptian Arabic SER (Flagship Architecture).
The 600M-parameter Definitive SOTA for individual audio.

Backbone: Wav2Vec2-BERT-2.0 (Conformer-based, 600M params).
Head: Multi-Head Self-Attention + Hybrid (Avg+Max) Pooling.
Pre-processing: 160-dim Log-Mel Filterbanks (Wav2Vec2BertFeatureExtractor).
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2BertModel, Wav2Vec2BertFeatureExtractor, get_cosine_schedule_with_warmup
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
# LOSS: STABLE FOCAL LOSS
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ---------------------------------------------------------------------------
# ARCHITECTURE: CONFORMER + HYBRID ATTENTION
# ---------------------------------------------------------------------------
class HybridAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        attn_logits = self.attention(x)
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(expanded_mask == 0, -1e9)
        weights = torch.softmax(attn_logits, dim=1)
        avg_pooled = torch.sum(x * weights, dim=1)
        
        if mask is not None:
            x_masked = x.masked_fill(expanded_mask == 0, -1e9)
            max_pooled, _ = torch.max(x_masked, dim=1)
        else:
            max_pooled, _ = torch.max(x, dim=1)
        return torch.cat([avg_pooled, max_pooled], dim=-1)

class ConformerSOTAFlagship(nn.Module):
    def __init__(self, num_labels, model_name="facebook/w2v-bert-2.0"):
        super().__init__()
        self.backbone = Wav2Vec2BertModel.from_pretrained(model_name)
        self.pooling = HybridAttentionPooling(1024)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_features, attention_mask=None):
        outputs = self.backbone(input_features, attention_mask=attention_mask).last_hidden_state 
        pooled = self.pooling(outputs, mask=attention_mask)
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------------------------
# DATASET: SPECTRAL EXTRACTION
# ---------------------------------------------------------------------------
class FlagshipEgyptianDataset(Dataset):
    def __init__(self, df, audio_map, feature_extractor):
        self.df = df
        self.audio_map = audio_map
        self.feature_extractor = feature_extractor
        self.max_len = 160000 

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        inputs = self.feature_extractor(audio, sampling_rate=16000, max_length=self.max_len, 
                                       truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_features": inputs.input_features.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# TRAINING SOTA
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["input_features"].to(device), b["attention_mask"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    P_NAME = "facebook/w2v-bert-2.0"
    feature_extractor = Wav2Vec2BertFeatureExtractor.from_pretrained(P_NAME)

    print("🏗️ Initializing Conformer Acoustic Masterclass...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(label2id)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(train_df["label_id"].values), 
                                               y=train_df["label_id"].values), dtype=torch.float32).to(device)

    model = ConformerSOTAFlagship(len(classes), P_NAME).to(device)
    loss_fn = FocalLoss(alpha=weights, gamma=2.0) 

    tr_loader = DataLoader(FlagshipEgyptianDataset(train_df, audio_map, feature_extractor), batch_size=4, shuffle=True)
    va_loader = DataLoader(FlagshipEgyptianDataset(val_df, audio_map, feature_extractor), batch_size=4)
    te_loader = DataLoader(FlagshipEgyptianDataset(test_df, audio_map, feature_extractor), batch_size=4)
    accum_steps = 8

    print("\n🔥 STAGE 1: ACOUSTIC WARMUP")
    for param in model.backbone.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 4):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for i, b in enumerate(pbar):
            x = b["input_features"].to(device); l = b["label"].to(device); m = b["attention_mask"].to(device)
            logits = model(x, m); loss = loss_fn(logits, l) / accum_steps; loss.backward()
            if (i+1) % accum_steps == 0: opt.step(); opt.zero_grad()
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: ACOUSTIC FINE-TUNING")
    for param in model.backbone.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 5e-7},
                            {"params": model.classifier.parameters(), "lr": 1e-4}], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*20)
    
    best_va = 0
    for epoch in range(1, 21):
        model.train()
        pbar = tqdm(tr_loader, desc=f"SOTA Ep{epoch}")
        for i, b in enumerate(pbar):
            x = b["input_features"].to(device); l = b["label"].to(device); m = b["attention_mask"].to(device)
            logits = model(x, m); loss = loss_fn(logits, l) / accum_steps; loss.backward()
            if (i+1) % accum_steps == 0: opt.step(); opt.zero_grad(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} F1: {vf:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "conformer_flagship_best.pt")

if __name__ == "__main__": main()
