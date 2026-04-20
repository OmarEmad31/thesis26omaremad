"""
HArnESS-Conformer-Masterclass: Egyptian Arabic SER (Flagship Architecture).
The 600M-parameter Definitive SOTA for individual audio.

Backbone: Wav2Vec2-BERT-2.0 (Conformer-based, 600M params).
Head: Multi-Head Self-Attention + Hybrid (Avg+Max) Pooling.
Loss: Focal Loss (Hard-Example Mining for Egyptian nuances).
Stability: 8-Step Gradient Accumulation + Discriminative Tuning.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2BertModel, get_cosine_schedule_with_warmup
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
        # Cross entropy loss
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
        # mask: 1 for real, 0 for pad
        # Attentive Avg
        attn_logits = self.attention(x)
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(expanded_mask == 0, -1e9)
        weights = torch.softmax(attn_logits, dim=1)
        avg_pooled = torch.sum(x * weights, dim=1)
        
        # Global Max (Ignoring pad)
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
        
        # Hybrid Head (1024*2 = 2048 dim)
        self.pooling = HybridAttentionPooling(1024)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None):
        # 1. Normalization
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        
        # 2. Conformer Backbone
        # Wav2Vec2-BERT expects raw wav, produces hidden_states
        outputs = self.backbone(wav).last_hidden_state # [BS, Seq, 1024]
        
        # 3. Mask Alignment (W2V-BERT downsamples by 320)
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
        else:
            down_mask = None
            
        # 4. Hybrid Pooling
        pooled = self.pooling(outputs, mask=down_mask)
        
        # 5. Classification
        logits = self.classifier(pooled)
        return logits

# ---------------------------------------------------------------------------
# DATASET: FULL CONTEXT (10S)
# ---------------------------------------------------------------------------
class FlagshipEgyptianDataset(Dataset):
    def __init__(self, df, audio_map):
        self.df = df
        self.audio_map = audio_map
        self.max_len = 160000 

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        if path is None: raise FileNotFoundError(f"Missing: {bn}")
        
        audio, _ = librosa.load(path, sr=16000, mono=True)
        
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
            mask[:] = 1.0
        else:
            mask[:len(audio)] = 1.0
            audio = np.pad(audio, (0, self.max_len - len(audio)))
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
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
            l = model(b["wav"].to(device), b["mask"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("🏗️ Initializing Conformer Masterclass Engine...")
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

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)

    model = ConformerSOTAFlagship(len(classes)).to(device)
    loss_fn = FocalLoss(alpha=weights, gamma=2.0) 

    # GRADIENT ACCUMULATION (Target Effective BS 32)
    # W2V-BERT is massive, BS 4 is safe on 16GB.
    tr_loader = DataLoader(FlagshipEgyptianDataset(train_df, audio_map), batch_size=4, shuffle=True)
    va_loader = DataLoader(FlagshipEgyptianDataset(val_df, audio_map), batch_size=4)
    te_loader = DataLoader(FlagshipEgyptianDataset(test_df, audio_map), batch_size=4)
    accum_steps = 8

    print("\n🔥 STAGE 1: CONFORMER HEAD WARMUP")
    for param in model.backbone.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 4):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for i, b in enumerate(pbar):
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m)
            loss = loss_fn(logits, l) / accum_steps
            loss.backward()
            if (i+1) % accum_steps == 0:
                opt.step(); opt.zero_grad()
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: CONFORMER FINE-TUNING (SOTA PUSH)")
    for param in model.backbone.parameters(): param.requires_grad = True
    # Discriminative LR: Ultra-low because 600M params shatter easily
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 5e-7},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*20)
    
    best_va = 0
    for epoch in range(1, 21):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Flagship Ep{epoch}")
        for i, b in enumerate(pbar):
            wav = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            logits = model(wav, m)
            loss = loss_fn(logits, l) / accum_steps
            loss.backward()
            if (i+1) % accum_steps == 0:
                opt.step(); opt.zero_grad(); sch.step()
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
            
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results:")
        print(f"   VALIDATION -> Acc: {va:.3f} | Macro-F1: {vf:.3f}")
        print(f"   TESTING    -> Acc: {ta:.3f} | Macro-F1: {tf:.3f}")
        
        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "conformer_flagship_best.pt")
            print("   🌟 CONFORMER FLAGSHIP BEST SAVED")

if __name__ == "__main__": main()
