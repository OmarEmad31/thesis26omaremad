"""
HArnESS-DeepContext-SOTA: Egyptian Arabic SER (The 50% Milestone Push).
Optimizing the 36% Winner with Tri-Layer Fusion + Focal Loss.

Backbone: WavLM-Base-Plus.
Innovation: Tri-Layer Concatenation (Layers 4, 8, 12) = 2304 Features.
Loss: Focal-SupCon (Focusing on hard Egyptian samples).
Regularization: High Dropout (0.5) + SWA.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, get_cosine_schedule_with_warmup
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
# LOSS: FOCAL LOSS (Handling Hard Samples)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

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
# ARCHITECTURE: TRI-LAYER CONCATENATION (4, 8, 12)
# ---------------------------------------------------------------------------
class DeepContextWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # 768 * 3 layers = 2304 features
        self.projector = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Higher dropout for 2304 features
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_labels)
        )

    def forward(self, wav, mask=None, mode="train"):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav, output_hidden_states=True)
        
        # Extract Layers 4, 8, and 12
        # states[0] is embedding, states[1-12] are transformer layers
        h4 = outputs.hidden_states[4]
        h8 = outputs.hidden_states[8]
        h12 = outputs.hidden_states[12]
        
        # Concatenate and Pool
        combined = torch.cat([h4, h8, h12], dim=-1) # (B, Seq, 2304)
        
        if mask is not None:
            down_mask = mask[:, ::320][:, :combined.shape[1]]
            mask_exp = down_mask.unsqueeze(-1).expand(combined.size()).float()
            pooled = torch.sum(combined * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        else:
            pooled = combined.mean(dim=1)
            
        if mode == "contrast": return self.projector(pooled)
        elif mode == "classify": return self.classifier(pooled)
        else: return self.projector(pooled), self.classifier(pooled)

# ---------------------------------------------------------------------------
# DATASET & UTILS
# ---------------------------------------------------------------------------
class StableDataset(Dataset):
    def __init__(self, df, audio_map):
        self.df = df; self.audio_map = audio_map; self.max_len = 160000 
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len: audio = audio[:self.max_len]; mask[:] = 1.0
        else: mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
        return {"wav": torch.tensor(audio, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32), 
                "label": torch.tensor(row["label_id"], dtype=torch.long)}

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), b["mask"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("🏗️ Initializing Deep-Context SOTA (WavLM Tri-Layer + Focal)...")
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

    model = DeepContextWavLM(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.07); l_focal = FocalLoss(alpha=weights, gamma=2.0)
    
    tr_loader = DataLoader(StableDataset(train_df, audio_map), batch_size=32, shuffle=True)
    va_loader = DataLoader(StableDataset(val_df, audio_map), batch_size=32)
    te_loader = DataLoader(StableDataset(test_df, audio_map), batch_size=32)

    # Stage 1: Warmup
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for epoch in range(1, 4):
        model.train(); pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for b in pbar:
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            p, c = model(w, m, mode="both"); loss = 1.0 * l_sup(p, l) + 1.0 * l_focal(c, l)
            opt.zero_grad(); loss.backward(); opt.step()
        va, vf = evaluate(model, va_loader, device); print(f"📈 Warmup {epoch} | Val Acc: {va:.3f}")

    # Stage 2: SOTA Sprint
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 2e-4}, # Slightly lower for stability
    ], weight_decay=0.05)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*30)
    swa_model = torch.optim.swa_utils.AveragedModel(model); swa_start = 12; best_va = 0

    for epoch in range(1, 31):
 acoustic        model.train(); pbar = tqdm(tr_loader, desc=f"Push Ep{epoch}")
        for b in pbar:
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            p, c = model(w, m, mode="both"); loss = 1.0 * l_sup(p, l) + 2.0 * l_focal(c, l) # Focal weight
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        
        if epoch >= swa_start: swa_model.update_parameters(model)
        eval_model = swa_model if epoch >= swa_start else model
        va, vf = evaluate(eval_model, va_loader, device); ta, tf = evaluate(eval_model, te_loader, device)
        print(f"🏆 Ep{epoch} | VAL Acc: {va:.3f} | TEST Acc: {ta:.3f} (F1: {tf:.3f})")
        if va > best_va: best_va = va; torch.save(eval_model.state_dict(), config.CHECKPOINT_DIR / "deep_context_best.pt")

if __name__ == "__main__": main()
