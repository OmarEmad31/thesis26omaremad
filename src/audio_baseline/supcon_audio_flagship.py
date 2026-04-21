"""
HArnESS-SupCon-Density: Egyptian Arabic SER (The 50% Final Push).
Geometric Boundary Learning (SupCon) + High-Density Signal (3s Crops).

Backbone: WavLM-Base-Plus.
Loss: SupCon (Geometric) + CrossEntropy (label_smoothing=0.1).
Sampler: WeightedRandomSampler (F1 Optimization).
Protocol: 3s Random Cropping (Density Multiplication).
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from pathlib import Path

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# LOSS: SUPERVISED CONTRASTIVE LOSS
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
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
        loss = -mean_log_prob_pos.mean()
        return loss

# ---------------------------------------------------------------------------
# ARCHITECTURE: PROJECTOR + CLASSIFIER
# ---------------------------------------------------------------------------
class SupConWavLM(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None, mode="train"):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav).last_hidden_state
        
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
            mask_expanded = down_mask.unsqueeze(-1).expand(outputs.size()).float()
            pooled = torch.sum(outputs * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        else:
            pooled = outputs.mean(dim=1)
            
        if mode == "contrast":
            return self.projector(pooled)
        elif mode == "classify":
            return self.classifier(pooled)
        else:
            return self.projector(pooled), self.classifier(pooled)

# ---------------------------------------------------------------------------
# DATASET: RANDOM 3S CROPPING ENGINE
# ---------------------------------------------------------------------------
class DensitySupConDataset(Dataset):
    def __init__(self, df, audio_map, is_train=False):
        self.df = df; self.audio_map = audio_map; self.crop_len = 48000; self.is_train = is_train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        if len(audio) > self.crop_len:
            if self.is_train:
                start = np.random.randint(0, len(audio) - self.crop_len)
            else:
                start = (len(audio) - self.crop_len) // 2
            audio = audio[start:start+self.crop_len]; mask = np.ones(self.crop_len, dtype=np.float32)
        else:
            mask = np.zeros(self.crop_len, dtype=np.float32)
            mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.crop_len - len(audio)))
            
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
            l = model(b["wav"].to(device), b["mask"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("🏗️ Initializing SupCon-Density Final Engine...")
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

    # WEIGHTED SAMPLER FOR F1 OPTIMIZATION
    y_tr = train_df["label_id"].values
    counts = np.bincount(y_tr)
    weights = 1. / counts
    sample_weights = torch.from_numpy(np.array([weights[t] for t in y_tr]))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    model = SupConWavLM(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.07); l_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    tr_loader = DataLoader(DensitySupConDataset(train_df, audio_map, is_train=True), batch_size=32, sampler=sampler)
    va_loader = DataLoader(DensitySupConDataset(val_df, audio_map), batch_size=32)
    te_loader = DataLoader(DensitySupConDataset(test_df, audio_map), batch_size=32)

    print("\n🔥 STAGE 1: DENSITY GEOMETRIC WARMUP")
    for param in model.wavlm.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 4):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for b in pbar:
            w = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            p, c = model(w, m, mode="both")
            loss = 1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)
            opt.zero_grad(); loss.backward(); opt.step()
        va, vf = evaluate(model, va_loader, device)
        print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: THE 50% DENSITY PUSH (25 Epochs)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-5},
        {"params": model.projector.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*25)
    
    best_va = 0
    for epoch in range(1, 26):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Push Ep{epoch}")
        for b in pbar:
            w = b["wav"].to(device); l = b["label"].to(device); m = b["mask"].to(device)
            p, c = model(w, m, mode="both")
            loss = 1.0 * l_sup(p, l) + 1.1 * l_ce(c, l) # Slight bias to CE for final accuracy
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        va, vf = evaluate(model, va_loader, device)
        ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "supcon_density_best.pt")

if __name__ == "__main__": main()
