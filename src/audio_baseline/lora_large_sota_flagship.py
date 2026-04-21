"""
HArnESS-LoRA-Masterclass: Egyptian Arabic SER (The 50% Milestone Push).
Large-Scale XLS-R (300M) + Low-Rank Adaptation (LoRA).

Backbone: facebook/wav2vec2-large-xlsr-53 (Elite Acoustic capacity).
Technique: LoRA (PEFT) for High-Parameter/Small-Data stability.
Loss: Hybrid SupCon (Geometric) + CrossEntropy.
Target: 50% Individual Audio Milestone.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, Wav2Vec2Model, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    # Manual LoRA placeholder/error if not available in environment
    print("❌ Critical: 'peft' library missing. Run '!pip install peft bitsandbytes'")
    sys.exit(1)

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# LOSS: SUPCON (Geometric Clustering)
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
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# ARCHITECTURE: LARGE XLS-R + LoRA
# ---------------------------------------------------------------------------
class LoRALargeSER(nn.Module):
    def __init__(self, num_labels, model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        # 1. Load the 300M Parameter Giant
        base_model = Wav2Vec2Model.from_pretrained(model_name)
        
        # 2. Configure LoRA: Target the Attention layers (query/value)
        # This allows us to train 300M param model with only ~1.2M trainable params.
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["query_proj", "value_proj"], 
            lora_dropout=0.1, 
            bias="none"
        )
        self.backbone = get_peft_model(base_model, lora_config)
        self.backbone.gradient_checkpointing_enable() # Memory stability
        
        # Large XLS-R hidden dim is 1024
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None, mode="train"):
        # Normalization
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        
        # XLSR-Large hidden states
        outputs = self.backbone(wav).last_hidden_state
        
        # Masked Mean Pooling (Proven stability)
        if mask is not None:
            down_mask = mask[:, ::320][:, :outputs.shape[1]]
            mask_exp = down_mask.unsqueeze(-1).expand(outputs.size()).float()
            pooled = torch.sum(outputs * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        else:
            pooled = outputs.mean(dim=1)
            
        if mode == "contrast": return self.projector(pooled)
        elif mode == "classify": return self.classifier(pooled)
        else: return self.projector(pooled), self.classifier(pooled)

# ---------------------------------------------------------------------------
# DATASET: STABLE 10S
# ---------------------------------------------------------------------------
class StableDataset(Dataset):
    def __init__(self, df, audio_map):
        self.df = df; self.audio_map = audio_map; self.max_len = 160000 

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        mask = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]; mask[:] = 1.0
        else:
            mask[:len(audio)] = 1.0; audio = np.pad(audio, (0, self.max_len - len(audio)))
        return {"wav": torch.tensor(audio, dtype=torch.float32), 
                "mask": torch.tensor(mask, dtype=torch.float32), 
                "label": torch.tensor(row["label_id"], dtype=torch.long)}

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), b["mask"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("🏗️ Initializing Large-Scale LoRA Flagship (XLS-R 300M)...")
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

    # 3. Instantiate model with LoRA
    model = LoRALargeSER(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.07); l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    # 4. Optimizer: High learning rate for LoRA adapters (standard PEFT practice)
    # Even though it is a Large model, we only train 1.2M params, so 1e-4 is safe.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    tr_loader = DataLoader(StableDataset(train_df, audio_map), batch_size=8, shuffle=True) # XLS-R Large is heavy
    va_loader = DataLoader(StableDataset(val_df, audio_map), batch_size=8)
    te_loader = DataLoader(StableDataset(test_df, audio_map), batch_size=8)

    print("\n🚀 STARTING LORA ADAPTATION (Targeting the 50% Milestone)")
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*30)
    
    best_va = 0
    for epoch in range(1, 31):
        model.train(); pbar = tqdm(tr_loader, desc=f"LoRA Ep{epoch}")
        for b in pbar:
            w, m, l = b["wav"].to(device), b["mask"].to(device), b["label"].to(device)
            p, c = model(w, m, mode="both")
            loss = 1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        
        va, vf = evaluate(model, va_loader, device); ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} F1: {vf:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "lora_large_best.pt")

if __name__ == "__main__": main()
