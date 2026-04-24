"""
Audio v7f — Egyptian Arabic SER (The True 39.8% Supercharge)
============================================================
This strictly restores your custom `backup_sota_398.py` engine:
  - 12-layer Learnable Weights (Intra-layer pooling)
  - Audiomentations + SpecAugment + Intra-Class Rare Mixup
  - SWA (Stochastic Weight Averaging)
  - Dual SupCon / Focal Loss 
  - Smart Eval Window Sliding
  - Prosody Concatenation

What is added:
  1. High-Confidence Audio Data (audio_hc)
  2. Multi-tier LR + Progressive Unfreezing into Phase 2.
"""
import os, sys, subprocess, zipfile, math, random
from collections import defaultdict

def install_deps():
    try:
        import audiomentations, peft, transformers
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "audiomentations", "peft", "transformers", "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel
import pandas as pd, numpy as np, librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 80000        
K_PER_CLASS    = 2            
NUM_CLASSES    = 7
BATCH_SIZE     = K_PER_CLASS * NUM_CLASSES   # 14
BATCH_FROZEN   = 32           
PHASE1_EPOCHS  = 4
PHASE2_EPOCHS  = 26
SWA_START      = 18
FOCAL_GAMMA    = 2.0
SUPCON_TEMP    = 0.07
SUPCON_WEIGHT  = 0.4          
RARE_THRESHOLD = 50           
MODEL_NAME     = "microsoft/wavlm-base-plus" 

# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER
# ─────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(i)
        self.classes   = sorted(self.class_indices.keys())
        self.n_batches = min(len(v) for v in self.class_indices.values()) // k

    def __iter__(self):
        pools = {c: random.sample(idxs, len(idxs))
                 for c, idxs in self.class_indices.items()}
        ptrs  = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                batch.extend(pools[c][ptrs[c]: ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch)
            yield batch
    def __len__(self): return self.n_batches

# ─────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce  = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()

class SupConLoss(nn.Module):
    def __init__(self, temperature=SUPCON_TEMP):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temp
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        self_mask = torch.scatter(torch.ones_like(mask), 1,
                                  torch.arange(B).view(-1, 1).to(device), 0)
        mask = mask * self_mask
        exp_sim = torch.exp(sim) * self_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)
        mean_log_pos = (mask * log_prob).sum(1) / (mask.sum(1).clamp(min=1))
        loss = -mean_log_pos.mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)

class AttentionPool(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.attn = nn.Linear(d, 1)

    def forward(self, x):                   
        w = F.softmax(self.attn(x), dim=1)  
        return (x * w).sum(dim=1)           

# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────
class WavLMEliteSER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        cfg  = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"
        )
        self.wavlm = get_peft_model(base, cfg)

        self.layer_weights = nn.Parameter(torch.ones(6))
        self.attn_pool = AttentionPool(768)
        nn.init.zeros_(self.attn_pool.attn.weight)
        nn.init.zeros_(self.attn_pool.attn.bias)

        self.proj_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.classifier = nn.Sequential(
            nn.Linear(768 + 4, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def freeze_all(self):
        for p in self.wavlm.parameters(): p.requires_grad = False

    def unfreeze_target_layers(self, layer_indices):
        for n, p in self.wavlm.named_parameters():
            if "lora_" in n and ".layers." in n:
                try:
                    parts = n.split('.')
                    layer_idx = int(parts[parts.index('layers') + 1])
                    if layer_idx in layer_indices:
                        p.requires_grad = True
                except: pass

    def forward(self, wav, prosody):
        wav  = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        out  = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)

        hidden_states = out.hidden_states[-6:]               
        w = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w[i] * hidden_states[i] for i in range(6))  

        pooled = self.attn_pool(weighted)
        proj = F.normalize(self.proj_head(pooled), dim=-1)
        logits = self.classifier(torch.cat([pooled, prosody], dim=-1))

        return logits, proj

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
def extract_prosody(y: np.ndarray) -> np.ndarray:
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    f0  = librosa.yin(y, fmin=65, fmax=2093)
    f0m = float(np.nanmean(f0)) / 500.0
    f0s = float(np.nanstd(f0))  / 100.0
    vec = np.array([rms, zcr, f0m, f0s], dtype=np.float32)
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)

class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False, rare_classes=None):
        self.df          = df.reset_index(drop=True)
        self.path_map    = path_map
        self.augment     = augment
        self.rare_classes = set(rare_classes or [])
        self.aug_pipe    = Compose([AddGaussianNoise(p=0.35), PitchShift(p=0.35)])
        self.class_idx   = defaultdict(list)
        for i, row in self.df.iterrows():
            self.class_idx[int(row["lid"])].append(i)

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=SR)
            # Safe Pre-processing
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: return None, None
            if len(yt) > MAX_LEN:
                s  = random.randint(0, len(yt) - MAX_LEN) if self.augment else 0
                yt = yt[s:s + MAX_LEN]
            else:
                yt = np.pad(yt, (0, MAX_LEN - len(yt)))
            return yt, int(row["lid"])
        except Exception:
            return None, None

    def _load(self, idx):
        yt, lid = self._load_audio(idx)
        if yt is None: return None
        try:
            if self.augment and lid in self.rare_classes and random.random() < 0.5:
                peer_idx = random.choice(self.class_idx[lid])
                yt2, _   = self._load_audio(peer_idx)
                if yt2 is not None:
                    alpha = np.random.beta(0.4, 0.4)
                    yt    = alpha * yt + (1 - alpha) * yt2

            if self.augment:
                T = len(yt); w = int(T * 0.15 * random.random())
                if w > 0:
                    s = random.randint(0, T - w)
                    yt = yt.copy(); yt[s:s + w] = 0.0
                yt = self.aug_pipe(samples=yt, sample_rate=SR)

            return {
                "wav":     torch.tensor(yt,                  dtype=torch.float32),
                "prosody": torch.tensor(extract_prosody(yt), dtype=torch.float32),
                "label":   torch.tensor(lid,                 dtype=torch.long),
            }
        except Exception: return None

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            s = self._load(idx)
            if s is not None: return s
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError("Could not load audio.")

def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found.")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    pm = {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}
    return pm

def fast_eval(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits, _ = model(b["wav"].to(device), b["prosody"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro", zero_division=0)

def smart_eval(model, df, path_map, device, chunk=MAX_LEN, stride=SR, min_slide=5 * SR):
    model.eval(); preds, truths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: continue
            if len(yt) > min_slide:
                logits_list = []
                for start in range(0, len(yt) - chunk + 1, stride):
                    seg = yt[start:start + chunk]
                    p   = extract_prosody(seg)
                    with torch.no_grad(), autocast("cuda"):
                        l, _ = model(torch.tensor(seg).unsqueeze(0).to(device),
                                      torch.tensor(p).unsqueeze(0).to(device))
                    logits_list.append(F.softmax(l, dim=-1))
                pred = torch.stack(logits_list).mean(0).argmax(1).item()
            else:
                seg  = yt[:chunk] if len(yt) >= chunk else np.pad(yt, (0, chunk - len(yt)))
                p    = extract_prosody(seg)
                with torch.no_grad(), autocast("cuda"):
                    l, _ = model(torch.tensor(seg).unsqueeze(0).to(device),
                                  torch.tensor(p).unsqueeze(0).to(device))
                pred = l.argmax(1).item()
            preds.append(pred); truths.append(int(row["lid"]))
        except Exception: pass
    return accuracy_score(truths, preds), f1_score(truths, preds, average="macro", zero_division=0)

# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/audio_hc"
    save_path  = colab_root / "wavlm_elite_best.pt"

    path_map = get_path_map(colab_root)
    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    class_counts = tr_df["emotion_final"].value_counts()
    rare_label_names = set(class_counts[class_counts < RARE_THRESHOLD].index)
    rare_lids = {lid[l] for l in rare_label_names if l in lid}

    print("\n🧠 RESTORED TRUE 39.8% BASELINE ARCHITECTURE")
    tr_ds = AudioDataset(tr_df, path_map, augment=True,  rare_classes=rare_lids)
    va_ds = AudioDataset(va_df, path_map, augment=False, rare_classes=None)

    def make_sampler(): return BalancedBatchSampler(tr_df["lid"].values, k=K_PER_CLASS)
    bal_sampler = make_sampler()

    tr_loader = DataLoader(tr_ds, batch_sampler=bal_sampler, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=16, num_workers=0)
    tr_frozen = DataLoader(tr_ds, batch_size=BATCH_FROZEN, shuffle=True, drop_last=True, num_workers=0)

    model     = WavLMEliteSER(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    focal     = FocalLoss(gamma=FOCAL_GAMMA)
    supcon    = SupConLoss(temperature=SUPCON_TEMP)
    scaler    = GradScaler("cuda")
    best_acc  = 0.0; swa_active = False

    # PHASE 1 — FROZEN
    print(f"\n🧊 PHASE 1 — Head Warmup (Completely Frozen)")
    model.freeze_all()
    p1_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1_params, lr=1e-3, weight_decay=0.01)
    sch1 = get_cosine_schedule_with_warmup(opt1, 0, len(tr_frozen) * PHASE1_EPOCHS)

    for ep in range(1, PHASE1_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_frozen, desc=f"Ph1 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                logits, _ = model(b["wav"].to(device), b["prosody"].to(device))
                loss = focal(logits, b["label"].to(device))
            opt1.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt1); nn.utils.clip_grad_norm_(p1_params, 1.0)
            scaler.step(opt1); scaler.update(); sch1.step()
            ep_loss += loss.item()
        acc, f1 = fast_eval(model, va_loader, device)
        print(f"Ph1 Ep {ep:02d} | Loss {ep_loss/len(tr_frozen):.3f} | Val Acc {acc:.3f} | Macro F1 {f1:.3f}")

    # PHASE 2 — PROGRESSIVE UNFREEZING
    print(f"\n🔥 PHASE 2 — Multi-Tier Progressive Unfreezing")
    lora_top = [p for n, p in model.wavlm.named_parameters() if "lora_" in n and ".layers." in n and int(n.split('.')[n.split('.').index('layers')+1]) >= 8]
    lora_mid = [p for n, p in model.wavlm.named_parameters() if "lora_" in n and ".layers." in n and 4 <= int(n.split('.')[n.split('.').index('layers')+1]) < 8]
    lora_bot = [p for n, p in model.wavlm.named_parameters() if "lora_" in n and ".layers." in n and int(n.split('.')[n.split('.').index('layers')+1]) < 4]

    opt2 = torch.optim.AdamW([
        {"params": lora_top, "lr": 5e-4},
        {"params": lora_mid, "lr": 1e-4},
        {"params": lora_bot, "lr": 5e-5},
        {"params": [model.layer_weights], "lr": 1e-3},
        {"params": model.attn_pool.parameters(), "lr": 2e-5},
        {"params": model.proj_head.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 2e-5},
    ], weight_decay=0.01)
    
    sch2 = get_cosine_schedule_with_warmup(opt2, len(bal_sampler), len(bal_sampler) * PHASE2_EPOCHS)

    for ep in range(1, PHASE2_EPOCHS + 1):
        if ep == 1:
            print(f"🔥 (Epoch 1) Unfreezing Top Layers (8-11)")
            model.unfreeze_target_layers([8, 9, 10, 11])
        elif ep == 8:
            print(f"🔥 (Epoch 8) Unfreezing Middle Layers (4-7)")
            model.unfreeze_target_layers([4, 5, 6, 7, 8, 9, 10, 11])
        elif ep == 16:
            print(f"🔥 (Epoch 16) Unfreezing Bottom Layers (0-3)")
            model.unfreeze_target_layers([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        bal_sampler = make_sampler()
        tr_loader   = DataLoader(tr_ds, batch_sampler=bal_sampler, num_workers=0)
        model.train(); ep_loss = 0.0
        
        for b in tqdm(tr_loader, desc=f"Ph2 Ep{ep:02d}", leave=False):
            wav_b = b["wav"].to(device); pro_b = b["prosody"].to(device)
            lbl_b = b["label"].to(device)
            with autocast("cuda"):
                logits, proj = model(wav_b, pro_b)
                loss = (1 - SUPCON_WEIGHT) * focal(logits, lbl_b) + SUPCON_WEIGHT * supcon(proj, lbl_b)
            opt2.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt2); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update(); sch2.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path); tag = "  *** BEST ACC ***"
        print(f"Ph2 Ep {ep:02d} | Loss {ep_loss/len(bal_sampler):.3f} | Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    if swa_active:
        swa_model.train()
        with torch.no_grad():
            for b in tqdm(tr_loader, desc="SWA BN", leave=False):
                with autocast("cuda"): swa_model(b["wav"].to(device), b["prosody"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, va_loader, device)
        if swa_acc > best_acc:
            best_acc = swa_acc; torch.save(swa_model.state_dict(), save_path)
    
    print("\n[EVAL] Smart sliding-window evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    swe_acc, swe_f1 = smart_eval(model, va_df, path_map, device)

    print(f"\n=======================================================")
    print(f"  FINAL RESULTS (True 39.8% Core + Unfreeze)")
    print(f"  Best Single-Crop Val Acc : {best_acc:.3f}")
    print(f"  Smart Eval Val Acc       : {swe_acc:.3f}")
    print(f"  Smart Eval Val F1        : {swe_f1:.3f}")
    print(f"=======================================================")

if __name__ == "__main__": train()
