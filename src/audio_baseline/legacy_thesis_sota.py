"""
Legacy Thesis SOTA — The Blueprint Reconstruction
============================================================
This script forces an exact mathematical replication of the proven thesis architecture
to natively bypass the Zero-Detection poison trap.

Core Implementations:
1. Backbone: Wav2Vec2 (12-layer) instead of WavLM.
2. Sequence: 10-second windows (160,000 samples) instead of 5-second crops.
3. Projections: 768 to 512 dimensions explicitly.
4. Protection Mechanism: Strict Frame-Level Masked Mean Pooling. 
   - Zeros from padded silent spaces physically CANNOT dilute the emotional concentration.
5. Optimization: LoRA Rank-16 specifically locked on q_proj & v_proj. 
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
from transformers import Wav2Vec2Model, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

# ─────────────────────────────────────────────────────────
# CONFIG EXACTLY MAPPED TO THESIS
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 160000       # Exactly 10 seconds as specified
NUM_CLASSES    = 7
K_PER_CLASS    = 2
BATCH_SIZE     = 14           
EPOCHS         = 30
SWA_START      = 20
FOCAL_GAMMA    = 2.0
RARE_THRESHOLD = 50           
MODEL_NAME     = "facebook/wav2vec2-base" # The proven 12-layer Wav2Vec2

# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER
# ─────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, k=K_PER_CLASS):
        self.k = k
        self.class_indices = defaultdict(list)
        for i, lbl in enumerate(labels): self.class_indices[int(lbl)].append(i)
        self.classes   = sorted(self.class_indices.keys())
        self.n_batches = min(len(v) for v in self.class_indices.values()) // k

    def __iter__(self):
        pools = {c: random.sample(idxs, len(idxs)) for c, idxs in self.class_indices.items()}
        ptrs  = {c: 0 for c in self.classes}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                batch.extend(pools[c][ptrs[c]: ptrs[c] + self.k])
                ptrs[c] += self.k
            random.shuffle(batch)
            yield batch
    def __len__(self): return self.n_batches

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce  = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()

# ─────────────────────────────────────────────────────────
# THE BLUEPRINT MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────
class ThesisLegacyArchitecture(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # "12 transformer layers process these frame-level features"
        base = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        
        # "The same LoRA configuration (rank 16 on query/value projections)"
        cfg  = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1, bias="none"
        )
        self.wav2vec2 = get_peft_model(base, cfg)

        # "We project these to 512 dimensions..."
        self.projection = nn.Linear(768, 512)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(self, wav, mode="classify"):
        # 1. First, detect the EXACT location of the padded zeros before any manipulation
        raw_mask = (wav != 0.0).long()
        
        # 2. We explicitly REMOVED the catastrophic naive normalization here.
        # Librosa already loads audio bound between [-1, 1]. The naive normalization 
        # was turning the 0.0 padded silences into non-zero static noise, 
        # destroying both the audio and the mask simultaneously.
        
        out = self.wav2vec2(wav, attention_mask=raw_mask)
        hidden_states = out.last_hidden_state  # 768-D sequence
        
        # Calculate exactly how many frames CNN extracted for the non-silent audio
        input_lengths = raw_mask.sum(dim=-1)
        feat_lengths = self.wav2vec2.base_model.model._get_feat_extract_output_lengths(input_lengths)
        
        B, T, _ = hidden_states.shape
        
        # Build strict Frame-Level attention block
        # "The masking is important: it ensures that zero padded regions don’t contribute to the mean"
        frame_mask = torch.arange(T, device=wav.device)[None, :] < feat_lengths[:, None]
        frame_mask = frame_mask.float().unsqueeze(-1)  # [B, T, 1]

        # Project 768 -> 512
        projected = self.projection(hidden_states)     # [B, T, 512]
        
        # Mean Pooling strictly blocked by Frame Mask
        masked_projected = projected * frame_mask
        pooled_sum = masked_projected.sum(dim=1)
        pooled_div = frame_mask.sum(dim=1).clamp(min=1e-5)
        
        pooled = pooled_sum / pooled_div               # [B, 512]
        logits = self.classifier(pooled)
        
        if mode == "classify": return logits
        return logits, pooled

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class LegacyDataset(Dataset):
    def __init__(self, df, path_map, augment=False, rare_classes=None):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.augment = augment
        self.rare_classes = set(rare_classes or [])
        self.aug_pipe = Compose([AddGaussianNoise(p=0.4), PitchShift(p=0.4)])
        self.class_idx = defaultdict(list)
        for i, row in self.df.iterrows(): self.class_idx[int(row["lid"])].append(i)

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=SR)
            # Safe Pre-processing
            yt, _ = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR//2: return None, None
            
            # 10 SECOND WINDOW (Strict Center Cropping or PADDING with exact 0.0s)
            if len(yt) > MAX_LEN:
                s = random.randint(0, len(yt) - MAX_LEN) if self.augment else (len(yt) - MAX_LEN)//2
                yt = yt[s:s + MAX_LEN]
            else:
                # Essential logic: padded explicitly with ZERO to trigger frame-mask
                yt = np.pad(yt, (0, MAX_LEN - len(yt)), constant_values=0.0)
            return yt, int(row["lid"])
        except Exception: return None, None

    def _load(self, idx):
        yt, lid = self._load_audio(idx)
        if yt is None: return None
        try:
            if self.augment and lid in self.rare_classes and random.random() < 0.5:
                # CutMix inside the valid frame region only
                peer_idx = random.choice(self.class_idx[lid])
                yt2, _   = self._load_audio(peer_idx)
                if yt2 is not None:
                    alpha = np.random.beta(0.4, 0.4)
                    yt    = alpha * yt + (1 - alpha) * yt2

            if self.augment: yt = self.aug_pipe(samples=yt.copy(), sample_rate=SR)
            return {"wav": torch.tensor(yt, dtype=torch.float32), "label": torch.tensor(lid, dtype=torch.long)}
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
    zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if "Thesis_Audio_Full.zip" in files: zpath = os.path.join(root, "Thesis_Audio_Full.zip"); break
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    return {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}

def fast_eval(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro", zero_division=0)

# ─────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/audio_hc"
    save_path  = colab_root / "legacy_sota_best.pt"

    path_map = get_path_map(colab_root)
    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    class_counts = tr_df["emotion_final"].value_counts()
    rare_lids = {lid[l] for l in class_counts[class_counts < RARE_THRESHOLD].index if l in lid}

    print("\n🧠 INITIALIZING LEGACY RECONSTRUCTION ALGORITHM (10s | Rank-16 | Masked Pool)")
    tr_ds = LegacyDataset(tr_df, path_map, augment=True, rare_classes=rare_lids)
    va_ds = LegacyDataset(va_df, path_map, augment=False)

    def make_sampler(): return BalancedBatchSampler(tr_df["lid"].values, k=K_PER_CLASS)
    val_loader = DataLoader(va_ds, batch_size=16, num_workers=0)

    model     = ThesisLegacyArchitecture(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    focal     = FocalLoss(gamma=FOCAL_GAMMA)
    scaler    = GradScaler("cuda")
    best_acc  = 0.0; swa_active = False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, len(make_sampler())*2, len(make_sampler())*EPOCHS)

    for ep in range(1, EPOCHS + 1):
        tr_loader = DataLoader(tr_ds, batch_sampler=make_sampler(), num_workers=0)
        model.train(); ep_loss = 0.0
        
        for b in tqdm(tr_loader, desc=f"Legacy Ep{ep:02d}", leave=False):
            w = b["wav"].to(device); l = b["label"].to(device)
            with autocast("cuda"):
                logits = model(w)
                loss = focal(logits, l)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1 = fast_eval(model, val_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path); tag = "  *** BEST ACC ***"
        
        print(f"Ep {ep:02d} | Loss {ep_loss/len(tr_loader):.3f} | Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    if swa_active:
        swa_model.train()
        with torch.no_grad():
            for b in tqdm(tr_loader, desc="SWA BN", leave=False):
                with autocast("cuda"): swa_model(b["wav"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, val_loader, device)
        if swa_acc > best_acc:
            best_acc = swa_acc; torch.save(swa_model.state_dict(), save_path)
            print(f">>> SWA Upgrade Captured! Val Acc {best_acc:.3f}")

    print(f"\n=======================================================")
    print(f"  FINAL LEGACY RECONSTRUCTION RESULTS")
    print(f"  Best Single-Crop Val Acc : {best_acc:.3f}")
    if swa_active: print(f"  SWA Final Valid F1       : {swa_f1:.3f}")
    print(f"=======================================================")

if __name__ == "__main__": train()
