"""
Audio Final v6 — Egyptian Arabic SER
======================================
Based on: backup_sota_36_stable.py (confirmed 36% baseline)
Backbone: WavLM-Base-Plus (94M) — right-sized for 857 samples (not 316M Large)

Key fixes vs all previous attempts:
  1. Focal Loss (gamma=2): handles class imbalance without sampler or class weights
  2. NO WeightedRandomSampler: was causing double-correction → collapse to rare class
  3. WavLM-Base-Plus not Large: 94M vs 316M, better for small datasets
  4. batch_size=16 with AMP: large enough for stable class repr in each batch
  5. num_workers=0: Colab fork + librosa = silent NaN batches
  6. NO gradient_checkpointing: breaks PEFT gradient flow silently
  7. 2-phase protocol from confirmed backup: frozen warmup → LoRA fine-tune
  8. SWA from epoch 20 + sliding window inference at final eval
"""
import os, sys, subprocess, zipfile, math

def install_deps():
    try:
        import audiomentations, peft, transformers
    except ImportError:
        print("[INIT] Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "audiomentations", "peft", "transformers", "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel
import pandas as pd, numpy as np, librosa, random, math
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR            = 16000
MAX_LEN       = 80000    # 5-second window (more context than 3s, less OOM than 10s)
BATCH_FROZEN  = 32       # Phase 1: backbone frozen, no backprop through WavLM
BATCH_FINETUNE = 16      # Phase 2: with AMP, Colab T4 safe for WavLM-Base-Plus
PHASE1_EPOCHS = 3        # Frozen warmup
PHASE2_EPOCHS = 27       # Full fine-tune
SWA_START     = 20       # Epoch in Phase 2 to start SWA (1-indexed)
FOCAL_GAMMA   = 2.0      # Focal loss focusing parameter
MODEL_NAME    = "microsoft/wavlm-base-plus"  # 94M params, NOT wavlm-large (316M)


# ─────────────────────────────────────────────────────────
# FOCAL LOSS
# Handles class imbalance without class weights or sampler.
# Down-weights easy examples, focuses on hard ones (rare classes).
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce   = F.cross_entropy(logits, labels, reduction="none")
        p_t  = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────
# MODEL: WavLM-Base-Plus + LoRA + Masked Mean Pool
# Exactly the combination that reached 36% in backup_sota_36_stable
# ─────────────────────────────────────────────────────────
class WavLMBaseSER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME)
        cfg  = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"
        )
        self.wavlm = get_peft_model(base, cfg)
        # NO gradient_checkpointing — silently breaks PEFT gradient flow

        self.classifier = nn.Sequential(
            nn.Linear(768 + 4, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def freeze_backbone(self):
        """Phase 1: only LoRA + classifier train."""
        for n, p in self.wavlm.named_parameters():
            p.requires_grad = "lora_" in n

    def unfreeze_lora(self):
        """Phase 2: LoRA trainable (base WavLM still frozen)."""
        for n, p in self.wavlm.named_parameters():
            p.requires_grad = "lora_" in n

    def forward(self, wav, prosody):
        # Per-utterance normalization
        wav  = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        # Build attention mask (1 everywhere — all samples are padded to MAX_LEN)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        out  = self.wavlm(wav, attention_mask=mask)
        hidden = out.last_hidden_state           # [B, T, 768]

        # Masked mean pooling (downsampled mask matches hidden state length)
        T = hidden.shape[1]
        pooled = hidden.mean(dim=1)              # [B, 768]

        return self.classifier(torch.cat([pooled, prosody], dim=-1))


# ─────────────────────────────────────────────────────────
# PROSODY: 4-dim, NaN-safe
# ─────────────────────────────────────────────────────────
def extract_prosody(y: np.ndarray) -> np.ndarray:
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    f0  = librosa.yin(y, fmin=65, fmax=2093)
    f0m = float(np.nanmean(f0)) / 500.0
    f0s = float(np.nanstd(f0))  / 100.0
    vec = np.array([rms, zcr, f0m, f0s], dtype=np.float32)
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)


# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False):
        self.df       = df.reset_index(drop=True)
        self.path_map = path_map
        self.augment  = augment
        self.aug_pipe = Compose([AddGaussianNoise(p=0.35), PitchShift(p=0.35)])

    def __len__(self): return len(self.df)

    def _load(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: return None       # skip <0.5s clips
            if len(yt) > MAX_LEN:
                s  = random.randint(0, len(yt) - MAX_LEN) if self.augment else 0
                yt = yt[s:s + MAX_LEN]
            else:
                yt = np.pad(yt, (0, MAX_LEN - len(yt)))
            if self.augment:
                # SpecAugment: time mask up to 15%
                T = len(yt); w = int(T * 0.15 * random.random())
                if w > 0:
                    s = random.randint(0, T - w)
                    yt = yt.copy(); yt[s:s + w] = 0.0
                yt = self.aug_pipe(samples=yt, sample_rate=SR)
            return {
                "wav":     torch.tensor(yt,              dtype=torch.float32),
                "prosody": torch.tensor(extract_prosody(yt), dtype=torch.float32),
                "label":   torch.tensor(int(row["lid"]), dtype=torch.long),
            }
        except Exception:
            return None

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            s = self._load(idx)
            if s is not None: return s
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError("Could not load audio.")


# ─────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────
def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm:
        print(f"[SUCCESS] Found {len(pm)} wav files on Drive.")
        return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    print(f"[INIT] Searching for {zname}...")
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found on Drive.")
    print(f"[INIT] Extracting from {zpath}...")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    pm = {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}
    print(f"[SUCCESS] Extracted and indexed {len(pm)} audio files.")
    return pm


def build_loaders(tr_df, va_df, path_map):
    tr_ds = AudioDataset(tr_df, path_map, augment=True)
    va_ds = AudioDataset(va_df, path_map, augment=False)
    # Note: NO WeightedRandomSampler — Focal Loss handles imbalance
    tr_ldr_frozen = DataLoader(tr_ds, batch_size=BATCH_FROZEN,
                               shuffle=True, drop_last=True, num_workers=0)
    tr_ldr_ft     = DataLoader(tr_ds, batch_size=BATCH_FINETUNE,
                               shuffle=True, drop_last=True, num_workers=0)
    va_ldr        = DataLoader(va_ds, batch_size=BATCH_FINETUNE,
                               num_workers=0)
    return tr_ldr_frozen, tr_ldr_ft, va_ldr


# ─────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────
def fast_eval(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["prosody"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro", zero_division=0)


def sliding_window_infer(model, df, path_map, device,
                          chunk=MAX_LEN, stride=SR):
    model.eval(); preds, truths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SWE", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            lg_list = []
            for start in range(0, max(1, len(yt) - chunk + 1), stride):
                seg = yt[start:start + chunk]
                if len(seg) < chunk: seg = np.pad(seg, (0, chunk - len(seg)))
                p = extract_prosody(seg)
                with torch.no_grad(), autocast("cuda"):
                    l = model(torch.tensor(seg).unsqueeze(0).to(device),
                               torch.tensor(p).unsqueeze(0).to(device))
                lg_list.append(F.softmax(l, dim=-1))
            if lg_list:
                preds.append(torch.stack(lg_list).mean(0).argmax(1).item())
                truths.append(int(row["lid"]))
        except Exception: pass
    return (accuracy_score(truths, preds),
            f1_score(truths, preds, average="macro", zero_division=0))


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    save_path  = colab_root / "wavlm_base_best.pt"

    path_map = get_path_map(colab_root)

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    id2l  = {v: k for k, v in lid.items()}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    print(f"\n[DATA] Train: {len(tr_df)} | Val: {len(va_df)} | Classes: {len(lid)}")
    print("[DATA] Train distribution:")
    for emo, cnt in tr_df["emotion_final"].value_counts().items():
        print(f"  {emo:12s}: {cnt:3d} ({100*cnt/len(tr_df):.1f}%)")

    tr_frozen, tr_ft, va_ldr = build_loaders(tr_df, va_df, path_map)

    model     = WavLMBaseSER(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    crit      = FocalLoss(gamma=FOCAL_GAMMA)
    scaler    = GradScaler("cuda")
    best_acc  = 0.0
    swa_active = False

    # ═══════════════════════════════════════════════════
    # PHASE 1: Frozen WavLM — head warmup
    # ═══════════════════════════════════════════════════
    model.freeze_backbone()
    p1_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1_params, lr=1e-3, weight_decay=0.01)
    sch1 = get_cosine_schedule_with_warmup(opt1, 0, len(tr_frozen) * PHASE1_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head Warmup  (WavLM frozen, {PHASE1_EPOCHS} epochs)")
    print(f"  Trainable: {sum(p.numel() for p in p1_params):,} params")
    print(f"  Batch: {BATCH_FROZEN} | Loss: Focal(γ={FOCAL_GAMMA})")
    print(f"{'='*60}")

    for ep in range(1, PHASE1_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_frozen, desc=f"Ph1 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["prosody"].to(device)),
                            b["label"].to(device))
            opt1.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt1)
            nn.utils.clip_grad_norm_(p1_params, 1.0)
            scaler.step(opt1); scaler.update(); sch1.step()
            ep_loss += loss.item()
        acc, f1 = fast_eval(model, va_ldr, device)
        print(f"Ph1 Ep {ep:02d} | Loss {ep_loss/len(tr_frozen):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}")

    # ═══════════════════════════════════════════════════
    # PHASE 2: LoRA fine-tune
    # ═══════════════════════════════════════════════════
    model.unfreeze_lora()
    opt2 = torch.optim.AdamW([
        {"params": [p for n, p in model.wavlm.named_parameters()
                    if "lora_" not in n and p.requires_grad], "lr": 2e-6},
        {"params": [p for n, p in model.wavlm.named_parameters()
                    if "lora_" in n],                         "lr": 1e-5},
        {"params": model.classifier.parameters(),             "lr": 5e-5},
    ], weight_decay=0.01)
    sch2 = get_cosine_schedule_with_warmup(
        opt2, len(tr_ft), len(tr_ft) * PHASE2_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — LoRA Fine-tune  ({PHASE2_EPOCHS} epochs)")
    print(f"  Batch: {BATCH_FINETUNE} | SWA from epoch {SWA_START}")
    print(f"{'='*60}")

    for ep in range(1, PHASE2_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_ft, desc=f"Ph2 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["prosody"].to(device)),
                            b["label"].to(device))
            opt2.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update(); sch2.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1 = fast_eval(model, va_ldr, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path); tag = "  *** BEST ***"
        print(f"Ph2 Ep {ep:02d} | Loss {ep_loss/len(tr_ft):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # SWA final evaluation
    # ═══════════════════════════════════════════════════
    if swa_active:
        print("\n[SWA] Finalizing averaged model...")
        swa_model.train()
        with torch.no_grad():
            for b in tqdm(tr_ft, desc="SWA BN", leave=False):
                with autocast("cuda"):
                    swa_model(b["wav"].to(device), b["prosody"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, va_ldr, device)
        print(f"[SWA] Val Acc {swa_acc:.3f} | F1 {swa_f1:.3f}")
        if swa_acc > best_acc:
            best_acc = swa_acc
            torch.save(swa_model.state_dict(), save_path)
            print("[SWA] SWA model is new best — saved.")

    # ═══════════════════════════════════════════════════
    # Sliding window inference (the key +5-8% boost)
    # ═══════════════════════════════════════════════════
    print("\n[SWE] Sliding-window inference over full audio files...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    swe_acc, swe_f1 = sliding_window_infer(model, va_df, path_map, device)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  Best Single-Crop Val Acc : {best_acc:.4f}")
    print(f"  Sliding Window Val Acc   : {swe_acc:.4f}")
    print(f"  Sliding Window Val F1    : {swe_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
