"""
ConformerTitan FINAL — Egyptian Arabic SER
==========================================
Root cause of all previous failures: class imbalance (Anger=28%)
with no class weights → model predicts Anger for everything → stuck 25.7%.

Fixes:
  1. WeightedRandomSampler: balanced batches every step
  2. InverseFrequency class weights on loss from epoch 1
  3. LoRA on q/k/v/out_proj: full attention adaptation
  4. Weighted layer pooling: last 12 WavLM hidden states
  5. AttentionPool: learned frame aggregation vs dumb mean
  6. SWA from epoch 22, Sliding Window at final eval
  7. num_workers=0, batch=4 (Colab-safe config)
  8. NO gradient_checkpointing (breaks PEFT gradients)
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel
import pandas as pd, numpy as np, librosa, random, math
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

SR         = 16000
MAX_LEN    = 48000   # 3 seconds
BATCH_SIZE = 4
NUM_EPOCHS = 35
SWA_START  = 22
NUM_LABELS = 7


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────
# Attention Pooling (learnable frame aggregation)
# ─────────────────────────────────────────────
class AttentionPool(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.attn = nn.Linear(d, 1)

    def forward(self, x):                  # x: [B, T, D]
        w = F.softmax(self.attn(x), dim=1) # [B, T, 1]
        return (x * w).sum(dim=1)          # [B, D]


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class ConformerTitan(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, model_name="microsoft/wavlm-large"):
        super().__init__()

        # WavLM-Large + LoRA on ALL attention projections
        base = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        cfg  = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05, bias="none"
        )
        self.wavlm = get_peft_model(base, cfg)
        # NO gradient_checkpointing

        # Learnable per-layer weights (last 12 layers)
        self.layer_weights = nn.Parameter(torch.ones(12))

        self.pos_enc  = PositionalEncoding(1024, max_len=512)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True)
        self.conformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.pool = AttentionPool(1024)

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 4, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, prosody):
        wav  = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        out  = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)

        # Weighted sum of last 12 hidden layers
        hidden_states = out.hidden_states[-12:]          # 12 x [B, T, 1024]
        w = F.softmax(self.layer_weights, dim=0)
        hidden = sum(w[i] * hidden_states[i] for i in range(12)) # [B, T, 1024]

        hidden = self.pos_enc(hidden)
        ctx    = self.conformer(hidden)   # [B, T, 1024]
        pooled = self.pool(ctx)           # [B, 1024]
        return self.classifier(torch.cat([pooled, prosody], dim=-1))


# ─────────────────────────────────────────────
# Prosody (4-dim, NaN-safe)
# ─────────────────────────────────────────────
def extract_prosody(y: np.ndarray) -> np.ndarray:
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    f0  = librosa.yin(y, fmin=65, fmax=2093)
    f0m = float(np.nanmean(f0)) / 500.0
    f0s = float(np.nanstd(f0))  / 100.0
    vec = np.array([rms, zcr, f0m, f0s], dtype=np.float32)
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False):
        self.df       = df.reset_index(drop=True)
        self.path_map = path_map
        self.augment  = augment
        self.aug_pipe = Compose([AddGaussianNoise(p=0.4), PitchShift(p=0.4)])

    def __len__(self): return len(self.df)

    def _load(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: return None           # discard <0.5s clips
            if len(yt) > MAX_LEN:
                s  = random.randint(0, len(yt) - MAX_LEN) if self.augment else 0
                yt = yt[s:s + MAX_LEN]
            else:
                yt = np.pad(yt, (0, MAX_LEN - len(yt)))
            if self.augment:
                # SpecAugment: time masking
                T = len(yt); width = int(T * 0.15 * random.random())
                if width > 0:
                    s = random.randint(0, T - width)
                    yt = yt.copy(); yt[s:s + width] = 0.0
                yt = self.aug_pipe(samples=yt, sample_rate=SR)
            prosody = extract_prosody(yt)
            return {
                "wav":     torch.tensor(yt, dtype=torch.float32),
                "prosody": torch.tensor(prosody, dtype=torch.float32),
                "label":   torch.tensor(int(row["lid"]), dtype=torch.long),
            }
        except Exception:
            return None

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            s = self._load(idx)
            if s is not None: return s
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError("Could not load any audio.")


# ─────────────────────────────────────────────
# Path map (extracts zip if needed)
# ─────────────────────────────────────────────
def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm:
        print(f"[SUCCESS] Found {len(pm)} wav files on Drive.")
        return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    print(f"[INIT] Scanning for {zname}...")
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found on Drive.")
    print(f"[INIT] Extracting from {zpath}...")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    pm = {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}
    print(f"[SUCCESS] Extracted and indexed {len(pm)} audio files.")
    return pm


# ─────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────
def fast_eval(model, loader, device, label_names=None):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["prosody"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    acc = accuracy_score(ts, ps)
    f1  = f1_score(ts, ps, average="macro", zero_division=0)
    return acc, f1, ps, ts


def sliding_window_infer(model, df, path_map, device, chunk=MAX_LEN, stride=SR):
    model.eval(); preds, truths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SWE", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            logits_list = []
            positions = range(0, max(1, len(yt) - chunk + 1), stride)
            for start in positions:
                seg = yt[start:start + chunk]
                if len(seg) < chunk: seg = np.pad(seg, (0, chunk - len(seg)))
                p = extract_prosody(seg)
                with torch.no_grad(), autocast("cuda"):
                    l = model(torch.tensor(seg).unsqueeze(0).to(device),
                               torch.tensor(p).unsqueeze(0).to(device))
                logits_list.append(F.softmax(l, dim=-1))
            if logits_list:
                preds.append(torch.stack(logits_list).mean(0).argmax(1).item())
                truths.append(int(row["lid"]))
        except Exception: pass
    acc = accuracy_score(truths, preds)
    f1  = f1_score(truths, preds, average="macro", zero_division=0)
    return acc, f1


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    save_path  = colab_root / "conformer_titan_best.pt"

    # ── Data ──────────────────────────────────
    path_map = get_path_map(colab_root)

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    id2lbl = {v: k for k, v in lid.items()}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    print(f"\n[DATA] Train: {len(tr_df)} | Val: {len(va_df)} | Classes: {len(lid)}")
    print("[DATA] Train distribution:")
    for l, c in tr_df["emotion_final"].value_counts().items():
        print(f"  {l:12s}: {c:3d} ({100*c/len(tr_df):.1f}%)")

    # ── Inverse-frequency class weights ───────
    counts  = tr_df["lid"].value_counts().sort_index().values.astype(float)
    w_inv   = 1.0 / counts
    w_inv  /= w_inv.sum()                          # normalize
    class_weights = torch.tensor(w_inv * len(counts), dtype=torch.float32).to(device)

    # ── WeightedRandomSampler (balanced batches) ──
    sample_weights = torch.tensor(
        [w_inv[lid_val] for lid_val in tr_df["lid"].values], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_df), replacement=True)

    tr_loader = DataLoader(
        AudioDataset(tr_df, path_map, augment=True),
        batch_size=BATCH_SIZE, sampler=sampler,
        drop_last=True, num_workers=0
    )
    va_loader = DataLoader(
        AudioDataset(va_df, path_map, augment=False),
        batch_size=BATCH_SIZE, num_workers=0
    )

    print(f"[DATA] {len(tr_loader)} batches/epoch with balanced sampling")

    # ── Model ─────────────────────────────────
    model     = ConformerTitan(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)

    # Discriminative LRs
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(),      "lr": 2e-6},
        {"params": model.layer_weights,            "lr": 1e-3},
        {"params": model.pos_enc.parameters(),    "lr": 5e-5},
        {"params": model.conformer.parameters(),  "lr": 5e-5},
        {"params": model.pool.parameters(),       "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=0.01)

    total_steps  = len(tr_loader) * NUM_EPOCHS
    warmup_steps = len(tr_loader) * 1              # 1 epoch warmup only
    sch    = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = GradScaler("cuda")
    crit   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_acc   = 0.0
    swa_active = False

    print(f"\n{'='*60}")
    print(f"  CONFORMER TITAN FINAL  —  {NUM_EPOCHS} epochs")
    print(f"  Weighted sampler + inverse-freq loss from epoch 1")
    print(f"  SWA from epoch {SWA_START}")
    print(f"{'='*60}\n")

    for ep in range(1, NUM_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ep {ep:02d}", leave=False):
            wav_b = b["wav"].to(device)
            pro_b = b["prosody"].to(device)
            lbl_b = b["label"].to(device)
            with autocast("cuda"):
                loss = crit(model(wav_b, pro_b), lbl_b)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1, ps, ts = fast_eval(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path); tag = "  *** BEST ***"

        print(f"Ep {ep:02d} | Loss {ep_loss/len(tr_loader):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

        # Per-class breakdown every 5 epochs
        if ep % 5 == 0:
            from sklearn.metrics import confusion_matrix
            print(f"  Per-class: { {id2lbl[i]: int(sum(np.array(ps)==i and np.array(ts)==i)) for i in range(len(lid))} }")

    # ── SWA ───────────────────────────────────
    if swa_active:
        print("\n[SWA] Finalizing averaged model...")
        swa_model.train()
        with torch.no_grad():
            for b in tqdm(tr_loader, desc="SWA BN", leave=False):
                with autocast("cuda"):
                    swa_model(b["wav"].to(device), b["prosody"].to(device))
        swa_acc, swa_f1, _, _ = fast_eval(swa_model, va_loader, device)
        print(f"[SWA] Val Acc {swa_acc:.3f} | F1 {swa_f1:.3f}")
        if swa_acc > best_acc:
            best_acc = swa_acc
            torch.save(swa_model.state_dict(), save_path)
            print("[SWA] SWA model is the new best — saved.")

    # ── Sliding Window Final Eval ──────────────
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
