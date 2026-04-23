"""
ConformerTitan v4 — End-to-End Training (No Freeze)
------------------------------------------------------
Lesson learned: freezing WavLM in Phase 1 starves the Conformer of
emotion-specific features. LoRA must be active from epoch 1.

Strategy:
  - LoRA active from epoch 1 with very low backbone LR (2e-6)
  - Conformer + classifier train faster (5e-5 / 2e-4)
  - No class weights early on (destabilizes training)
  - Sqrt-class weights applied after epoch 10 via LR scheduler swap
  - SWA from epoch 20 onward
  - Sliding window inference at final evaluation
  - All v3 preprocessing upgrades kept (16-dim prosody, SpecAugment)
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
import pandas as pd, numpy as np, librosa, random
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

SR          = 16000
MAX_LEN     = 48000      # 3-second window
PROSODY_DIM = 16         # 13 MFCC means + F0 mean/std + voiced fraction
BATCH_SIZE  = 6
NUM_EPOCHS  = 30
SWA_START   = 20         # Start SWA after epoch 20


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class ConformerTitan(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-large"):
        super().__init__()
        base = WavLMModel.from_pretrained(model_name)
        cfg  = LoraConfig(r=16, lora_alpha=32,
                          target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)
        # NO gradient_checkpointing — incompatible with PEFT+Conformer

        self.pos_enc  = PositionalEncoding(1024, max_len=512, dropout=0.1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True)
        self.conformer = nn.TransformerEncoder(enc_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(1024 + PROSODY_DIM, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, prosody):
        wav = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask   = torch.ones(wav.shape[:2], device=wav.device)
        hidden = self.wavlm(wav, attention_mask=mask).last_hidden_state
        hidden = self.pos_enc(hidden)
        ctx    = self.conformer(hidden)
        pooled = ctx.mean(1)
        return self.classifier(torch.cat([pooled, prosody], dim=-1))


def extract_prosody(y: np.ndarray) -> np.ndarray:
    mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13).mean(axis=1) / 100.0
    f0       = librosa.yin(y, fmin=65, fmax=2093)
    voiced   = f0[f0 > 0]
    f0_mean  = float(np.mean(voiced)) / 500.0 if len(voiced) else 0.0
    f0_std   = float(np.std(voiced))  / 100.0 if len(voiced) else 0.0
    v_frac   = len(voiced) / max(len(f0), 1)
    return np.array([*mfcc, f0_mean, f0_std, v_frac], dtype=np.float32)


def time_mask(wav: np.ndarray, max_pct: float = 0.15) -> np.ndarray:
    T = len(wav); width = int(T * max_pct * random.random())
    if width == 0: return wav
    start = random.randint(0, T - width)
    wav = wav.copy(); wav[start:start + width] = 0.0
    return wav


class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False):
        self.df       = df.reset_index(drop=True)
        self.path_map = path_map
        self.augment  = augment
        self.aug_pipe = Compose([
            AddGaussianNoise(p=0.35),
            PitchShift(p=0.35),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.25),
        ])

    def __len__(self): return len(self.df)

    def _load(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None
        try:
            wav, _ = librosa.load(self.path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=25)
            if len(yt) > MAX_LEN:
                start = random.randint(0, len(yt) - MAX_LEN) if self.augment else 0
                yt = yt[start:start + MAX_LEN]
            else:
                yt = np.pad(yt, (0, MAX_LEN - len(yt)))
            if self.augment:
                yt = time_mask(yt)
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


def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found on Drive.")
    print(f"[INIT] Extracting {zname}...")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    return {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}


def fast_eval(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["prosody"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")


def sliding_window_infer(model, df, path_map, device, chunk=MAX_LEN, stride=16000):
    model.eval(); preds, truths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SWE", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=25)
            logits_list = []
            for start in range(0, max(1, len(yt) - chunk + 1), stride):
                seg = yt[start:start + chunk]
                if len(seg) < chunk: seg = np.pad(seg, (0, chunk - len(seg)))
                p   = extract_prosody(seg)
                w_t = torch.tensor(seg).unsqueeze(0).to(device)
                p_t = torch.tensor(p).unsqueeze(0).to(device)
                with torch.no_grad(), autocast("cuda"):
                    logits_list.append(model(w_t, p_t))
            if logits_list:
                avg = torch.stack(logits_list).mean(0)
                preds.append(avg.argmax(1).item()); truths.append(int(row["lid"]))
        except Exception: pass
    return accuracy_score(truths, preds), f1_score(truths, preds, average="macro")


def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    save_path  = colab_root / "conformer_titan_best.pt"

    path_map = get_path_map(colab_root)
    print(f"[SUCCESS] Indexed {len(path_map)} audio files.")

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)
    print(f"[DATA] Train: {len(tr_df)} | Val: {len(va_df)} | Classes: {len(lid)}")

    tr_loader = DataLoader(AudioDataset(tr_df, path_map, augment=True),
                           batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(AudioDataset(va_df, path_map, augment=False),
                           batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    model     = ConformerTitan(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)

    # Discriminative learning rates — LoRA active from epoch 1
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(),      "lr": 2e-6},   # tiny — preserve pretrained knowledge
        {"params": model.pos_enc.parameters(),    "lr": 5e-5},
        {"params": model.conformer.parameters(),  "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 2e-4},
    ], weight_decay=0.01)

    sch    = get_cosine_schedule_with_warmup(opt, len(tr_loader) * 2, len(tr_loader) * NUM_EPOCHS)
    scaler = GradScaler("cuda")
    crit   = nn.CrossEntropyLoss(label_smoothing=0.1)   # plain CE — no class weights yet

    # Sqrt class weights for later epochs
    counts  = tr_df["lid"].value_counts().sort_index().values.astype(float)
    w_sqrt  = torch.tensor(1.0 / np.sqrt(counts), dtype=torch.float32).to(device)
    w_sqrt  = w_sqrt / w_sqrt.sum() * len(counts)
    crit_w  = nn.CrossEntropyLoss(weight=w_sqrt, label_smoothing=0.05)

    best_acc   = 0.0
    swa_active = False

    print(f"\n{'='*55}")
    print(f"  CONFORMER TITAN v4  —  {NUM_EPOCHS} epochs end-to-end")
    print(f"  LoRA active from epoch 1  |  SWA from epoch {SWA_START}")
    print(f"  Class weights switch on at epoch 10")
    print(f"{'='*55}\n")

    for ep in range(1, NUM_EPOCHS + 1):
        # Switch to class-weighted loss after model is stable
        current_crit = crit_w if ep >= 10 else crit

        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ep {ep:02d}", leave=False):
            wav_b = b["wav"].to(device)
            pro_b = b["prosody"].to(device)
            lbl_b = b["label"].to(device)
            with autocast("cuda"):
                loss = current_crit(model(wav_b, pro_b), lbl_b)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            ep_loss += loss.item()

        # SWA accumulation
        if ep >= SWA_START:
            swa_model.update_parameters(model)
            swa_active = True

        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            tag = "  *** BEST ***"

        print(f"Ep {ep:02d} | Loss {ep_loss/len(tr_loader):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # SWA final
    if swa_active:
        print("\n[SWA] Finalizing averaged model...")
        swa_model.train()
        with torch.no_grad():
            for b in tqdm(tr_loader, desc="SWA BN", leave=False):
                with autocast("cuda"):
                    swa_model(b["wav"].to(device), b["prosody"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, va_loader, device)
        print(f"[SWA] Val Acc {swa_acc:.3f} | F1 {swa_f1:.3f}")
        if swa_acc > best_acc:
            best_acc = swa_acc
            torch.save(swa_model.state_dict(), save_path)
            print("[SWA] SWA model is the new best — saved.")

    # Final sliding window eval
    print("\n[SWE] Sliding-window inference on full validation files...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    swe_acc, swe_f1 = sliding_window_infer(model, va_df, path_map, device)

    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS")
    print(f"  Best Single-Crop Val Acc : {best_acc:.4f}")
    print(f"  Sliding Window Val Acc   : {swe_acc:.4f}")
    print(f"  Sliding Window Val F1    : {swe_f1:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()
