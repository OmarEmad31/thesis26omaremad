"""
Audio v12 — Egyptian Arabic SER — Maximum Push
================================================
ELITE upgrades across every dimension to break the 40% ceiling.

vs v11:
  1. 221-dim features (was 35):
     - MFCC 1-20 + per-utterance CMVN (speaker normalization)
     - Δ-MFCC and ΔΔ-MFCC (velocity + acceleration of speech)
     - Spectral contrast, chroma, F0 slope, HNR, tempo
  2. Multi-scale WavLM (3 layer groups: early/mid/late):
     - Early layers (1-4): fine acoustic phoneme features
     - Mid layers  (5-8): speaker + speaking style features
     - Late layers (9-12): semantic + emotion features
     Statistics pooling (mean+std) per group → 1536-dim → 256 each
  3. Feature group encoders (separate MLPs per feature type):
     - Prevents feature type interference in the joint space
  4. Phase 2: LoRA r=4 on q+k+v+out (all projections), LR=5e-6
  5. Deep residual classifier with BatchNorm
  6. 8 Phase 1 epochs (more warmup for richer feature space)
"""
import os, sys, subprocess, zipfile, random
from collections import defaultdict

def install_deps():
    pkgs = []
    for mod, pkg in [("audiomentations","audiomentations"),("peft","peft"),
                     ("transformers","transformers"),("noisereduce","noisereduce"),
                     ("pyloudnorm","pyloudnorm")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable,"-m","pip","install",*pkgs,"-q"])

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
import noisereduce as nr
import pyloudnorm as pyln

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = 80000         # 5 seconds
K_PER_CLASS    = 2
NUM_CLASSES    = 7
BATCH_SIZE     = K_PER_CLASS * NUM_CLASSES   # 14
BATCH_FROZEN   = 32
PHASE1_EPOCHS  = 8
PHASE2_EPOCHS  = 12
SWA_START      = 6
FOCAL_GAMMA    = 2.0
RARE_THRESHOLD = 50
MODEL_NAME     = "microsoft/wavlm-base-plus"

# Feature group sizes (computed from extract_comprehensive_features)
MFCC_DIM      = 160   # MFCC(20) + ΔMFCC(20) + ΔΔMFCC(20), each mean+std+min+max or mean+std
SPECTRAL_DIM  = 46    # spectral centroid/bandwidth/rolloff/contrast/flatness/chroma
PROSODY_DIM   = 15    # F0 slope + voiced ratio + RMS + ZCR + tempo + HNR
TOTAL_FEAT    = MFCC_DIM + SPECTRAL_DIM + PROSODY_DIM   # 221


# ─────────────────────────────────────────────────────────
# FOCAL LOSS with label smoothing for better generalization
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits, labels):
        n = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (n - 1))
            smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
        log_p = F.log_softmax(logits, dim=-1)
        ce    = -(smooth_targets * log_p).sum(dim=-1)
        p_t   = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        return (((1 - p_t) ** self.gamma) * ce).mean()


# ─────────────────────────────────────────────────────────
# BALANCED BATCH SAMPLER — min-class, no distribution shift
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
            random.shuffle(batch); yield batch

    def __len__(self): return self.n_batches


# ─────────────────────────────────────────────────────────
# MULTI-SCALE WavLM with statistics pooling
# ─────────────────────────────────────────────────────────
def stats_pool(h: torch.Tensor) -> torch.Tensor:
    """[B, T, D] → [B, 2D]: mean + std statistics pooling"""
    return torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)


class WavLMv12(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        # LoRA r=4 on ALL 4 projections — more expressive, less disruptive
        cfg  = LoraConfig(r=4, lora_alpha=8,
                          target_modules=["q_proj","k_proj","v_proj","out_proj"],
                          lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)

        # 3 layer groups — weighted sum per group
        self.w_early = nn.Parameter(torch.ones(4))   # layers  1-4
        self.w_mid   = nn.Parameter(torch.ones(4))   # layers  5-8
        self.w_late  = nn.Parameter(torch.ones(4))   # layers 9-12

        # Statistics pooling projects: [768×2] → 256 per group
        self.proj_early = nn.Sequential(nn.Linear(768*2, 256), nn.LayerNorm(256), nn.GELU())
        self.proj_mid   = nn.Sequential(nn.Linear(768*2, 256), nn.LayerNorm(256), nn.GELU())
        self.proj_late  = nn.Sequential(nn.Linear(768*2, 256), nn.LayerNorm(256), nn.GELU())

        # Feature group encoders
        self.mfcc_enc     = nn.Sequential(
            nn.Linear(MFCC_DIM, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3))
        self.spectral_enc = nn.Sequential(
            nn.Linear(SPECTRAL_DIM, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2))
        self.prosody_enc  = nn.Sequential(
            nn.Linear(PROSODY_DIM, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.2))

        # Total: 256×3 + 128 + 64 + 32 = 992 → deep residual classifier
        total = 256 * 3 + 128 + 64 + 32  # 992
        self.fc1     = nn.Linear(total, 512)
        self.bn1     = nn.BatchNorm1d(512)
        self.fc2     = nn.Linear(512, 256)
        self.bn2     = nn.BatchNorm1d(256)
        self.res_fc  = nn.Linear(total, 256)   # residual projection
        self.res_bn  = nn.BatchNorm1d(256)
        self.fc3     = nn.Linear(256, 128)
        self.bn3     = nn.BatchNorm1d(128)
        self.head    = nn.Linear(128, num_labels)
        self.drop    = nn.Dropout(0.5)
        self.drop2   = nn.Dropout(0.4)

    def freeze_backbone(self):
        for n, p in self.wavlm.named_parameters():
            p.requires_grad = "lora_" in n
        for p in [self.w_early, self.w_mid, self.w_late]:
            p.requires_grad = True

    def unfreeze_lora(self):
        for n, p in self.wavlm.named_parameters():
            p.requires_grad = "lora_" in n

    def forward(self, wav, feats):
        # Normalize waveform
        wav = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        out  = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)
        hs   = out.hidden_states  # [13][B, T, 768]: 0=embedding, 1-12=layers

        # 3 layer groups — weighted mean then statistics pool
        we = F.softmax(self.w_early, 0); wm = F.softmax(self.w_mid, 0); wl = F.softmax(self.w_late, 0)
        h_early = sum(we[i] * hs[i + 1] for i in range(4))   # layers 1-4
        h_mid   = sum(wm[i] * hs[i + 5] for i in range(4))   # layers 5-8
        h_late  = sum(wl[i] * hs[i + 9] for i in range(4))   # layers 9-12

        e = self.proj_early(stats_pool(h_early))  # [B, 256]
        m = self.proj_mid(stats_pool(h_mid))       # [B, 256]
        l = self.proj_late(stats_pool(h_late))     # [B, 256]

        # Feature group encoders
        mfcc_f     = self.mfcc_enc(feats[:, :MFCC_DIM])
        spectral_f = self.spectral_enc(feats[:, MFCC_DIM:MFCC_DIM + SPECTRAL_DIM])
        prosody_f  = self.prosody_enc(feats[:, MFCC_DIM + SPECTRAL_DIM:])

        # Combine all representations
        x = torch.cat([e, m, l, mfcc_f, spectral_f, prosody_f], dim=-1)  # [B, 992]

        # Residual deep classifier
        h1 = self.drop(F.gelu(self.bn1(self.fc1(x))))
        h2_main = self.bn2(self.fc2(h1))
        h2_res  = self.res_bn(self.res_fc(x))   # skip connection from input
        h2 = F.gelu(h2_main + h2_res)
        h2 = self.drop2(h2)
        h3 = F.gelu(self.bn3(self.fc3(h2)))
        return self.head(h3)


# ─────────────────────────────────────────────────────────
# ELITE PREPROCESSING — loudness + denoise + pre-emphasis + smart crop
# ─────────────────────────────────────────────────────────
def elite_preprocess(y: np.ndarray, sr: int = SR) -> np.ndarray:
    try:
        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(loudness) and -80 < loudness < 0:
            y = pyln.normalize.loudness(y.astype(np.float64),
                                        loudness, -23.0).astype(np.float32)
    except Exception: pass
    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True,
                            prop_decrease=0.75).astype(np.float32)
    except Exception: pass
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < sr // 4: yt = y
    if len(yt) > MAX_LEN:
        step = sr // 4; best_start = 0; best_rms = -1.0
        for s in range(0, len(yt) - MAX_LEN + 1, step):
            rms = float(np.mean(yt[s:s + MAX_LEN] ** 2))
            if rms > best_rms: best_rms, best_start = rms, s
        yt = yt[best_start:best_start + MAX_LEN]
    else:
        yt = np.pad(yt, (0, MAX_LEN - len(yt)))
    return np.clip(yt, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# COMPREHENSIVE 221-DIM FEATURE EXTRACTION
# Group A (160): MFCC×20 + ΔMFCC + ΔΔMFCC  with CMVN
# Group B (46): spectral centroid/bandwidth/rolloff/contrast/flatness + chroma
# Group C (15): F0 + RMS + ZCR + tempo + HNR estimate
# ─────────────────────────────────────────────────────────
def extract_comprehensive_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    hop = 512   # larger hop = faster computation
    feats = []

    # ── GROUP A: MFCC statistics (160-dim) ──────────────────
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                                     hop_length=hop, n_fft=1024)
        # Per-utterance CMVN: remove speaker-specific mean per coefficient
        mfcc_n = (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-6)
        dmfcc  = librosa.feature.delta(mfcc_n, order=1)
        ddmfcc = librosa.feature.delta(mfcc_n, order=2)
        # Base MFCC: mean+std+min+max = 80
        feats += mfcc_n.mean(1).tolist() + mfcc_n.std(1).tolist() + \
                 mfcc_n.min(1).tolist()  + mfcc_n.max(1).tolist()
        # ΔMFCC: mean+std = 40
        feats += dmfcc.mean(1).tolist() + dmfcc.std(1).tolist()
        # ΔΔMFCC: mean+std = 40
        feats += ddmfcc.mean(1).tolist() + ddmfcc.std(1).tolist()
    except Exception:
        feats += [0.0] * MFCC_DIM

    # ── GROUP B: Spectral + Chroma (46-dim) ─────────────────
    try:
        sc   = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        sb   = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
        sro  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
        feats += [sc.mean()/sr, sc.std()/sr,
                  sb.mean()/sr, sb.std()/sr,
                  sro.mean()/sr, sro.std()/sr]  # 6
        # Spectral contrast (n_bands=6 → 7 sub-bands)
        scon = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, hop_length=hop)
        feats += scon.mean(1).tolist() + scon.std(1).tolist()  # 14
        # Spectral flatness
        sf = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
        feats += [float(sf.mean()), float(sf.std())]  # 2
        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
        feats += chroma.mean(1).tolist() + chroma.std(1).tolist()  # 24
    except Exception:
        feats += [0.0] * SPECTRAL_DIM

    # ── GROUP C: Prosodic + Voice quality (15-dim) ──────────
    try:
        # F0 with larger hop (faster)
        f0    = librosa.yin(y, fmin=65, fmax=2093, hop_length=hop)
        valid = f0[np.isfinite(f0) & (f0 > 0)]
        if len(valid) > 2:
            slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0])
            feats += [valid.mean()/500, valid.std()/100, valid.ptp()/500,
                      slope/100, len(valid)/len(f0),
                      float(np.percentile(valid, 25))/500,
                      float(np.percentile(valid, 75))/500]  # 7
        else:
            feats += [0.] * 7
        # RMS energy: mean, std, max = 3
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        feats += [float(rms.mean()), float(rms.std()), float(rms.max())]
        # ZCR: mean, std = 2
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        feats += [float(zcr.mean()), float(zcr.std())]
        # Tempo = 1
        tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=hop)[0])
        feats += [tempo / 200.0]
        # HNR estimate (harmonic energy / total energy) = 1
        y_h = librosa.effects.harmonic(y, margin=8.0)
        hnr = float(np.mean(y_h**2)) / (float(np.mean(y**2)) + 1e-8)
        feats += [np.clip(hnr, 0, 1)]
    except Exception:
        feats += [0.0] * PROSODY_DIM

    arr = np.array(feats[:TOTAL_FEAT], dtype=np.float32)
    if len(arr) < TOTAL_FEAT:
        arr = np.pad(arr, (0, TOTAL_FEAT - len(arr)))
    return np.clip(np.nan_to_num(arr, nan=0., posinf=1., neginf=-1.), -10, 10)


# ─────────────────────────────────────────────────────────
# DATASET — elite preprocessing + comprehensive features + in-memory cache
# ─────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, df, path_map, augment=False, rare_classes=None):
        self.df           = df.reset_index(drop=True)
        self.path_map     = path_map
        self.augment      = augment
        self.rare_classes = set(rare_classes or [])
        self.aug_pipe     = Compose([AddGaussianNoise(p=0.4), PitchShift(p=0.4)])
        self.class_idx    = defaultdict(list)
        for i, row in self.df.iterrows():
            self.class_idx[int(row["lid"])].append(i)
        self._cache: dict = {}   # fname → (audio_np, feat_np)

    def __len__(self): return len(self.df)

    def _load_audio(self, idx):
        row   = self.df.iloc[idx]
        fname = Path(row["audio_relpath"]).name
        if fname not in self.path_map: return None, None, None
        if fname in self._cache:
            yt, ft = self._cache[fname]
            return yt.copy(), int(row["lid"]), ft.copy()
        try:
            raw, _ = librosa.load(self.path_map[fname], sr=SR)
            yt     = elite_preprocess(raw)
            ft     = extract_comprehensive_features(yt)
            self._cache[fname] = (yt, ft)
            return yt.copy(), int(row["lid"]), ft.copy()
        except Exception:
            return None, None, None

    def _load(self, idx):
        yt, lid, ft = self._load_audio(idx)
        if yt is None: return None
        try:
            if self.augment:
                # Intra-class mixup for rare classes
                if lid in self.rare_classes and random.random() < 0.5:
                    peer_idx = random.choice(self.class_idx[lid])
                    r2 = self._load_audio(peer_idx)
                    if r2[0] is not None:
                        a  = np.random.beta(0.4, 0.4)
                        yt = a * yt + (1 - a) * r2[0]
                        ft = a * ft + (1 - a) * r2[2]   # mix features too

                # SpecAugment time mask
                T = len(yt); w = int(T * 0.15 * random.random())
                if w > 0:
                    s = random.randint(0, T - w)
                    yt = yt.copy(); yt[s:s + w] = 0.0

                # Audio augmentation → recompute features
                yt = self.aug_pipe(samples=yt.astype(np.float32), sample_rate=SR)
                ft = extract_comprehensive_features(yt)    # fresh features on augmented audio

            return {
                "wav":   torch.tensor(yt, dtype=torch.float32),
                "feats": torch.tensor(ft, dtype=torch.float32),
                "label": torch.tensor(lid, dtype=torch.long),
            }
        except Exception:
            return None

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            s = self._load(idx)
            if s is not None: return s
            idx = (idx + 1) % len(self.df)
        raise RuntimeError("No loadable sample found.")


# ─────────────────────────────────────────────────────────
# PATH MAP
# ─────────────────────────────────────────────────────────
def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: print(f"[SUCCESS] Found {len(pm)} wav files."); return pm
    zname = "Thesis_Audio_Full.zip"; zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if zname in files: zpath = os.path.join(root, zname); break
    if not zpath: raise FileNotFoundError(f"{zname} not found.")
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    pm = {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}
    print(f"[SUCCESS] Extracted {len(pm)} files."); return pm


# ─────────────────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────────────────
def fast_eval(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            with autocast("cuda"):
                logits = model(b["wav"].to(device), b["feats"].to(device))
            ps.extend(logits.argmax(1).cpu().numpy())
            ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro", zero_division=0)


def tta_eval(model, df, path_map, device):
    """TTA: original + 0.9x + 1.1x speed, averaged."""
    model.eval(); all_probs = []; all_labels = []
    for _, row in df.iterrows():
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            raw, _ = librosa.load(path_map[fname], sr=SR)
            yt     = elite_preprocess(raw)
            seg_probs = []
            for spd in [1.0, 0.9, 1.1]:
                if spd != 1.0:
                    try:
                        yr = librosa.resample(yt, orig_sr=SR, target_sr=int(SR*spd))
                        seg = yr[:MAX_LEN] if len(yr) >= MAX_LEN else np.pad(yr,(0,MAX_LEN-len(yr)))
                    except Exception: seg = yt
                else: seg = yt
                ft = extract_comprehensive_features(seg)
                with torch.no_grad(), autocast("cuda"):
                    l = model(torch.tensor(seg).unsqueeze(0).to(device),
                               torch.tensor(ft).unsqueeze(0).to(device))
                seg_probs.append(F.softmax(l, dim=-1).cpu())
            all_probs.append(torch.stack(seg_probs).mean(0))
            all_labels.append(int(row["lid"]))
        except Exception: pass
    if not all_probs: return 0., 0.
    probs  = torch.cat(all_probs, dim=0)
    preds  = probs.argmax(1).numpy()
    return (accuracy_score(all_labels, preds),
            f1_score(all_labels, preds, average="macro", zero_division=0))


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────
def train():
    device     = "cuda"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p      = colab_root / "data/processed/splits/text_hc"
    p1_save    = colab_root / "v12_phase1_best.pt"
    p2_save    = colab_root / "v12_phase2_best.pt"
    swa_save   = colab_root / "v12_swa.pt"

    path_map = get_path_map(colab_root)

    tr_df = pd.read_csv(csv_p / "train.csv")
    va_df = pd.read_csv(csv_p / "val.csv")
    lid   = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    tr_df["lid"] = tr_df["emotion_final"].map(lid)
    va_df["lid"] = va_df["emotion_final"].map(lid)

    class_counts = tr_df["emotion_final"].value_counts()
    rare_names   = set(class_counts[class_counts < RARE_THRESHOLD].index)
    rare_lids    = {lid[l] for l in rare_names if l in lid}

    print(f"\n[DATA] Train: {len(tr_df)} | Val: {len(va_df)} | Classes: {len(lid)}")
    print(f"[DATA] Features: {TOTAL_FEAT}-dim | Model input: 992-dim")
    for emo, cnt in class_counts.items():
        print(f"  {emo:12s}: {cnt:3d}{' ← mixup' if lid.get(emo,-1) in rare_lids else ''}")

    tr_ds     = AudioDataset(tr_df, path_map, augment=True,  rare_classes=rare_lids)
    va_ds     = AudioDataset(va_df, path_map, augment=False, rare_classes=None)
    tr_frozen = DataLoader(tr_ds, batch_size=BATCH_FROZEN, shuffle=True,
                           drop_last=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=16, num_workers=0)

    def make_sampler():
        return BalancedBatchSampler(tr_df["lid"].values, k=K_PER_CLASS)

    model     = WavLMv12(num_labels=len(lid)).to(device)
    swa_model = AveragedModel(model)
    crit      = FocalLoss(gamma=FOCAL_GAMMA, smoothing=0.05)
    scaler    = GradScaler("cuda")
    p1_best   = 0.0; p2_best = 0.0; swa_active = False

    # ═══════════════════════════════════════════════════
    # PHASE 1 — Head Warmup (frozen WavLM backbone)
    # ═══════════════════════════════════════════════════
    model.freeze_backbone()
    p1_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1_params, lr=1e-3, weight_decay=0.01)
    sch1 = get_cosine_schedule_with_warmup(opt1, 0, len(tr_frozen) * PHASE1_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Head Warmup ({PHASE1_EPOCHS} epochs, backbone frozen)")
    print(f"  Trainable: {sum(p.numel() for p in p1_params):,} | Batch: {BATCH_FROZEN}")
    print(f"  Features: {TOTAL_FEAT}-dim | Groups: MFCC({MFCC_DIM})+Spectral({SPECTRAL_DIM})+Prosody({PROSODY_DIM})")
    print(f"{'='*60}")

    for ep in range(1, PHASE1_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_frozen, desc=f"Ph1 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["feats"].to(device)),
                            b["label"].to(device))
            opt1.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt1); nn.utils.clip_grad_norm_(p1_params, 1.0)
            scaler.step(opt1); scaler.update(); sch1.step()
            ep_loss += loss.item()
        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > p1_best:
            p1_best = acc; torch.save(model.state_dict(), p1_save); tag = "  *** P1 BEST ***"
        print(f"Ph1 Ep {ep:02d} | Loss {ep_loss/len(tr_frozen):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # PHASE 2 — LoRA fine-tune (r=4 on all projections, LR=5e-6)
    # ═══════════════════════════════════════════════════
    model.unfreeze_lora()
    non_lora = [p for n, p in model.wavlm.named_parameters()
                if "lora_" not in n and p.requires_grad]
    lora     = [p for n, p in model.wavlm.named_parameters() if "lora_" in n]
    opt2 = torch.optim.AdamW([
        {"params": non_lora,                           "lr": 5e-7},
        {"params": lora,                               "lr": 5e-6},
        {"params": [*model.w_early, *[model.w_mid], *[model.w_late]] +
                   list(model.proj_early.parameters()) +
                   list(model.proj_mid.parameters())   +
                   list(model.proj_late.parameters()),  "lr": 1e-4},
        {"params": list(model.mfcc_enc.parameters())   +
                   list(model.spectral_enc.parameters()) +
                   list(model.prosody_enc.parameters()), "lr": 5e-5},
        {"params": [p for n, p in model.named_parameters()
                    if any(x in n for x in ["fc1","fc2","fc3","bn1","bn2","bn3",
                                            "head","res_fc","res_bn"])], "lr": 5e-5},
    ], weight_decay=0.01)
    smp  = make_sampler()
    sch2 = get_cosine_schedule_with_warmup(opt2, len(smp), len(smp) * PHASE2_EPOCHS)
    lora_n = sum(p.numel() for p in lora)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — LoRA Fine-tune ({PHASE2_EPOCHS} epochs, Focal only)")
    print(f"  LoRA r=4 q+k+v+out | {lora_n:,} LoRA params | LR=5e-6")
    print(f"  SWA from Ep{SWA_START}")
    print(f"{'='*60}")

    for ep in range(1, PHASE2_EPOCHS + 1):
        smp       = make_sampler()
        tr_loader = DataLoader(tr_ds, batch_sampler=smp, num_workers=0)
        model.train(); ep_loss = 0.0
        for b in tqdm(tr_loader, desc=f"Ph2 Ep{ep:02d}", leave=False):
            with autocast("cuda"):
                loss = crit(model(b["wav"].to(device), b["feats"].to(device)),
                            b["label"].to(device))
            opt2.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt2); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update(); sch2.step()
            ep_loss += loss.item()

        if ep >= SWA_START:
            swa_model.update_parameters(model); swa_active = True

        acc, f1 = fast_eval(model, va_loader, device)
        tag = ""
        if acc > p2_best:
            p2_best = acc; torch.save(model.state_dict(), p2_save); tag = "  *** P2 BEST ***"
        print(f"Ph2 Ep {ep:02d} | Loss {ep_loss/len(smp):.3f} | "
              f"Val Acc {acc:.3f} | F1 {f1:.3f}{tag}")

    # ═══════════════════════════════════════════════════
    # SWA
    # ═══════════════════════════════════════════════════
    if swa_active:
        print("\n[SWA] Finalizing batch norm statistics...")
        swa_model.train()
        smp = make_sampler()
        with torch.no_grad():
            for b in DataLoader(tr_ds, batch_sampler=smp, num_workers=0):
                with autocast("cuda"):
                    swa_model(b["wav"].to(device), b["feats"].to(device))
        swa_acc, swa_f1 = fast_eval(swa_model, va_loader, device)
        print(f"[SWA] Val Acc {swa_acc:.3f} | F1 {swa_f1:.3f}")
        torch.save(swa_model.state_dict(), swa_save)

    # ═══════════════════════════════════════════════════
    # FINAL: Multi-checkpoint TTA ensemble
    # ═══════════════════════════════════════════════════
    print("\n[FINAL] Multi-checkpoint TTA ensemble evaluation...")
    best_single = max(p1_best, p2_best)
    results = []

    for name, path in [("P1-best", p1_save), ("P2-best", p2_save),
                        ("SWA",     swa_save if swa_active else p2_save)]:
        if not Path(path).exists(): continue
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        ta, tf = tta_eval(model, va_df, path_map, device)
        print(f"  [{name}] TTA Acc {ta:.4f} | F1 {tf:.4f}")
        results.append(ta)

    print(f"\n{'='*60}")
    print(f"  Best single-crop : {best_single:.4f}")
    print(f"  Best TTA result  : {max(results) if results else 0.:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
