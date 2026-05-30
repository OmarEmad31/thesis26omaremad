"""
fusion_contrastive_v2.py — 2-Phase Contrastive Pipeline (Colab)
================================================================
PHASE 1  Self-Supervised Pre-training  (InfoNCE, NO labels)
  1-A : Audio  SimCLR  (WavLM-Base-Plus, waveform augmentation)
  1-B : Text   SimCSE  (MARBERT, dropout augmentation)
  1-C : Video  SimCLR  (MSW Transformer on CLIP+DINOv2+ResNet50)
  1-D : Cross-modal    (Audio-Video InfoNCE, labelled + small unlabelled pool)

PHASE 2  Supervised Fine-tuning  (CE + SupCon, WITH labels)
  4 ablation scenarios: Baseline / SupCon-only / SSL-only / SSL+SupCon

Run on Colab T4.  exec() compatible.
"""

import os, re, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import WavLMModel, AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS & GLOBAL CONFIG (Smart Auto-Detection)
# ─────────────────────────────────────────────────────────
import glob
cands = glob.glob("/content/*thesis*") + glob.glob("/content/*omaremad*")
cands = [c for c in cands if os.path.isdir(c) and (Path(c)/"src").exists()]
_repo_str = cands[0] if cands else "/content/thesis"

REPO       = Path(_repo_str)
SPLIT_DIR  = REPO / "data/processed/splits/multimodal_eligible"
SAVE_DIR   = Path("/content/fusion_models")
SSL_DIR    = Path("/content/ssl_pretrained")
for d in [SAVE_DIR, SSL_DIR]: d.mkdir(exist_ok=True)

def auto_detect():
    print("  Smart-detecting data locations...")
    v_dir = None

    # 1. Search /content first (fast local SSD)
    for p in Path("/content").rglob("*_clip_seq.npy"):
        if "drive" not in str(p):
            v_dir = p.parent
            break

    # 2. Fallback to Drive
    if not v_dir:
        v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features/video_sequences_v1")
        if not v_dir.exists():
            v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features")

    a_dir = Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED")
    return v_dir, a_dir

VID_DIR, AUDIO_BASE = auto_detect()
print(f"  Final VID_DIR:    {VID_DIR}")
print(f"  Final AUDIO_BASE: {AUDIO_BASE}")

print("  [DEBUG] Sample files in VID_DIR:")
if VID_DIR.exists():
    for f in list(VID_DIR.glob("*_clip_seq.npy"))[:3]: print(f"    - {f.name}")
else: print("    (Directory does not exist)")

print("  [DEBUG] Contents of /content/audio:")
if Path("/content/audio").exists():
    for f in list(Path("/content/audio").iterdir())[:10]:
        print(f"    - {f.name} (IsDir: {f.is_dir()})")
        if f.is_dir():
            for sub_f in list(f.iterdir())[:3]:
                print(f"      -> {sub_f.name}")
else: print("    (Directory does not exist)")


LID     = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6}
CLASSES = list(LID.keys())
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# Per-modal SSL hyper-parameters
SSL_EPOCHS        = 40
SSL_TEMP          = 0.07
SSL_PROJ_DIM      = 128
GRAD_ACC          = 4       # effective audio batch = 8 * 4 = 32

# Cross-modal SSL (Phase 1-D)
CM_SSL_EPOCHS     = 20
CM_TEMP           = 0.07
CM_UNLABELLED_CAP = 610     # max unlabelled samples added to cross-modal pool

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sep(t=""):
    print("\n" + "="*56)
    if t: print(f"  {t}")
    print("="*56)

# ─────────────────────────────────────────────────────────
# DATA LOADING  (same as fusion_production_v1)
# ─────────────────────────────────────────────────────────
def resolve_audio_path(row):
    audio_rel = str(row.get('audio_relpath', ''))
    if not audio_rel: return None
    folder = str(row.get('folder', ''))

    bases = [
        AUDIO_BASE,
        Path("/content/audio/Thesis Project/dataset/Final Modalink Dataset MERGED"),
        Path("/content/audio/Thesis Project/data/raw"),
        Path("/content/audio/data/processed"),
        Path("/content/audio"),
        Path("/content/audio/Thesis_Audio_Full"),
        Path("/content/Thesis_Audio_Full"),
        Path("/content/drive/MyDrive/Thesis_Audio_Full"),
        Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED"),
        Path("/content/drive/MyDrive/Thesis Project/data/raw"),
        Path("/content/drive/MyDrive/Thesis Project"),
        Path("/content/drive/MyDrive")
    ]

    for base in bases:
        # Standard path
        p = base / folder / audio_rel if folder else base / audio_rel
        if p.exists(): return p
        # Windows backslash artifact
        if folder:
            bs_name = f"{folder}\\{audio_rel.replace('/', '\\')}"
            if (base / bs_name).exists(): return base / bs_name
        # Flat fallback
        p_flat = base / Path(audio_rel).name
        if p_flat.exists(): return p_flat

    return None

def get_vid_paths(sid):
    # Normal path
    p1 = VID_DIR / f"{sid}_clip_seq.npy"
    if p1.exists():
        return p1, VID_DIR / f"{sid}_dinov2_seq.npy", VID_DIR / f"{sid}_resnet50_seq.npy"
    # Windows backslash extraction artifact
    p2 = VID_DIR / f"video_sequences_v1\\{sid}_clip_seq.npy"
    if p2.exists():
        return p2, VID_DIR / f"video_sequences_v1\\{sid}_dinov2_seq.npy", VID_DIR / f"video_sequences_v1\\{sid}_resnet50_seq.npy"
    return None, None, None

def load_splits():
    if not SPLIT_DIR.exists():
        print(f"  [ERROR] Split directory not found: {SPLIT_DIR}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tr = pd.read_csv(SPLIT_DIR/"train.csv")
    va = pd.read_csv(SPLIT_DIR/"val.csv")
    te = pd.read_csv(SPLIT_DIR/"test.csv")

    sep("PATH DIAGNOSTIC")
    row0 = tr.iloc[0]
    sid0 = row0['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")

    v_test, _, _ = get_vid_paths(sid0)
    a_test = resolve_audio_path(row0)
    vid_ok = v_test is not None and v_test.exists()
    aud_ok = a_test is not None and a_test.exists()
    txt_ok = isinstance(row0.get('transcript'), str) and len(str(row0['transcript']).strip()) > 2

    print(f"  Sample ID: {sid0}")
    print(f"  Video Status: {'OK' if vid_ok else 'MISSING'} ({v_test.name if v_test else 'Not found'})")
    if aud_ok:
        print(f"  Audio Status: OK ({a_test.name})")
    else:
        print(f"  Audio Status: MISSING")
        print(f"      -> folder='{row0.get('folder','')}' file='{row0.get('audio_relpath','')}'")
    print(f"  Text  Status: {'OK' if txt_ok else 'EMPTY'}")

    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, _, _ = get_vid_paths(sid)
        return (pc is not None and pc.exists() and
                resolve_audio_path(row) is not None and
                isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2)

    tr_f = tr[tr.apply(ok, axis=1)].reset_index(drop=True)
    va_f = va[va.apply(ok, axis=1)].reset_index(drop=True)
    te_f = te[te.apply(ok, axis=1)].reset_index(drop=True)

    sep("ALIGNED SPLITS")
    print(f"  Train: {len(tr_f)} | Val: {len(va_f)} | Test: {len(te_f)}")
    if len(tr_f) == 0:
        print("  [CRITICAL] 0 samples aligned. Check if VID_DIR and AUDIO_BASE are correct.")
        print(f"  VID_DIR: {VID_DIR}  |  AUDIO_BASE: {AUDIO_BASE}")
    return tr_f, va_f, te_f


def load_unlabelled():
    """Load up to CM_UNLABELLED_CAP unlabelled samples for cross-modal SSL.
    Reads from all_segments.xlsx (primary) or unlabelled.csv (fallback).
    No emotion_final column needed — only sample_id, folder, audio_relpath.
    Returns empty DataFrame if neither file is found.
    """
    xlsx_paths = [
        REPO / "data/processed/all_segments.xlsx",
        REPO / "data/all_segments.xlsx",
        Path("/content/drive/MyDrive/Thesis Project/data/processed/all_segments.xlsx"),
        Path("/content/drive/MyDrive/Thesis Project/all_segments.xlsx"),
    ]
    csv_paths = [
        REPO / "data/processed/splits/unlabelled.csv",
        SPLIT_DIR / "unlabelled.csv",
    ]

    df = None
    for p in xlsx_paths:
        if p.exists():
            print(f"  [Unlabelled] Loading from {p.name} ...")
            df = pd.read_excel(p)
            print(f"  [Unlabelled] {len(df)} segments loaded")
            break

    if df is None:
        for p in csv_paths:
            if p.exists():
                print(f"  [Unlabelled] Loading from {p.name} ...")
                df = pd.read_csv(p)
                print(f"  [Unlabelled] {len(df)} segments loaded")
                break

    if df is None:
        print("  [Unlabelled] No all_segments.xlsx or unlabelled.csv found — cross-modal SSL uses labelled pool only")
        return pd.DataFrame()

    if len(df) > CM_UNLABELLED_CAP:
        df = df.sample(CM_UNLABELLED_CAP, random_state=42).reset_index(drop=True)
        print(f"  [Unlabelled] Capped to {CM_UNLABELLED_CAP} samples (CM_UNLABELLED_CAP)")
    return df


# ─────────────────────────────────────────────────────────
# CONTRASTIVE LOSSES
# ─────────────────────────────────────────────────────────
class InfoNCELoss(nn.Module):
    """NT-Xent / SimCLR loss. No labels. Positive = two views of same sample."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B  = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)          # [2B, D]
        sim = torch.mm(z, z.T) / self.T           # [2B, 2B]
        sim.fill_diagonal_(float('-inf'))
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0,   B, device=z.device)
        ])
        return F.cross_entropy(sim, labels)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim)
        )
    def forward(self, x): return self.net(x)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020). Used in Phase 2."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        B = features.size(0)
        sim = torch.mm(features, features.T) / self.T
        # Mask diagonal to -inf before softmax so self-similarity never enters denominator
        sim.masked_fill_(torch.eye(B, device=features.device).bool(), float('-inf'))
        pos_mask = labels.view(-1,1).eq(labels.view(1,-1)).float()
        pos_mask.fill_diagonal_(0)
        log_prob = F.log_softmax(sim, dim=1)
        n_pos    = pos_mask.sum(1).clamp(min=1)   # per-anchor positive count
        return (-(pos_mask * log_prob).sum(1) / n_pos).mean()


# ─────────────────────────────────────────────────────────
# AUGMENTATIONS  (inline — Colab exec() compatible)
# ─────────────────────────────────────────────────────────
def _audio_one_view(wav, maxlen=80000):
    w = wav.copy()
    if np.random.rand() > 0.3:
        snr = np.random.uniform(15, 30)
        rms = np.sqrt(np.mean(w**2)) + 1e-9
        w  += np.random.randn(len(w)) * rms / (10**(snr/20))
    if np.random.rand() > 0.4:
        ml = int(len(w) * np.random.uniform(0.05, 0.15))
        ms = np.random.randint(0, max(1, len(w)-ml))
        w[ms:ms+ml] = 0.0
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        idx  = np.linspace(0, len(w)-1, int(len(w)/rate)).astype(int)
        w    = w[idx]
        w    = w[:maxlen] if len(w) > maxlen else np.pad(w, (0, maxlen-len(w)))
    return w

def audio_augment(wav): return _audio_one_view(wav), _audio_one_view(wav)

def _video_one_view(seq, n_mask=4, noise_std=0.02):
    s    = seq.copy()
    idxs = np.random.choice(s.shape[0], n_mask, replace=False)
    s[idxs] = 0.0
    if np.random.rand() > 0.4:
        s += np.random.randn(*s.shape) * noise_std
    return s

def video_augment(seq): return _video_one_view(seq), _video_one_view(seq)


# ─────────────────────────────────────────────────────────
# PHASE 1-A — AUDIO SSL (SimCLR)
# ─────────────────────────────────────────────────────────
class AudioSSLDS(Dataset):
    def __init__(self, df, sr=16000, maxlen=80000):
        self.df = df.reset_index(drop=True)
        self.sr = sr; self.maxlen = maxlen
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            p = resolve_audio_path(r)
            if p is None: raise FileNotFoundError
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y    = y[:self.maxlen] if len(y)>self.maxlen else np.pad(y,(0,self.maxlen-len(y)))
        except: y = np.zeros(self.maxlen)
        v1, v2 = audio_augment(y)
        return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)

class AudioSSLModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus",
                                                    output_hidden_states=True)
        # Freeze bottom 6 layers for efficiency; top 6 adapt during SSL
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < 6:
                for p in layer.parameters(): p.requires_grad = False
        self.lw   = nn.Parameter(torch.ones(13))
        self.proj = ProjectionHead(768*2, proj_dim)

    def encode(self, x):
        hs  = torch.stack(self.backbone(x).hidden_states, 0)
        out = (hs * F.softmax(self.lw, 0).view(-1,1,1,1)).sum(0)
        return torch.cat([out.mean(1), out.std(1)], 1)   # [B, 1536]

    def forward(self, x):
        return self.proj(self.encode(x))

def train_audio_ssl(pool):
    sep("PHASE 1-A -- AUDIO SSL (SimCLR | WavLM-Base-Plus)")
    ckpt = SSL_DIR / "audio_ssl.pt"
    if ckpt.exists():
        print("  [SKIP] audio_ssl.pt already cached — delete to retrain.")
        return
    set_seed(42)
    ds  = AudioSSLDS(pool)
    dl  = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2,
                     pin_memory=True, drop_last=True)
    m       = AudioSSLModel(SSL_PROJ_DIM).to(DEVICE)
    opt     = torch.optim.AdamW(
                  filter(lambda p: p.requires_grad, m.parameters()),
                  lr=1e-4, weight_decay=1e-2)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
    loss_fn = InfoNCELoss(SSL_TEMP)
    scaler  = GradScaler()
    print(f"  Pool: {len(pool)} | EffBatch: {8*GRAD_ACC} | Epochs: {SSL_EPOCHS}")
    for ep in range(1, SSL_EPOCHS+1):
        m.train(); ep_loss = 0.0; opt.zero_grad()
        for step, (v1, v2) in enumerate(dl):
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            with autocast("cuda"):
                loss = loss_fn(m(v1), m(v2)) / GRAD_ACC
            scaler.scale(loss).backward()
            ep_loss += loss.item() * GRAD_ACC
            if (step+1) % GRAD_ACC == 0 or (step+1) == len(dl):
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{SSL_EPOCHS} | InfoNCE: {ep_loss/len(dl):.4f}")
    # Save encoder only (no projection head)
    torch.save({'backbone': m.backbone.state_dict(), 'lw': m.lw.data}, str(ckpt))
    print(f"  [SAVED] {ckpt}")
    del m; torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
# PHASE 1-B — TEXT SSL (SimCSE)
# ─────────────────────────────────────────────────────────
MODEL_NAME = "UBC-NLP/MARBERT"
_FILLERS   = re.compile(
    r'\b(اه|ايه|يعني|بص|كده|كدا|اهو|والله|عشان|بقا|بقى|يا|اوه|هاه|اوكي|اوكى|تمام|صح|ايوه|لا|مش|ميش)\b'
)
def clean(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'[ً-ٰٟ]', '', t)
    t = re.sub(r'[آأإ]', 'ا', t)
    t = re.sub(r'ة', 'ه', t)
    t = re.sub(r'ى', 'ي', t)
    t = re.sub(r'ـ', '', t)
    t = _FILLERS.sub(' ', t)
    t = re.sub(r'(.)\1+', r'\1\1', t)
    return re.sub(r'\s+', ' ', t).strip()

class TextSSLDS(Dataset):
    def __init__(self, texts, tok):
        self.enc = tok([clean(str(t)) for t in texts],
                       truncation=True, padding="max_length",
                       max_length=64, return_tensors="pt")
    def __len__(self): return self.enc['input_ids'].size(0)
    def __getitem__(self, i): return {k: v[i] for k,v in self.enc.items()}

class TextSSLModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        # Freeze bottom 8 BERT layers; top 4 adapt during SSL
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < 8:
                for p in layer.parameters(): p.requires_grad = False
        self.proj = ProjectionHead(768*3, proj_dim)

    def encode(self, ids, mask):
        lh  = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        msk = mask.unsqueeze(-1).expand(lh.size()).float()
        mp  = (lh*msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        xp  = (lh*msk - (1-msk)*1e9).max(1)[0]
        return torch.cat([lh[:,0,:], mp, xp], 1)   # [B, 2304]

    def forward(self, ids, mask):
        return self.proj(self.encode(ids, mask))

def train_text_ssl(pool):
    sep("PHASE 1-B -- TEXT SSL (SimCSE | MARBERT)")
    ckpt = SSL_DIR / "text_ssl.pt"
    if ckpt.exists():
        print("  [SKIP] text_ssl.pt already cached — delete to retrain.")
        return
    set_seed(42)
    tok    = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool_t = pool[pool['transcript'].apply(
        lambda t: isinstance(t, str) and len(t.strip()) > 2)]
    ds     = TextSSLDS(pool_t['transcript'].values, tok)
    dl     = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m      = TextSSLModel(SSL_PROJ_DIM).to(DEVICE)
    opt    = torch.optim.AdamW([
        {'params': [p for n,p in m.named_parameters() if 'bert' in n and p.requires_grad], 'lr': 2e-5},
        {'params': [p for n,p in m.named_parameters() if 'proj' in n],                    'lr': 1e-3}
    ], weight_decay=0.01)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
    loss_fn = InfoNCELoss(SSL_TEMP)
    print(f"  Pool: {len(pool_t)} | Batches/ep: {len(dl)} | Epochs: {SSL_EPOCHS}")
    for ep in range(1, SSL_EPOCHS+1):
        m.train(); ep_loss = 0.0
        for batch in dl:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            opt.zero_grad()
            # SimCSE: same tokens, two forward passes — dropout is the augmentation
            z1 = m(ids, mask)
            z2 = m(ids, mask)
            loss = loss_fn(z1, z2)
            loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{SSL_EPOCHS} | InfoNCE: {ep_loss/len(dl):.4f}")
    # Save BERT weights only (drop projection head)
    torch.save({k: v for k,v in m.state_dict().items() if 'proj' not in k}, str(ckpt))
    print(f"  [SAVED] {ckpt}")
    del m; torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
# PHASE 1-C — VIDEO SSL (Feature SimCLR)
# ─────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(nn.Linear(c, c//16), nn.ReLU(),
                                  nn.Linear(c//16, c), nn.Sigmoid())
    def forward(self, x):
        b,n,c = x.shape
        return x * self.fc(self.pool(x.transpose(1,2)).view(b,c)).view(b,1,c)

class VideoSSLDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r   = self.df.iloc[i]
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd_, pr = get_vid_paths(sid)
        seq = np.concatenate([np.load(pc), np.load(pd_), np.load(pr)], -1)
        v1, v2 = video_augment(seq)
        return (torch.tensor(v1, dtype=torch.float32),
                torch.tensor(v2, dtype=torch.float32))

class VideoSSLModel(nn.Module):
    def __init__(self, d=512, proj_dim=128, drop=0.3):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(3584, d), nn.LayerNorm(d), nn.GELU())
        self.se      = SEBlock(d)
        self.pos     = nn.Parameter(torch.randn(1, 16, d))
        enc          = nn.TransformerEncoderLayer(d, 8, d*2, drop, batch_first=True)
        self.tfm     = nn.TransformerEncoder(enc, 2)
        self.attn    = nn.Linear(d, 1)
        self.fuse    = nn.Linear(d*3, d)
        self.proj    = ProjectionHead(d, proj_dim)

    def _pool(self, x):
        return (x * F.softmax(self.attn(x), 1)).sum(1)

    def encode(self, x):
        x = self.tfm(self.se(self.proj_in(x)) + self.pos)
        return self.fuse(torch.cat([
            self._pool(x), self._pool(x[:,4:12,:]), self._pool(x[:,6:10,:])
        ], -1))

    def forward(self, x):
        return self.proj(self.encode(x))

def train_video_ssl(pool):
    sep("PHASE 1-C -- VIDEO SSL (SimCLR | MSW Transformer)")
    ckpt = SSL_DIR / "video_ssl.pt"
    if ckpt.exists():
        print("  [SKIP] video_ssl.pt already cached — delete to retrain.")
        return
    set_seed(42)
    ds      = VideoSSLDS(pool)
    dl      = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    m       = VideoSSLModel(proj_dim=SSL_PROJ_DIM).to(DEVICE)
    opt     = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
    loss_fn = InfoNCELoss(SSL_TEMP)
    print(f"  Pool: {len(pool)} | Batches/ep: {len(dl)} | Epochs: {SSL_EPOCHS}")
    for ep in range(1, SSL_EPOCHS+1):
        m.train(); ep_loss = 0.0
        for v1, v2 in dl:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(m(v1), m(v2))
            loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{SSL_EPOCHS} | InfoNCE: {ep_loss/len(dl):.4f}")
    # Save encoder only (not the projection head)
    torch.save({k: v for k,v in m.state_dict().items()
                if not k.startswith('proj.')}, str(ckpt))
    print(f"  [SAVED] {ckpt}")
    del m; torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
# PHASE 1-D — CROSS-MODAL SSL (Audio ↔ Video InfoNCE)
# ─────────────────────────────────────────────────────────
class CrossModalPairDS(Dataset):
    """Paired (audio_waveform, video_seq) for the same utterance.
    Works with labelled or unlabelled rows — no emotion column required.
    """
    def __init__(self, df, sr=16000, maxlen=80000):
        rows = []
        for _, row in df.iterrows():
            sid = str(row['sample_id']).replace("::","__").replace("/","_").replace(".mp4","")
            pc, _, _ = get_vid_paths(sid)
            if pc is not None and pc.exists() and resolve_audio_path(row) is not None:
                rows.append(row)
        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.sr = sr; self.maxlen = maxlen
        print(f"  Cross-modal valid pairs: {len(self.df)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # Audio
        try:
            p = resolve_audio_path(r)
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y = y[:self.maxlen] if len(y) > self.maxlen else np.pad(y, (0, self.maxlen-len(y)))
        except: y = np.zeros(self.maxlen, dtype=np.float32)
        # Video
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd_, pr = get_vid_paths(sid)
        try:
            seq = np.concatenate([np.load(pc), np.load(pd_), np.load(pr)], -1).astype(np.float32)
        except: seq = np.zeros((16, 3584), dtype=np.float32)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(seq, dtype=torch.float32)


def train_cross_modal_ssl(pool, unlabelled_df=None):
    """Phase 1-D: Audio-Video cross-modal InfoNCE alignment.

    Combines the labelled pool (~606 pairs) with up to CM_UNLABELLED_CAP
    unlabelled samples that have both audio and video.  Starts from per-modal
    SSL checkpoints; lightly fine-tunes WavLM layers 9-11 (lr=2e-6) and the
    full video encoder (lr=1e-5) via new cross-modal projection heads.
    Saves audio_cm_ssl.pt / video_cm_ssl.pt — FT auto-prefers these over
    the per-modal checkpoints when they exist.
    """
    sep("PHASE 1-D -- CROSS-MODAL SSL (Audio-Video | labelled + unlabelled)")
    ckpt_a = SSL_DIR / "audio_cm_ssl.pt"
    ckpt_v = SSL_DIR / "video_cm_ssl.pt"
    if ckpt_a.exists() and ckpt_v.exists():
        print("  [SKIP] Cross-modal checkpoints cached — delete to retrain.")
        return
    set_seed(42)

    # Combine labelled pool with small unlabelled sample
    if unlabelled_df is not None and len(unlabelled_df) > 0:
        combined = pd.concat([pool, unlabelled_df], ignore_index=True)
        print(f"  Pool: {len(pool)} labelled + {len(unlabelled_df)} unlabelled = {len(combined)} total")
    else:
        combined = pool
        print(f"  Pool: {len(pool)} labelled only (no unlabelled.csv found)")

    ds = CrossModalPairDS(combined)
    if len(ds) < 8:
        print(f"  [SKIP] Only {len(ds)} valid pairs — need >= 8.")
        return
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True,
                    num_workers=2, pin_memory=True)

    # ── Audio encoder: start from per-modal SSL checkpoint ─────────────────
    a_enc = AudioSSLModel(SSL_PROJ_DIM).to(DEVICE)
    if (SSL_DIR / "audio_ssl.pt").exists():
        sd = torch.load(SSL_DIR / "audio_ssl.pt", map_location=DEVICE)
        a_enc.backbone.load_state_dict(sd['backbone'])
        a_enc.lw.data = sd['lw']
        print("  Audio: loaded audio_ssl.pt")
    else:
        print("  Audio: no per-modal checkpoint — using WavLM pretrained weights")

    # Freeze all WavLM; unfreeze only top-3 layers (9-11) for cross-modal adaptation
    for p in a_enc.backbone.parameters(): p.requires_grad = False
    a_enc.lw.requires_grad = True
    a_top3 = []
    for i in [9, 10, 11]:
        for p in a_enc.backbone.encoder.layers[i].parameters():
            p.requires_grad = True
            a_top3.append(p)

    # ── Video encoder: start from per-modal SSL checkpoint ─────────────────
    v_enc = VideoSSLModel(proj_dim=SSL_PROJ_DIM, drop=0.1).to(DEVICE)
    if (SSL_DIR / "video_ssl.pt").exists():
        v_enc.load_state_dict(
            torch.load(SSL_DIR / "video_ssl.pt", map_location=DEVICE), strict=False)
        print("  Video: loaded video_ssl.pt")
    else:
        print("  Video: no per-modal checkpoint — using random init")

    # ── Cross-modal projection heads (new, randomly initialised) ───────────
    a_cm_proj = ProjectionHead(768*2, SSL_PROJ_DIM).to(DEVICE)
    v_cm_proj = ProjectionHead(512,   SSL_PROJ_DIM).to(DEVICE)
    proj_params = list(a_cm_proj.parameters()) + list(v_cm_proj.parameters())

    opt = torch.optim.AdamW([
        {'params': [a_enc.lw],        'lr': 1e-3,  'weight_decay': 0.01},
        {'params': a_top3,             'lr': 2e-6,  'weight_decay': 0.01},
        {'params': v_enc.parameters(), 'lr': 1e-5,  'weight_decay': 0.01},
        {'params': proj_params,        'lr': 1e-3,  'weight_decay': 0.01},
    ])
    sch        = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CM_SSL_EPOCHS)
    loss_fn    = InfoNCELoss(CM_TEMP)
    scaler     = GradScaler()
    clip_params = a_top3 + [a_enc.lw] + list(v_enc.parameters()) + proj_params

    print(f"  Pairs: {len(ds)} | Batch: 16 | Epochs: {CM_SSL_EPOCHS} | Temp: {CM_TEMP}")
    for ep in range(1, CM_SSL_EPOCHS+1):
        a_enc.train(); v_enc.train(); a_cm_proj.train(); v_cm_proj.train()
        ep_loss = 0.0
        for aud, vid in dl:
            aud, vid = aud.to(DEVICE), vid.to(DEVICE)
            opt.zero_grad()
            with autocast("cuda"):
                z_a = a_cm_proj(a_enc.encode(aud))
                z_v = v_cm_proj(v_enc.encode(vid))
                loss = loss_fn(z_a, z_v)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{CM_SSL_EPOCHS} | A-V InfoNCE: {ep_loss/len(dl):.4f}")

    torch.save({'backbone': a_enc.backbone.state_dict(), 'lw': a_enc.lw.data}, str(ckpt_a))
    torch.save({k: v for k,v in v_enc.state_dict().items()
                if not k.startswith('proj.')}, str(ckpt_v))
    print(f"  [SAVED] {ckpt_a.name}, {ckpt_v.name}")
    del a_enc, v_enc, a_cm_proj, v_cm_proj
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
# PHASE 2 — SUPERVISED FINE-TUNING (FT) WITH SUPCON
# ─────────────────────────────────────────────────────────

# FT Datasets
class AudioFTDS(Dataset):
    def __init__(self, df, sr=16000, maxlen=80000):
        self.df = df.reset_index(drop=True)
        self.sr = sr; self.maxlen = maxlen
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            p = resolve_audio_path(r)
            if p is None: raise FileNotFoundError
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y    = y[:self.maxlen] if len(y)>self.maxlen else np.pad(y,(0,self.maxlen-len(y)))
        except: y = np.zeros(self.maxlen)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

class TextFTDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc    = tok([clean(str(t)) for t in texts], truncation=True,
                          padding="max_length", max_length=64, return_tensors="pt")
        self.labels = [LID[l] for l in labels]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {k: v[i] for k,v in self.enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)

class VideoFTDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r   = self.df.iloc[i]
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd_, pr = get_vid_paths(sid)
        seq = np.concatenate([np.load(pc), np.load(pd_), np.load(pr)], -1)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)


# FT Models
class AudioFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone   = AudioSSLModel(proj_dim)
        self.classifier = nn.Sequential(nn.Linear(768*2, 512), nn.LayerNorm(512),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 7))
        self.proj_ft    = ProjectionHead(768*2, proj_dim)
    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.classifier(feat), self.proj_ft(feat)

class TextFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone   = TextSSLModel(proj_dim)
        self.classifier = nn.Linear(768*3, 7)
        self.drops      = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.proj_ft    = ProjectionHead(768*3, proj_dim)
    def forward(self, ids, mask):
        feat   = self.backbone.encode(ids, mask)
        logits = torch.stack([self.classifier(d(feat)) for d in self.drops]).mean(0)
        return logits, self.proj_ft(feat)

class VideoFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone   = VideoSSLModel(proj_dim=proj_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.3),
                                        nn.Linear(512, 256), nn.GELU(),
                                        nn.Dropout(0.3), nn.Linear(256, 7))
        self.proj_ft    = ProjectionHead(512, proj_dim)
    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.classifier(feat), self.proj_ft(feat)


# FT Training Loop
def train_modality_ft(name, train_df, val_df, test_df, use_ssl=True, use_supcon=True):
    sep(f"PHASE 2 -- {name} FT (SSL={use_ssl}, SupCon={use_supcon})")
    set_seed(42)

    if name == "AUDIO":
        ds_tr = AudioFTDS(train_df)
        ds_va = AudioFTDS(val_df)
        ds_te = AudioFTDS(test_df)
        m = AudioFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            # Prefer cross-modal checkpoint; fall back to per-modal SSL
            sd_path = (SSL_DIR/"audio_cm_ssl.pt" if (SSL_DIR/"audio_cm_ssl.pt").exists()
                       else SSL_DIR/"audio_ssl.pt")
            if sd_path.exists():
                sd = torch.load(sd_path, map_location=DEVICE)
                m.backbone.backbone.load_state_dict(sd['backbone'])
                m.backbone.lw.data = sd['lw']
                print(f"  [SSL] loaded {sd_path.name}")
        supcon_temp = 0.1    # batch=8, ~7 negatives — warmer than 0.07 for stability
        lr_bb, lr_hd, bs = 1e-5, 1e-3, 8

    elif name == "TEXT":
        tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
        ds_tr = TextFTDS(train_df['transcript'].values, train_df['emotion_final'].values, tok)
        ds_va = TextFTDS(val_df['transcript'].values,   val_df['emotion_final'].values,   tok)
        ds_te = TextFTDS(test_df['transcript'].values,  test_df['emotion_final'].values,  tok)
        m = TextFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl and (SSL_DIR/"text_ssl.pt").exists():
            m.backbone.load_state_dict(
                torch.load(SSL_DIR/"text_ssl.pt", map_location=DEVICE), strict=False)
            print("  [SSL] loaded text_ssl.pt")
        supcon_temp = 0.12   # batch=16, ~15 negatives
        lr_bb, lr_hd, bs = 1e-5, 5e-4, 16

    else:  # VIDEO
        ds_tr = VideoFTDS(train_df)
        ds_va = VideoFTDS(val_df)
        ds_te = VideoFTDS(test_df)
        m = VideoFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            sd_path = (SSL_DIR/"video_cm_ssl.pt" if (SSL_DIR/"video_cm_ssl.pt").exists()
                       else SSL_DIR/"video_ssl.pt")
            if sd_path.exists():
                m.backbone.load_state_dict(
                    torch.load(sd_path, map_location=DEVICE), strict=False)
                print(f"  [SSL] loaded {sd_path.name}")
        supcon_temp = 0.07   # batch=32, ~31 negatives
        lr_bb, lr_hd, bs = 3e-5, 1e-3, 32

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=bs)
    dl_te = DataLoader(ds_te, batch_size=bs)

    # Soft-freeze backbone via 100x lower LR (stable with OneCycleLR unlike hard-freeze)
    bb_params = [p for n, p in m.named_parameters() if 'backbone' in n]
    hd_params = [p for n, p in m.named_parameters() if 'backbone' not in n]
    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': lr_bb},
        {'params': hd_params, 'lr': lr_hd}
    ], weight_decay=0.05)

    sch       = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_bb, lr_hd], steps_per_epoch=len(dl_tr), epochs=20)
    supcon_fn = SupConLoss(supcon_temp)
    best_f1, ckpt = 0.0, SAVE_DIR / f"{name.lower()}_ft.pt"

    y_tr      = np.array([LID[e] for e in train_df['emotion_final']])
    cw        = compute_class_weight('balanced', classes=np.arange(7), y=y_tr)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    for ep in range(1, 21):
        m.train()
        cur_supcon_w = 0.3 if use_supcon else 0.0
        for batch in dl_tr:
            opt.zero_grad()
            if name == "TEXT":
                logits, proj = m(batch[0]['input_ids'].to(DEVICE),
                                 batch[0]['attention_mask'].to(DEVICE))
            else:
                logits, proj = m(batch[0].to(DEVICE))
            labels = batch[1].to(DEVICE)
            loss = F.cross_entropy(logits, labels, weight=cw_tensor, label_smoothing=0.1)
            if use_supcon:
                loss = loss + cur_supcon_w * supcon_fn(proj, labels)
            loss.backward(); opt.step(); sch.step()

        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for batch in dl_va:
                if name == "TEXT":
                    logits, _ = m(batch[0]['input_ids'].to(DEVICE),
                                  batch[0]['attention_mask'].to(DEVICE))
                else:
                    logits, _ = m(batch[0].to(DEVICE))
                ps.extend(logits.argmax(1).cpu().numpy())
                ts.extend(batch[1].numpy())
        acc = accuracy_score(ts, ps)
        f1  = f1_score(ts, ps, average='macro', zero_division=0)
        improved = f1 > best_f1
        if improved: best_f1 = f1; torch.save(m.state_dict(), str(ckpt))
        if ep % 2 == 0 or ep == 1:
            print(f"  Ep {ep:02d} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
                  f"{' *' if improved else ''}")

    m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    m.eval()

    def _infer(dl):
        probs = []
        with torch.no_grad():
            for batch in dl:
                if name == "TEXT":
                    logits, _ = m(batch[0]['input_ids'].to(DEVICE),
                                  batch[0]['attention_mask'].to(DEVICE))
                else:
                    logits, _ = m(batch[0].to(DEVICE))
                probs.append(F.softmax(logits, 1).cpu().numpy())
        return np.vstack(probs)

    test_probs = _infer(dl_te)
    val_probs  = _infer(dl_va)

    t_diag = np.array([LID[e] for e in test_df['emotion_final']])
    print(f"  [{name}] Standalone — "
          f"Acc={accuracy_score(t_diag, test_probs.argmax(1)):.4f}  "
          f"F1={f1_score(t_diag, test_probs.argmax(1), average='macro', zero_division=0):.4f}")

    return test_probs, val_probs


# ─────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────
def run_ablation(tr, va, te):
    scenarios = [
        {"name": "Baseline",     "ssl": False, "supcon": False},
        {"name": "SupCon only",  "ssl": False, "supcon": True},
        {"name": "SSL only",     "ssl": True,  "supcon": False},
        {"name": "SSL + SupCon", "ssl": True,  "supcon": True},
    ]
    results  = []
    v_labels = np.array([LID[e] for e in va['emotion_final'].values])   # fusion search
    t_labels = np.array([LID[e] for e in te['emotion_final'].values])   # final reporting

    for sc in scenarios:
        sep(f"RUNNING SCENARIO: {sc['name']}")
        vp_te, vp_va = train_modality_ft("VIDEO", tr, va, te, sc['ssl'], sc['supcon'])
        ap_te, ap_va = train_modality_ft("AUDIO", tr, va, te, sc['ssl'], sc['supcon'])
        tp_te, tp_va = train_modality_ft("TEXT",  tr, va, te, sc['ssl'], sc['supcon'])

        # Grid search on VAL macro F1 (test labels never touched here)
        best_f1_val, best_w = 0.0, (0.33, 0.33, 0.34)
        for w_v in np.linspace(0, 1, 11):
            for w_a in np.linspace(0, 1, 11):
                w_t = round(1.0 - w_v - w_a, 8)
                if w_t < 0 or w_t > 1: continue
                fp  = w_v*vp_va + w_a*ap_va + w_t*tp_va
                f1  = f1_score(v_labels, fp.argmax(1), average='macro', zero_division=0)
                if f1 > best_f1_val:
                    best_f1_val, best_w = f1, (w_v, w_a, w_t)

        # Apply best val weights to TEST
        w_v, w_a, w_t = best_w
        fp_te  = w_v*vp_te + w_a*ap_te + w_t*tp_te
        te_acc = accuracy_score(t_labels, fp_te.argmax(1))
        te_f1  = f1_score(t_labels, fp_te.argmax(1), average='macro', zero_division=0)

        results.append({
            "Scenario": sc['name'],
            "Test Acc": round(te_acc, 4),
            "Test F1":  round(te_f1,  4),
            "Weights":  f"V={w_v:.2f} A={w_a:.2f} T={w_t:.2f}",
        })
        print(f"\n  >>> {sc['name']}  Acc={te_acc:.4f}  F1={te_f1:.4f}  "
              f"w=({w_v:.2f},{w_a:.2f},{w_t:.2f})")

    sep("FINAL ABLATION RESULTS")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_results.csv", index=False)
    print("\n  (Fusion weights chosen on val set — test labels never seen during search)")


# ─────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sep(f"CONTRASTIVE PIPELINE v2 | Device: {DEVICE}")
    tr, va, te = load_splits()

    # SSL pool = labelled train + val
    pool = pd.concat([tr, va]).reset_index(drop=True)
    print(f"  SSL labelled pool: {len(pool)} samples (train + val)")

    # Phase 1: per-modal SSL on labelled pool
    train_audio_ssl(pool)
    train_text_ssl(pool)
    train_video_ssl(pool)

    # Phase 1-D: cross-modal SSL with small unlabelled augmentation
    unlabelled_df = load_unlabelled()
    train_cross_modal_ssl(pool, unlabelled_df)

    sep("PHASE 1 COMPLETE -- All encoders pre-trained.")

    # Phase 2: supervised fine-tuning ablation
    run_ablation(tr, va, te)
