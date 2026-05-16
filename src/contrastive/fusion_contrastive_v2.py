"""
fusion_contrastive_v2.py — 2-Phase Contrastive Pipeline (Colab)
================================================================
PHASE 1  Self-Supervised Pre-training  (InfoNCE, NO labels)
  Audio : SimCLR with waveform augmentation
  Text  : SimCSE (same sentence, two dropout masks)
  Video : Feature SimCLR (frame masking + noise on [16 x 3584])

PHASE 2  Supervised Fine-tuning  (CE + SupCon, WITH labels)  <- TODO next session

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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import WavLMModel, AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS & GLOBAL CONFIG (Smart Auto-Detection)
# ─────────────────────────────────────────────────────────
REPO       = Path("/content/thesis")
SPLIT_DIR  = REPO / "data/processed/splits/multimodal_eligible"
SAVE_DIR   = Path("/content/fusion_models")
SSL_DIR    = Path("/content/ssl_pretrained")
for d in [SAVE_DIR, SSL_DIR]: d.mkdir(exist_ok=True)

def auto_detect():
    print("  Smart-detecting data locations...")
    v_dir, a_dir = None, None
    
    # 1. Search /content first (fast)
    for p in Path("/content").rglob("*_clip_seq.npy"):
        if "drive" not in str(p):
            v_dir = p.parent
            break
            
    # 2. Fallback to Drive
    if not v_dir:
        v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features/video_sequences_v1")
        if not v_dir.exists(): v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features")
        
    # 3. Audio
    a_dir = Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED")
    
    return v_dir, a_dir

VID_DIR, AUDIO_BASE = auto_detect()
print(f"  Final VID_DIR: {VID_DIR}")
print(f"  Final AUDIO_BASE: {AUDIO_BASE}")

print("  [DEBUG] Sample files in VID_DIR:")
if VID_DIR.exists():
    for f in list(VID_DIR.glob("*_clip_seq.npy"))[:3]: print(f"    - {f.name}")
else: print("    (Directory does not exist)")

print("  [DEBUG] Sample files in AUDIO_BASE:")
if AUDIO_BASE.exists():
    for f in list(AUDIO_BASE.rglob("*.wav"))[:3]: print(f"    - {f.name}")
else: print("    (Directory does not exist)")

LID     = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6}
CLASSES = list(LID.keys())
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# SSL hyper-parameters
SSL_EPOCHS   = 40
SSL_TEMP     = 0.07
SSL_PROJ_DIM = 128
GRAD_ACC     = 4      # effective audio batch = 8 * 4 = 32

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
    
    # Try detected AUDIO_BASE or /content/audio
    for base in [AUDIO_BASE, Path("/content/audio"), Path("/content/audio/Thesis Project/dataset/Final Modalink Dataset MERGED")]:
        p = base / folder / audio_rel if folder else base / audio_rel
        if p.exists(): return p
        
    return None

def load_splits():
    if not SPLIT_DIR.exists():
        print(f"  [ERROR] Split directory not found: {SPLIT_DIR}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    tr = pd.read_csv(SPLIT_DIR/"train.csv")
    va = pd.read_csv(SPLIT_DIR/"val.csv")
    te = pd.read_csv(SPLIT_DIR/"test.csv")
    
    sep("🔍 PATH DIAGNOSTIC")
    row0 = tr.iloc[0]
    sid0 = row0['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
    v_test = VID_DIR / f"{sid0}_clip_seq.npy"
    a_test = resolve_audio_path(row0)
    
    vid_ok = v_test.exists()
    aud_ok = a_test is not None and a_test.exists()
    txt_ok = isinstance(row0.get('transcript'), str) and len(str(row0['transcript']).strip()) > 2
    
    print(f"  Sample ID: {sid0}")
    print(f"  Video Status: {'✅ OK' if vid_ok else '❌ MISSING'} ({v_test.name})")
    print(f"  Audio Status: {'✅ OK' if aud_ok else '❌ MISSING'} ({a_test.name if a_test else 'N/A'})")
    print(f"  Text  Status: {'✅ OK' if txt_ok else '❌ EMPTY'}")
    
    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        vid = (VID_DIR/f"{sid}_clip_seq.npy").exists()
        aud = resolve_audio_path(row) is not None
        txt = isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2
        return vid and aud and txt

    tr_f = tr[tr.apply(ok, axis=1)].reset_index(drop=True)
    va_f = va[va.apply(ok, axis=1)].reset_index(drop=True)
    te_f = te[te.apply(ok, axis=1)].reset_index(drop=True)
    
    sep("ALIGNED SPLITS")
    print(f"  Train: {len(tr_f)} | Val: {len(va_f)} | Test: {len(te_f)}")
    if len(tr_f) == 0:
        print("  [CRITICAL] 0 samples aligned. Check if VID_DIR and AUDIO_BASE are correct.")
        print(f"  VID_DIR currently: {VID_DIR}")
        print(f"  AUDIO_BASE currently: {AUDIO_BASE}")
        
    return tr_f, va_f, te_f

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
        mask_self = torch.eye(B, device=features.device).bool()
        sim.masked_fill_(mask_self, float('-inf'))
        pos_mask = labels.view(-1,1).eq(labels.view(1,-1)).float()
        pos_mask.fill_diagonal_(0)
        log_prob = F.log_softmax(sim, dim=1)
        n_pos    = pos_mask.sum(1).clamp(min=1)
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
    m   = AudioSSLModel(SSL_PROJ_DIM).to(DEVICE)
    # Freeze bottom 6 WavLM layers — fine-tune top 6 only
    for i, layer in enumerate(m.backbone.encoder.layers):
        if i < 6:
            for p in layer.parameters(): p.requires_grad = False
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

# ─────────────────────────────────────────────────────────
# PHASE 1-B — TEXT SSL (SimCSE)
# ─────────────────────────────────────────────────────────
MODEL_NAME = "UBC-NLP/MARBERT"
_FILLERS   = re.compile(
    r'\b(اه|ايه|يعني|بص|كده|كدا|اهو|والله|عشان|بقا|بقى|يا|اوه|هاه|اوكي|اوكى|تمام|صح|ايوه|لا|مش|ميش)\b'
)
def clean(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)
    t = re.sub(r'[\u0622\u0623\u0625]', '\u0627', t)
    t = re.sub(r'\u0629', '\u0647', t)
    t = re.sub(r'\u0649', '\u064A', t)
    t = re.sub(r'\u0640', '', t)
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
    tok     = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds      = TextSSLDS(pool['transcript'].values, tok)
    dl      = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    m       = TextSSLModel(SSL_PROJ_DIM).to(DEVICE)
    opt     = torch.optim.AdamW([
        {'params': [p for n,p in m.named_parameters() if 'bert' in n and p.requires_grad], 'lr': 2e-5},
        {'params': [p for n,p in m.named_parameters() if 'proj' in n],                    'lr': 1e-3}
    ], weight_decay=0.01)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
    loss_fn = InfoNCELoss(SSL_TEMP)
    print(f"  Pool: {len(pool)} | Batches/ep: {len(dl)} | Epochs: {SSL_EPOCHS}")
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
    # Save bert weights only (drop projection head)
    torch.save({k: v for k,v in m.state_dict().items() if 'proj' not in k}, str(ckpt))
    print(f"  [SAVED] {ckpt}")

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
        c   = np.load(VID_DIR/f"{sid}_clip_seq.npy")
        d   = np.load(VID_DIR/f"{sid}_dinov2_seq.npy")
        r2  = np.load(VID_DIR/f"{sid}_resnet50_seq.npy")
        seq = np.concatenate([c, d, r2], -1)   # [16, 3584]
        v1, v2 = video_augment(seq)
        return v1, v2

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
    # Save encoder (no projection head)
    torch.save({k: v for k,v in m.state_dict().items() if 'proj' not in k}, str(ckpt))
    print(f"  [SAVED] {ckpt}")

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
        self.enc = tok([clean(str(t)) for t in texts], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
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
        c   = np.load(VID_DIR/f"{sid}_clip_seq.npy")
        d   = np.load(VID_DIR/f"{sid}_dinov2_seq.npy")
        r2  = np.load(VID_DIR/f"{sid}_resnet50_seq.npy")
        seq = np.concatenate([c, d, r2], -1)   # [16, 3584]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

# FT Models
class AudioFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = AudioSSLModel(proj_dim) # Re-use backbone + lw
        self.classifier = nn.Sequential(nn.Linear(768*2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 7))
        self.proj_ft = ProjectionHead(768*2, proj_dim) # Separate proj head for SupCon
    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.classifier(feat), self.proj_ft(feat)

class TextFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = TextSSLModel(proj_dim)
        self.classifier = nn.Linear(768*3, 7)
        self.drops = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.proj_ft = ProjectionHead(768*3, proj_dim)
    def forward(self, ids, mask):
        feat = self.backbone.encode(ids, mask)
        logits = torch.stack([self.classifier(d(feat)) for d in self.drops]).mean(0)
        return logits, self.proj_ft(feat)

class VideoFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = VideoSSLModel(proj_dim=proj_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.3), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 7))
        self.proj_ft = ProjectionHead(512, proj_dim)
    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.classifier(feat), self.proj_ft(feat)

# FT Training Loops
def train_modality_ft(name, train_df, val_df, test_df, use_ssl=True, use_supcon=True):
    sep(f"PHASE 2 -- {name} FT (SSL={use_ssl}, SupCon={use_supcon})")
    set_seed(42)
    
    if name == "AUDIO":
        ds_tr, ds_va, ds_te = AudioFTDS(train_df), AudioFTDS(val_df), AudioFTDS(test_df)
        m = AudioFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl: 
            sd = torch.load(SSL_DIR/"audio_ssl.pt", map_location=DEVICE)
            m.backbone.backbone.load_state_dict(sd['backbone'])
            m.backbone.lw.data = sd['lw']
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4) # Simplified for brevity
        bs = 8
    elif name == "TEXT":
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        ds_tr = TextFTDS(train_df['transcript'].values, train_df['emotion_final'].values, tok)
        ds_va = TextFTDS(val_df['transcript'].values, val_df['emotion_final'].values, tok)
        ds_te = TextFTDS(test_df['transcript'].values, test_df['emotion_final'].values, tok)
        m = TextFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl: m.backbone.load_state_dict(torch.load(SSL_DIR/"text_ssl.pt", map_location=DEVICE), strict=False)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
        bs = 16
    else: # VIDEO
        ds_tr, ds_va, ds_te = VideoFTDS(train_df), VideoFTDS(val_df), VideoFTDS(test_df)
        m = VideoFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl: m.backbone.load_state_dict(torch.load(SSL_DIR/"video_ssl.pt", map_location=DEVICE), strict=False)
        opt = torch.optim.AdamW(m.parameters(), lr=7e-5)
        bs = 32

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=bs)
    dl_te = DataLoader(ds_te, batch_size=bs)
    
    supcon_fn = SupConLoss(SSL_TEMP)
    best_acc, ckpt = 0, SAVE_DIR/f"{name.lower()}_ft.pt"
    
    for ep in range(1, 21):
        m.train()
        for batch in dl_tr:
            opt.zero_grad()
            if name == "TEXT":
                logits, proj = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
            else:
                logits, proj = m(batch[0].to(DEVICE))
            labels = batch[1].to(DEVICE)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            if use_supcon: loss += 0.3 * supcon_fn(proj, labels)
            loss.backward(); opt.step()
        
        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for batch in dl_va:
                if name == "TEXT": logits, _ = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
                else: logits, _ = m(batch[0].to(DEVICE))
                ps.extend(logits.argmax(1).cpu().numpy()); ts.extend(batch[1].numpy())
        acc = accuracy_score(ts, ps)
        if acc > best_acc: best_acc = acc; torch.save(m.state_dict(), str(ckpt))
        if ep % 5 == 0: print(f"  Ep {ep:02d} | Val Acc: {acc:.4f}")

    m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    m.eval(); probs = []
    with torch.no_grad():
        for batch in dl_te:
            if name == "TEXT": logits, _ = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
            else: logits, _ = m(batch[0].to(DEVICE))
            probs.append(F.softmax(logits, 1).cpu().numpy())
    return np.vstack(probs)

# ─────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────
def run_ablation(tr, va, te):
    scenarios = [
        {"name": "Baseline",    "ssl": False, "supcon": False},
        {"name": "SupCon only", "ssl": False, "supcon": True},
        {"name": "SSL only",     "ssl": True,  "supcon": False},
        {"name": "SSL + SupCon", "ssl": True,  "supcon": True},
    ]
    results = []
    t_labels = [LID[e] for e in te['emotion_final'].values]
    
    for sc in scenarios:
        sep(f"RUNNING SCENARIO: {sc['name']}")
        vp = train_modality_ft("VIDEO", tr, va, te, sc['ssl'], sc['supcon'])
        ap = train_modality_ft("AUDIO", tr, va, te, sc['ssl'], sc['supcon'])
        tp = train_modality_ft("TEXT",  tr, va, te, sc['ssl'], sc['supcon'])
        
        # Simple Mean Fusion for Ablation Comparison
        fp = (vp + ap + tp) / 3.0
        acc = accuracy_score(t_labels, fp.argmax(1))
        f1 = f1_score(t_labels, fp.argmax(1), average='macro')
        results.append({"Scenario": sc['name'], "Acc": acc, "F1": f1})
        print(f"\n  >>> {sc['name']} Result: Acc={acc:.4f}, F1={f1:.4f}")

    sep("FINAL ABLATION RESULTS")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_results.csv", index=False)

# ─────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sep(f"CONTRASTIVE PIPELINE v2 | Device: {DEVICE}")
    tr, va, te = load_splits()
    
    # 1. PHASE 1: SELF-SUPERVISED PRE-TRAINING
    pool = pd.concat([tr, va]).reset_index(drop=True)
    print(f"  SSL training pool: {len(pool)} samples (train + val)")
    
    train_video_ssl(pool)
    train_text_ssl(pool)
    train_audio_ssl(pool)
    
    sep("PHASE 1 COMPLETE -- Encoders pre-trained.")
    
    # 2. PHASE 2: ABLATION (Fine-tuning)
    run_ablation(tr, va, te)
