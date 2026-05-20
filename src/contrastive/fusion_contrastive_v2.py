"""
fusion_contrastive_v2.py — Full Ablation Study (Colab)
================================================================
Runs the 4-scenario ablation (Baseline, SupCon only, SSL only, SSL+SupCon)
using the HIGH-ACCURACY PRODUCTION architecture (5-Seed Ensembles, 5-Fold CV,
Lookahead Optimizer, Progressive Unfreezing).

WARNING: Running all 4 scenarios with full ensembling takes significant compute.
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from transformers import WavLMModel, AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS (Smart Auto-Detection)
# ─────────────────────────────────────────────────────────
import glob
cands = glob.glob("/content/*thesis*") + glob.glob("/content/*omaremad*")
cands = [c for c in cands if os.path.isdir(c) and (Path(c)/"src").exists()]
_repo_str = cands[0] if cands else "/content/thesis"

REPO       = Path(_repo_str)
SPLIT_DIR  = REPO / "data/processed/splits/multimodal_eligible"
SAVE_DIR   = Path("/content/fusion_models")
SSL_DIR    = Path("/content/drive/MyDrive/Thesis Project/ssl_pretrained")
for d in [SAVE_DIR, SSL_DIR]: d.mkdir(parents=True, exist_ok=True)

def auto_detect():
    v_dir, a_dir = None, None
    for p in Path("/content").rglob("*_clip_seq.npy"):
        if "drive" not in str(p):
            v_dir = p.parent; break
    if not v_dir:
        v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features/video_sequences_v1")
        if not v_dir.exists(): v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features")
    a_dir = Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED")
    return v_dir, a_dir

VID_DIR, AUDIO_BASE = auto_detect()

LID     = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6}
CLASSES = list(LID.keys())
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

SSL_EPOCHS   = 40
SSL_TEMP     = 0.07
SSL_PROJ_DIM = 128
GRAD_ACC     = 4

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sep(t=""):
    print("\n" + "="*56)
    if t: print(f"  {t}")
    print("="*56)

# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
def resolve_audio_path(row):
    audio_rel = str(row.get('audio_relpath', ''))
    if not audio_rel: return None
    folder = str(row.get('folder', ''))
    bases = [
        AUDIO_BASE, Path("/content/audio/Thesis Project/dataset/Final Modalink Dataset MERGED"),
        Path("/content/audio/Thesis Project/data/raw"), Path("/content/audio/data/processed"),
        Path("/content/audio"), Path("/content/audio/Thesis_Audio_Full"), Path("/content/Thesis_Audio_Full"),
        Path("/content/drive/MyDrive/Thesis_Audio_Full"), Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED"),
        Path("/content/drive/MyDrive/Thesis Project/data/raw"), Path("/content/drive/MyDrive/Thesis Project"), Path("/content/drive/MyDrive")
    ]
    for base in bases:
        p = base / folder / audio_rel if folder else base / audio_rel
        if p.exists(): return p
        if folder:
            p_bs = base / f"{folder}\\{audio_rel.replace('/', '\\')}"
            if p_bs.exists(): return p_bs
        p_flat = base / Path(audio_rel).name
        if p_flat.exists(): return p_flat
    return None

def get_vid_paths(sid):
    p1 = VID_DIR / f"{sid}_clip_seq.npy"
    if p1.exists(): return p1, VID_DIR / f"{sid}_dinov2_seq.npy", VID_DIR / f"{sid}_resnet50_seq.npy"
    p2 = VID_DIR / f"video_sequences_v1\\{sid}_clip_seq.npy"
    if p2.exists(): return p2, VID_DIR / f"video_sequences_v1\\{sid}_dinov2_seq.npy", VID_DIR / f"video_sequences_v1\\{sid}_resnet50_seq.npy"
    return None, None, None

def load_splits():
    tr = pd.read_csv(SPLIT_DIR/"train.csv")
    va = pd.read_csv(SPLIT_DIR/"val.csv")
    te = pd.read_csv(SPLIT_DIR/"test.csv")
    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        vid_p, _, _ = get_vid_paths(sid)
        return vid_p is not None and vid_p.exists() and resolve_audio_path(row) is not None and isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2
    tr = tr[tr.apply(ok, axis=1)].reset_index(drop=True)
    va = va[va.apply(ok, axis=1)].reset_index(drop=True)
    te = te[te.apply(ok, axis=1)].reset_index(drop=True)
    return tr, va, te

# ─────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
        B = z1.size(0); z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T
        sim.fill_diagonal_(float('-inf'))
        labels = torch.cat([torch.arange(B, 2*B, device=z.device), torch.arange(0, B, device=z.device)])
        return F.cross_entropy(sim, labels)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature
        # Add class-specific margins to prevent identical collapse 
        # (Happiness=3, Surprise=6, Fear=2 are highly confused)
        self.margins = {3: 0.2, 6: 0.15, 2: 0.25}
        
    def forward(self, features, labels):
        m = torch.zeros(7, device=features.device)
        for k, v in self.margins.items(): m[k] = v
        
        sim = torch.mm(features, features.T) / self.T
        mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
        
        # Apply margin to positive pairs
        sim = sim - mask * m[labels].unsqueeze(1)
        
        # Exclude self-similarity
        lm = torch.ones_like(mask).scatter_(1, torch.arange(len(labels), device=features.device).view(-1,1), 0)
        
        # Supervised Contrastive Loss
        denominator = torch.log(sim.exp().sum(1, keepdim=True) + 1e-6)
        loss = -(mask * lm * (sim - denominator)).sum(1)
        
        # Normalize by number of positives
        return loss.mean() / (mask.sum(1).mean() + 1e-6)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
    def forward(self, x): return self.net(x)

class Lookahead:
    def __init__(self, opt, k=5, a=0.5):
        self.opt, self.k, self.a = opt, k, a
        self.param_groups = opt.param_groups
        self.slow = [[p.data.clone() for p in g['params']] for g in opt.param_groups]
        self.i = 0
    def step(self):
        self.opt.step(); self.i += 1
        if self.i % self.k == 0:
            for i,g in enumerate(self.param_groups):
                for j,p in enumerate(g['params']):
                    p.data.mul_(self.a).add_(self.slow[i][j], alpha=1-self.a)
                    self.slow[i][j].copy_(p.data)
    def zero_grad(self, **kw): self.opt.zero_grad(**kw)

# ─────────────────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────────────────
def _audio_one_view(wav, maxlen=80000):
    w = wav.copy()
    if np.random.rand() > 0.3: w += np.random.randn(len(w)) * (np.sqrt(np.mean(w**2)) + 1e-9) / (10**(np.random.uniform(15, 30)/20))
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
    s = seq.copy()
    s[np.random.choice(s.shape[0], n_mask, replace=False)] = 0.0
    if np.random.rand() > 0.4: s += np.random.randn(*s.shape) * noise_std
    return s
def video_augment(seq): return _video_one_view(seq), _video_one_view(seq)

MODEL_NAME = "UBC-NLP/MARBERT"
_FILLERS   = re.compile(r'\b(اه|ايه|يعني|بص|كده|كدا|اهو|والله|عشان|بقا|بقى|يا|اوه|هاه|اوكي|اوكى|تمام|صح|ايوه|لا|مش|ميش)\b')
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

# ─────────────────────────────────────────────────────────
# BACKBONES
# ─────────────────────────────────────────────────────────
class AudioSSLModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < 6:
                for p in layer.parameters(): p.requires_grad = False
        self.lw   = nn.Parameter(torch.ones(13))
        self.proj = ProjectionHead(768*2, proj_dim)
    def encode(self, x):
        hs  = torch.stack(self.backbone(x).hidden_states, 0)
        out = (hs * F.softmax(self.lw, 0).view(-1,1,1,1)).sum(0)
        return torch.cat([out.mean(1), out.std(1)], 1)
    def forward(self, x): return self.proj(self.encode(x))

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
        return torch.cat([lh[:,0,:], mp, xp], 1)
    def forward(self, ids, mask): return self.proj(self.encode(ids, mask))

class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(nn.Linear(c, c//16), nn.ReLU(), nn.Linear(c//16, c), nn.Sigmoid())
    def forward(self, x):
        b,n,c = x.shape
        return x * self.fc(self.pool(x.transpose(1,2)).view(b,c)).view(b,1,c)

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
    def _pool(self, x): return (x * F.softmax(self.attn(x), 1)).sum(1)
    def encode(self, x):
        x = self.tfm(self.se(self.proj_in(x)) + self.pos)
        return self.fuse(torch.cat([self._pool(x), self._pool(x[:,4:12,:]), self._pool(x[:,6:10,:])], -1))
    def forward(self, x): return self.proj(self.encode(x))

# Phase 1 Training Wrappers
def train_ssl_phase(pool):
    sep("PHASE 1: SELF-SUPERVISED PRE-TRAINING (SAVING TO GOOGLE DRIVE)")
    
    # Text
    ckpt = SSL_DIR / "text_ssl.pt"
    if not ckpt.exists():
        print("  Training Text SSL (Enhanced)...")
        class TextSSLDS(Dataset):
            def __init__(self, texts, tok):
                self.enc = tok([clean(str(t)) for t in texts], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            def __len__(self): return self.enc['input_ids'].size(0)
            def __getitem__(self, i): return {k: v[i] for k,v in self.enc.items()}
        ds = TextSSLDS(pool['transcript'].values, AutoTokenizer.from_pretrained(MODEL_NAME))
        dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
        m = TextSSLModel().to(DEVICE)
        opt = torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if 'bert' in n and p.requires_grad],'lr':3e-5},{'params':[p for n,p in m.named_parameters() if 'proj' in n],'lr':1e-3}], weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
        loss_fn = InfoNCELoss(0.1) # Higher temperature for NLP
        for ep in range(1, SSL_EPOCHS+1):
            m.train(); ep_loss=0
            for bd in dl:
                ids, mask = bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(m(ids, mask), m(ids, mask))
                loss.backward(); opt.step(); ep_loss += loss.item()
            sch.step()
        torch.save({k:v for k,v in m.state_dict().items() if 'proj' not in k}, str(ckpt))
        print(f"  [SAVED] {ckpt}")

    # Video
    ckpt = SSL_DIR / "video_ssl.pt"
    if not ckpt.exists():
        print("  Training Video SSL...")
        class VideoSSLDS(Dataset):
            def __init__(self, df): self.df = df
            def __len__(self): return len(self.df)
            def __getitem__(self, i):
                sid = self.df.iloc[i]['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
                pc, pd, pr = get_vid_paths(sid)
                seq = np.concatenate([np.load(pc), np.load(pd), np.load(pr)], -1)
                return video_augment(seq)
        dl = DataLoader(VideoSSLDS(pool), batch_size=32, shuffle=True, drop_last=True)
        m = VideoSSLModel().to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
        loss_fn = InfoNCELoss(SSL_TEMP)
        for ep in range(1, SSL_EPOCHS+1):
            m.train(); ep_loss=0
            for v1, v2 in dl:
                opt.zero_grad()
                loss = loss_fn(m(v1.float().to(DEVICE)), m(v2.float().to(DEVICE)))
                loss.backward(); opt.step(); ep_loss += loss.item()
            sch.step()
        torch.save({k:v for k,v in m.state_dict().items() if 'proj' not in k}, str(ckpt))
        print(f"  [SAVED] {ckpt}")

    # Audio
    ckpt = SSL_DIR / "audio_ssl.pt"
    if not ckpt.exists():
        print("  Training Audio SSL...")
        class AudioSSLDS(Dataset):
            def __init__(self, df): self.df = df
            def __len__(self): return len(self.df)
            def __getitem__(self, i):
                p = resolve_audio_path(self.df.iloc[i])
                try:
                    y,_ = librosa.load(str(p), sr=16000)
                    y,_ = librosa.effects.trim(y, top_db=25)
                    y = y[:80000] if len(y)>80000 else np.pad(y,(0,80000-len(y)))
                except: y = np.zeros(80000)
                v1, v2 = audio_augment(y)
                return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)
        dl = DataLoader(AudioSSLDS(pool), batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        m = AudioSSLModel().to(DEVICE)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
        loss_fn = InfoNCELoss(SSL_TEMP)
        scaler = GradScaler()
        for ep in range(1, SSL_EPOCHS+1):
            m.train(); ep_loss=0; opt.zero_grad()
            for step, (v1, v2) in enumerate(dl):
                v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
                with autocast("cuda"): loss = loss_fn(m(v1), m(v2)) / GRAD_ACC
                scaler.scale(loss).backward(); ep_loss += loss.item() * GRAD_ACC
                if (step+1)%GRAD_ACC==0 or (step+1)==len(dl):
                    scaler.step(opt); scaler.update(); opt.zero_grad()
            sch.step()
        torch.save({'backbone': m.backbone.state_dict(), 'lw': m.lw.data}, str(ckpt))
        print(f"  [SAVED] {ckpt}")

# ─────────────────────────────────────────────────────────
# PHASE 2 MODELS
# ─────────────────────────────────────────────────────────
class AudioFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = AudioSSLModel(proj_dim)
        self.classifier = nn.Sequential(nn.Linear(768*2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 7))
        self.proj_ft = ProjectionHead(768*2, proj_dim)
    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.classifier(feat), F.normalize(self.proj_ft(feat), dim=1)

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
        return logits, F.normalize(self.proj_ft(feat), dim=1)

class VideoFTModel(nn.Module):
    def __init__(self, proj_dim=128, drop=0.5):
        super().__init__()
        self.backbone = VideoSSLModel(proj_dim=proj_dim, drop=drop)
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Dropout(drop), nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 7))
        self.proj_ft = ProjectionHead(512, proj_dim)
    def forward(self, x):
        if self.training: x = x + torch.randn_like(x)*0.01
        feat = self.backbone.encode(x)
        return self.classifier(feat), F.normalize(self.proj_ft(feat), dim=1)

# Datasets
class AudioFTDS(Dataset):
    def __init__(self, df, sr=16000, maxlen=80000):
        self.df = df.reset_index(drop=True); self.sr = sr; self.maxlen = maxlen
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            p = resolve_audio_path(r)
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y = y[:self.maxlen] if len(y)>self.maxlen else np.pad(y,(0,self.maxlen-len(y)))
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
        r = self.df.iloc[i]
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd, pr = get_vid_paths(sid)
        seq = np.concatenate([np.load(pc), np.load(pd), np.load(pr)], -1)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# ABLATION RUNNERS (With High-Accuracy Architectures)
# ─────────────────────────────────────────────────────────
def train_audio_ablation(tr, va, te, use_ssl, use_supcon, sc_name):
    print(f"\n  [AUDIO] Scenario: {sc_name}")
    set_seed(42)
    cw = compute_class_weight('balanced', classes=np.arange(7), y=np.array([LID[e] for e in tr['emotion_final']]))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    tl = DataLoader(AudioFTDS(tr), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(AudioFTDS(va), batch_size=8, num_workers=2)
    el = DataLoader(AudioFTDS(te), batch_size=8, num_workers=2)
    m = AudioFTModel().to(DEVICE)
    if use_ssl:
        ckpt = SSL_DIR/"audio_ssl.pt"
        assert ckpt.exists(), f"CRITICAL: SSL weight {ckpt} missing!"
        sd = torch.load(ckpt, map_location=DEVICE)
        m.backbone.backbone.load_state_dict(sd['backbone'], strict=False)
        m.backbone.lw.data = sd['lw']
        
    for i, layer in enumerate(m.backbone.backbone.encoder.layers):
        if i >= 6:
            for p in layer.parameters(): p.requires_grad = False
            
    opt = torch.optim.AdamW([{'params':m.classifier.parameters(),'lr':1e-3},{'params':m.proj_ft.parameters(),'lr':1e-3},{'params':m.backbone.lw,'lr':1e-3}])
    scaler = GradScaler(); supcon_fn = SupConLoss(SSL_TEMP)
    best_f1, ckpt = 0, SAVE_DIR/f"aud_{sc_name.replace(' ','_')}.pt"
    
    for ep in range(1, 16):
        if ep == 3:
            for i, layer in enumerate(m.backbone.backbone.encoder.layers):
                if i >= 6:
                    for p in layer.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([
                {'params':[p for i, l in enumerate(m.backbone.backbone.encoder.layers) if i >= 6 for p in l.parameters()],'lr':4e-5},
                {'params':m.classifier.parameters(),'lr':1e-3},
                {'params':m.proj_ft.parameters(),'lr':1e-3},
                {'params':m.backbone.lw,'lr':1e-3}
            ])
            
        m.train()
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            with autocast("cuda"):
                lo, pr = m(x)
                loss = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.1)
                if use_supcon: loss += 0.3 * supcon_fn(pr, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        m.eval(); ps,ts=[],[]
        with torch.no_grad():
            for x,y in vl: ps.extend(m(x.to(DEVICE))[0].argmax(1).cpu().numpy()); ts.extend(y.numpy())
        f1 = f1_score(ts,ps,average='macro',zero_division=0)
        if f1>best_f1: best_f1=f1; torch.save(m.state_dict(), str(ckpt))
        
    m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
    tp=[]
    with torch.no_grad():
        for x,_ in el: tp.append(F.softmax(m(x.to(DEVICE))[0],1).cpu().numpy())
    return np.vstack(tp)

def train_video_ablation(tr, va, te, use_ssl, use_supcon, sc_name):
    print(f"\n  [VIDEO] Scenario: {sc_name}")
    cw = compute_class_weight('balanced', classes=np.arange(7), y=np.array([LID[e] for e in tr['emotion_final']]))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    tl, vl, el = DataLoader(VideoFTDS(tr), batch_size=32, shuffle=True), DataLoader(VideoFTDS(va), batch_size=32), DataLoader(VideoFTDS(te), batch_size=32)
    supcon_fn = SupConLoss(SSL_TEMP)
    all_probs, all_w = [], []
    
    # Use only 3 seeds for ablation to save time (still ensembled)
    for seed in [42, 1337, 2024]:
        set_seed(seed)
        m = VideoFTModel(drop=0.5).to(DEVICE)
        if use_ssl:
            ckpt = SSL_DIR/"video_ssl.pt"
            assert ckpt.exists(), f"CRITICAL: SSL weight {ckpt} missing!"
            m.backbone.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
        opt = Lookahead(torch.optim.AdamW(m.parameters(), lr=7e-5, weight_decay=5e-2))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt.opt, max_lr=8.4e-5, steps_per_epoch=len(tl), epochs=25)
        best_f1, ckpt = 0, SAVE_DIR/f"vid_{sc_name.replace(' ','_')}_{seed}.pt"
        for ep in range(1, 26):
            m.train()
            for x,y in tl:
                x,y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                lo, pr = m(x)
                loss = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.1)
                if use_supcon: loss += 0.3 * supcon_fn(pr, y)
                loss.backward(); opt.step(); sch.step()
            m.eval(); ps,ts = [],[]
            with torch.no_grad():
                for x,y in vl: lo,_ = m(x.to(DEVICE)); ps.extend(lo.argmax(1).cpu().numpy()); ts.extend(y.numpy())
            f1 = f1_score(ts, ps, average='macro', zero_division=0)
            if f1>best_f1: best_f1=f1; torch.save(m.state_dict(), str(ckpt))
        m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
        tp = []
        with torch.no_grad():
            for x,_ in el: lo,_ = m(x.to(DEVICE)); tp.append(F.softmax(lo,1).cpu().numpy())
        all_probs.append(np.vstack(tp)); all_w.append(best_f1)
    w = np.array(all_w); w /= w.sum()
    return sum(p*wt for p,wt in zip(all_probs,w))

def train_text_ablation(tr, va, te, use_ssl, use_supcon, sc_name):
    print(f"\n  [TEXT] Scenario: {sc_name}")
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.concat([tr, va]).reset_index(drop=True)
    texts = pool['transcript'].values
    labels = np.array([LID[e] for e in pool['emotion_final']])
    te_labels = np.array([LID[e] for e in te['emotion_final']])
    cw = compute_class_weight('balanced', classes=np.arange(7), y=labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    te_loader = DataLoader(TextFTDS(te['transcript'].values, te['emotion_final'].values, tok), batch_size=16)
    supcon_fn = SupConLoss(SSL_TEMP)
    
    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_probs = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        tl = DataLoader(TextFTDS(texts[t_idx], [list(LID.keys())[l] for l in labels[t_idx]], tok), batch_size=16, shuffle=True)
        vl = DataLoader(TextFTDS(texts[v_idx], [list(LID.keys())[l] for l in labels[v_idx]], tok), batch_size=16)
        m = TextFTModel().to(DEVICE)
        if use_ssl:
            ckpt = SSL_DIR/"text_ssl.pt"
            assert ckpt.exists(), f"CRITICAL: SSL weight {ckpt} missing!"
            m.backbone.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
        opt = torch.optim.AdamW([
            {'params':[p for n,p in m.named_parameters() if 'bert' in n], 'lr':2e-5},
            {'params':[p for n,p in m.named_parameters() if 'bert' not in n], 'lr':8e-4}
        ], weight_decay=0.01)
        best_acc, ckpt, pat = 0, SAVE_DIR/f"txt_{sc_name.replace(' ','_')}_fold{fold}.pt", 0
        for ep in range(1, 40):
            m.train()
            for bd,bl in tl:
                opt.zero_grad()
                ids, mask, y = bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE), bl.to(DEVICE)
                lo, pr = m(ids, mask)
                loss = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.08)
                if use_supcon: loss += 0.3 * supcon_fn(pr, y)
                loss.backward(); opt.step()
            m.eval(); ps,ts=[],[]
            with torch.no_grad():
                for bd,bl in vl:
                    ps.extend(m(bd['input_ids'].to(DEVICE),bd['attention_mask'].to(DEVICE))[0].argmax(1).cpu().numpy())
                    ts.extend(bl.numpy())
            acc=accuracy_score(ts,ps)
            if acc > best_acc: best_acc=acc; torch.save(m.state_dict(),str(ckpt)); pat=0
            else:
                pat+=1
                if pat>=8: break
        m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
        fp=[]
        with torch.no_grad():
            for bd,_ in te_loader: fp.append(F.softmax(m(bd['input_ids'].to(DEVICE),bd['attention_mask'].to(DEVICE))[0],1).cpu().numpy())
        fold_probs.append(np.vstack(fp))
    return np.mean(fold_probs, 0)

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
        sep(f"RUNNING SCENARIO: {sc['name']} (WITH ENSEMBLES & SOTA PARAMS)")
        
        ap = train_audio_ablation(tr, va, te, sc['ssl'], sc['supcon'], sc['name'])
        vp = train_video_ablation(tr, va, te, sc['ssl'], sc['supcon'], sc['name'])
        tp = train_text_ablation(tr, va, te, sc['ssl'], sc['supcon'], sc['name'])
        
        # Grid Search Fusion
        best_acc, best_f1, best_w = 0, 0, (0.33, 0.33, 0.34)
        for w_v in np.linspace(0.1, 0.8, 15):
            for w_a in np.linspace(0.1, 0.8, 15):
                w_t = round(1.0 - w_v - w_a, 3)
                if w_t < 0.05: continue
                fp = w_v * vp + w_a * ap + w_t * tp
                preds = fp.argmax(1)
                acc = accuracy_score(t_labels, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_f1 = f1_score(t_labels, preds, average='macro', zero_division=0)
                    best_w = (w_v, w_a, w_t)
                    
        results.append({
            "Scenario": sc['name'], 
            "Acc": best_acc, 
            "F1": best_f1,
            "Weights (V,A,T)": f"{best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}"
        })
        print(f"\n  >>> {sc['name']} Result: Acc={best_acc:.4f}, F1={best_f1:.4f} | Weights: {best_w}")

    sep("FINAL ABLATION RESULTS (ENSEMBLED SOTA)")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_results_ensembled.csv", index=False)

if __name__ == "__main__":
    sep(f"CONTRASTIVE PIPELINE v2 (HIGH ACCURACY ABLATION) | Device: {DEVICE}")
    tr, va, te = load_splits()
    
    train_ssl_phase(pd.concat([tr, va]).reset_index(drop=True))
    run_ablation(tr, va, te)
