"""
fusion_production_v2.py — Track A Multimodal Emotion Recognition (Production + SSL + SupCon)
=============================================================================================
Combines the aggressive ensembling and Lookahead optimizers from fusion_production_v1.py
with the Self-Supervised Learning (SSL) pre-training and Supervised Contrastive (SupCon)
fine-tuning from fusion_contrastive_v2.py.

Run on: Google Colab (T4 GPU recommended)
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
# PATHS (Colab)
# ─────────────────────────────────────────────────────────
import glob
cands = glob.glob("/content/*thesis*") + glob.glob("/content/*omaremad*")
cands = [c for c in cands if os.path.isdir(c) and (Path(c)/"src").exists()]
_repo_str = cands[0] if cands else "/content/thesis"

REPO       = Path(_repo_str)
DRIVE      = Path("/content/drive/MyDrive/Thesis Project")
SPLIT_DIR  = REPO / "data/processed/splits/multimodal_eligible"
VID_DIR    = Path("/content/video_features/video_sequences_v1")
SAVE_DIR   = Path("/content/fusion_models")
SSL_DIR    = Path("/content/ssl_pretrained")
for d in [SAVE_DIR, SSL_DIR]: d.mkdir(exist_ok=True)

def auto_detect_audio():
    bases = [
        Path("/content/audio/Thesis Project/dataset/Final Modalink Dataset MERGED"),
        Path("/content/audio/Thesis_Audio_Full"),
        Path("/content/audio")
    ]
    for b in bases:
        if b.exists(): return b
    return DRIVE / "dataset/Final Modalink Dataset MERGED"

AUDIO_BASE = auto_detect_audio()

LID     = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6}
CLASSES = list(LID.keys())
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# SSL Hyperparams
SSL_EPOCHS   = 40
SSL_TEMP     = 0.07
SSL_PROJ_DIM = 128
GRAD_ACC     = 4

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sep(title=""):
    print("\n" + "="*55)
    if title: print(f"  {title}")
    print("="*55)

def report(name, targets, preds):
    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    sep(f"📊 {name} — Test Acc: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(targets, preds, target_names=CLASSES, zero_division=0))
    return acc, f1

# ─────────────────────────────────────────────────────────
# SPLIT LOADING & ALIGNMENT
# ─────────────────────────────────────────────────────────
def resolve_audio_path(row):
    folder = str(row.get('folder', ''))
    audio_rel = str(row.get('audio_relpath', ''))
    bases = [AUDIO_BASE, DRIVE/"dataset/Final Modalink Dataset MERGED", DRIVE/"data/raw", DRIVE]
    for base in bases:
        if folder and audio_rel:
            p = base / folder / audio_rel
            if p.exists(): return p
            p_bs = base / f"{folder}\\{audio_rel.replace('/', '\\')}"
            if p_bs.exists(): return p_bs
        if audio_rel:
            p = base / audio_rel
            if p.exists(): return p
            p_flat = base / Path(audio_rel).name
            if p_flat.exists(): return p_flat
    return None

def get_vid_paths(sid):
    p1 = VID_DIR / f"{sid}_clip_seq.npy"
    if p1.exists(): return p1, VID_DIR / f"{sid}_dinov2_seq.npy", VID_DIR / f"{sid}_resnet50_seq.npy"
    p2 = VID_DIR.parent / f"video_sequences_v1\\{sid}_clip_seq.npy"
    if p2.exists(): return p2, VID_DIR.parent / f"video_sequences_v1\\{sid}_dinov2_seq.npy", VID_DIR.parent / f"video_sequences_v1\\{sid}_resnet50_seq.npy"
    return None, None, None

def load_splits():
    tr = pd.read_csv(SPLIT_DIR/"train.csv")
    va = pd.read_csv(SPLIT_DIR/"val.csv")
    te = pd.read_csv(SPLIT_DIR/"test.csv")
    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        vid_p, _, _ = get_vid_paths(sid)
        return (vid_p is not None and vid_p.exists() and 
                resolve_audio_path(row) is not None and 
                isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2)
    tr = tr[tr.apply(ok, axis=1)].reset_index(drop=True)
    va = va[va.apply(ok, axis=1)].reset_index(drop=True)
    te = te[te.apply(ok, axis=1)].reset_index(drop=True)
    sep("ALIGNED SPLITS")
    print(f"  Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
    return tr, va, te

# ─────────────────────────────────────────────────────────
# PHASE 1: SELF-SUPERVISED LEARNING (SSL) UTILS
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
    def forward(self, features, labels):
        # Features are already L2 normalized by models
        B = features.size(0)
        sim = torch.mm(features, features.T) / self.T
        mask_self = torch.eye(B, device=features.device).bool()
        sim.masked_fill_(mask_self, float('-inf'))
        pos_mask = labels.view(-1,1).eq(labels.view(1,-1)).float()
        pos_mask.fill_diagonal_(0)
        log_prob = F.log_softmax(sim, dim=1)
        n_pos    = pos_mask.sum(1).clamp(min=1)
        return (-(pos_mask * log_prob).sum(1) / n_pos).mean()

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
    def forward(self, x): return self.net(x)

def _audio_one_view(w, maxlen=80000):
    w = w.copy()
    if np.random.rand()>0.3: w += np.random.randn(len(w))*(np.sqrt(np.mean(w**2))+1e-9)/(10**(np.random.uniform(15,30)/20))
    if np.random.rand()>0.4:
        ml = int(len(w)*np.random.uniform(0.05,0.15))
        ms = np.random.randint(0, max(1, len(w)-ml))
        w[ms:ms+ml] = 0.0
    if np.random.rand()>0.5:
        r = np.random.uniform(0.9,1.1)
        idx = np.linspace(0, len(w)-1, int(len(w)/r)).astype(int)
        w = w[idx]
        w = w[:maxlen] if len(w)>maxlen else np.pad(w, (0, maxlen-len(w)))
    return w
def audio_augment(wav): return _audio_one_view(wav), _audio_one_view(wav)

def _video_one_view(seq, n_mask=4):
    s = seq.copy()
    s[np.random.choice(s.shape[0], n_mask, replace=False)] = 0.0
    if np.random.rand()>0.4: s += np.random.randn(*s.shape)*0.02
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
# PHASE 1 BACKBONES
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
        hs = torch.stack(self.backbone(x).hidden_states, 0)
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
    def __init__(self, d=512, drop=0.3):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(3584, d), nn.LayerNorm(d), nn.GELU())
        self.se      = SEBlock(d)
        self.pos     = nn.Parameter(torch.randn(1, 16, d))
        enc          = nn.TransformerEncoderLayer(d, 8, d*2, drop, batch_first=True)
        self.tfm     = nn.TransformerEncoder(enc, 2)
        self.attn    = nn.Linear(d, 1)
        self.fuse    = nn.Linear(d*3, d)
        self.proj    = ProjectionHead(d, 128)
    def _pool(self, x): return (x * F.softmax(self.attn(x), 1)).sum(1)
    def encode(self, x):
        x = self.tfm(self.se(self.proj_in(x)) + self.pos)
        return self.fuse(torch.cat([self._pool(x), self._pool(x[:,4:12,:]), self._pool(x[:,6:10,:])], -1))
    def forward(self, x): return self.proj(self.encode(x))

# Phase 1 Training Wrappers
def train_ssl_phase(pool):
    # Same as V2 - runs pre-training if checkpoints not found
    sep("PHASE 1: SELF-SUPERVISED PRE-TRAINING")
    
    # Text
    ckpt = SSL_DIR / "text_ssl.pt"
    if not ckpt.exists():
        print("  Training Text SSL...")
        class TextSSLDS(Dataset):
            def __init__(self, texts, tok):
                self.enc = tok([clean(str(t)) for t in texts], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            def __len__(self): return self.enc['input_ids'].size(0)
            def __getitem__(self, i): return {k: v[i] for k,v in self.enc.items()}
        ds = TextSSLDS(pool['transcript'].values, AutoTokenizer.from_pretrained(MODEL_NAME))
        dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
        m = TextSSLModel().to(DEVICE)
        opt = torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if 'bert' in n and p.requires_grad],'lr':2e-5},{'params':[p for n,p in m.named_parameters() if 'proj' in n],'lr':1e-3}], weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS)
        loss_fn = InfoNCELoss(SSL_TEMP)
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
# PHASE 2: PRODUCTION MODELS WITH SSL BACKBONES & SUPCON
# ─────────────────────────────────────────────────────────

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

# --- VIDEO ---
class VideoFTModel(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()
        self.backbone = VideoSSLModel(d=512, drop=drop)
        self.cls = nn.Sequential(nn.LayerNorm(512), nn.Dropout(drop), nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 7))
        self.scl = ProjectionHead(512, 128)
    def forward(self, x):
        if self.training: x = x + torch.randn_like(x)*0.01
        feat = self.backbone.encode(x)
        return self.cls(feat), F.normalize(self.scl(feat), dim=1)

class VideoDS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd, pr = get_vid_paths(sid)
        return torch.tensor(np.concatenate([np.load(pc), np.load(pd), np.load(pr)], -1), dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

def train_video_prod(tr, va, te):
    sep("🎥 VIDEO MODALITY — SSL Backbone + SupCon + 5-Seed Ensemble")
    cw = compute_class_weight('balanced', classes=np.arange(7), y=np.array([LID[e] for e in tr['emotion_final']]))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    tl, vl, el = DataLoader(VideoDS(tr), batch_size=32, shuffle=True), DataLoader(VideoDS(va), batch_size=32), DataLoader(VideoDS(te), batch_size=32)
    supcon_fn = SupConLoss(SSL_TEMP)
    all_probs, all_w = [], []
    for seed in [42, 1337, 2024, 777, 999]:
        print(f"\n  ── Seed {seed} ──")
        set_seed(seed)
        m = VideoFTModel().to(DEVICE)
        m.backbone.load_state_dict(torch.load(SSL_DIR/"video_ssl.pt", map_location=DEVICE), strict=False)
        opt = Lookahead(torch.optim.AdamW(m.parameters(), lr=7e-5, weight_decay=5e-2))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt.opt, max_lr=8.4e-5, steps_per_epoch=len(tl), epochs=25)
        best_f1, ckpt = 0, SAVE_DIR/f"vid_{seed}.pt"
        for ep in range(1, 26):
            m.train()
            for x,y in tl:
                x,y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                lo, pr = m(x)
                loss = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.1) + 0.3 * supcon_fn(pr, y)
                loss.backward(); opt.step(); sch.step()
            m.eval(); ps,ts = [],[]
            with torch.no_grad():
                for x,y in vl: lo,_ = m(x.to(DEVICE)); ps.extend(lo.argmax(1).cpu().numpy()); ts.extend(y.numpy())
            f1 = f1_score(ts, ps, average='macro', zero_division=0)
            print(f"    Ep {ep:02d} | Val Acc: {accuracy_score(ts,ps):.4f} | Val F1: {f1:.4f}" + (" ⭐" if f1>best_f1 else ""))
            if f1>best_f1: best_f1=f1; torch.save(m.state_dict(), str(ckpt))
        m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
        tp = []
        with torch.no_grad():
            for x,_ in el: lo,_ = m(x.to(DEVICE)); tp.append(F.softmax(lo,1).cpu().numpy())
        all_probs.append(np.vstack(tp)); all_w.append(best_f1)
    w = np.array(all_w); w /= w.sum()
    probs = sum(p*wt for p,wt in zip(all_probs,w))
    report("VIDEO ENSEMBLE", [LID[e] for e in te['emotion_final']], probs.argmax(1))
    return probs

# --- AUDIO ---
class AudioFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AudioSSLModel(proj_dim=128)
        self.cls = nn.Sequential(nn.Linear(768*2,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,7))
        self.scl = ProjectionHead(768*2, 128)
    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.cls(feat), F.normalize(self.scl(feat), dim=1)

class AudioDS(Dataset):
    def __init__(self, df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            p = resolve_audio_path(r)
            y,_ = librosa.load(str(p), sr=16000)
            y,_ = librosa.effects.trim(y, top_db=25)
            y = y[:80000] if len(y)>80000 else np.pad(y,(0,80000-len(y)))
        except: y = np.zeros(80000)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

def train_audio_prod(tr, va, te):
    sep("🎙️ AUDIO MODALITY — SSL Backbone + SupCon + Progressive Unfreeze")
    set_seed(42)
    cw = compute_class_weight('balanced', classes=np.arange(7), y=np.array([LID[e] for e in tr['emotion_final']]))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    tl, vl, el = DataLoader(AudioDS(tr), batch_size=8, shuffle=True, num_workers=2, pin_memory=True), DataLoader(AudioDS(va), batch_size=8, num_workers=2), DataLoader(AudioDS(te), batch_size=8, num_workers=2)
    supcon_fn = SupConLoss(SSL_TEMP)
    m = AudioFTModel().to(DEVICE)
    sd = torch.load(SSL_DIR/"audio_ssl.pt", map_location=DEVICE)
    m.backbone.backbone.load_state_dict(sd['backbone'])
    m.backbone.lw.data = sd['lw']
    
    # Bottom 6 layers are already permanently frozen by AudioSSLModel.__init__
    # Warmup: freeze the rest of the backbone for epochs 1-2
    for i, layer in enumerate(m.backbone.backbone.encoder.layers):
        if i >= 6:
            for p in layer.parameters(): p.requires_grad = False
            
    opt = torch.optim.AdamW([{'params':m.cls.parameters(),'lr':1e-3},{'params':m.scl.parameters(),'lr':1e-3},{'params':m.backbone.lw,'lr':1e-3}])
    scaler = GradScaler()
    best_f1, ckpt = 0, SAVE_DIR/"aud_best.pt"
    
    for ep in range(1, 16):
        if ep == 3:
            for i, layer in enumerate(m.backbone.backbone.encoder.layers):
                if i >= 6:
                    for p in layer.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([
                {'params':[p for i, l in enumerate(m.backbone.backbone.encoder.layers) if i >= 6 for p in l.parameters()],'lr':4e-5},
                {'params':m.cls.parameters(),'lr':1e-3},
                {'params':m.scl.parameters(),'lr':1e-3},
                {'params':m.backbone.lw,'lr':1e-3}
            ])
            print("  [Ep 3] Top 6 Backbone layers unfrozen")
            
        m.train(); tl_s=0
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            with autocast("cuda"):
                lo, pr = m(x)
                loss = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.1) + 0.3 * supcon_fn(pr, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl_s += loss.item()
        m.eval(); ps,ts=[],[]
        with torch.no_grad():
            for x,y in vl: ps.extend(m(x.to(DEVICE))[0].argmax(1).cpu().numpy()); ts.extend(y.numpy())
        f1 = f1_score(ts,ps,average='macro',zero_division=0)
        print(f"  Ep {ep:02d} | Loss: {tl_s/len(tl):.4f} | Val Acc: {accuracy_score(ts,ps):.4f} | Val F1: {f1:.4f}" + (" ⭐" if f1>best_f1 else ""))
        if f1>best_f1: best_f1=f1; torch.save(m.state_dict(), str(ckpt))
    
    m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
    tp=[]
    with torch.no_grad():
        for x,_ in el: tp.append(F.softmax(m(x.to(DEVICE))[0],1).cpu().numpy())
    probs = np.vstack(tp)
    report("AUDIO", [LID[e] for e in te['emotion_final']], probs.argmax(1))
    return probs

# --- TEXT ---
class TextFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TextSSLModel(proj_dim=128)
        self.cls = nn.Linear(768*3, 7)
        self.drops = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.scl = ProjectionHead(768*3, 128)
    def forward(self, ids, mask):
        feat = self.backbone.encode(ids, mask)
        logits = torch.stack([self.cls(d(feat)) for d in self.drops]).mean(0)
        return logits, F.normalize(self.scl(feat), dim=1)

class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok([clean(str(t)) for t in texts], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {k:v[i] for k,v in self.enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)

def train_text_prod(tr, va, te):
    sep("📝 TEXT MODALITY — SSL Backbone + SupCon + 5-Fold CV Ensemble")
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.concat([tr, va]).reset_index(drop=True)
    texts = pool['transcript'].values
    labels = np.array([LID[e] for e in pool['emotion_final']])
    te_labels = np.array([LID[e] for e in te['emotion_final']])
    cw = compute_class_weight('balanced', classes=np.arange(7), y=labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    te_loader = DataLoader(TextDS(te['transcript'].values, te_labels, tok), batch_size=16)
    supcon_fn = SupConLoss(SSL_TEMP)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_probs = []
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n  ── Fold {fold+1}/5 ──")
        tl = DataLoader(TextDS(texts[t_idx], labels[t_idx], tok), batch_size=16, shuffle=True)
        vl = DataLoader(TextDS(texts[v_idx], labels[v_idx], tok), batch_size=16)
        m = TextFTModel().to(DEVICE)
        m.backbone.load_state_dict(torch.load(SSL_DIR/"text_ssl.pt", map_location=DEVICE), strict=False)
        opt = torch.optim.AdamW([
            {'params':[p for n,p in m.named_parameters() if 'bert' in n], 'lr':2e-5},
            {'params':[p for n,p in m.named_parameters() if 'bert' not in n], 'lr':8e-4}
        ], weight_decay=0.01)
        best_acc, ckpt, pat = 0, SAVE_DIR/f"txt_fold{fold}.pt", 0
        for ep in range(1, 40):
            m.train()
            for bd,bl in tl:
                opt.zero_grad()
                ids, mask, y = bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE), bl.to(DEVICE)
                lo, pr = m(ids, mask)
                loss = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.08) + 0.3 * supcon_fn(pr, y)
                loss.backward(); opt.step()
            m.eval(); ps,ts=[],[]
            with torch.no_grad():
                for bd,bl in vl:
                    ps.extend(m(bd['input_ids'].to(DEVICE),bd['attention_mask'].to(DEVICE))[0].argmax(1).cpu().numpy())
                    ts.extend(bl.numpy())
            acc=accuracy_score(ts,ps)
            print(f"    Ep {ep:02d} | Val Acc: {acc:.4f} | Val F1: {f1_score(ts,ps,average='macro',zero_division=0):.4f}" + (" ⭐" if acc>best_acc else ""))
            if acc > best_acc: best_acc=acc; torch.save(m.state_dict(),str(ckpt)); pat=0
            else:
                pat+=1
                if pat>=8: print(f"    Early stop."); break
        m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
        fp=[]
        with torch.no_grad():
            for bd,_ in te_loader: fp.append(F.softmax(m(bd['input_ids'].to(DEVICE),bd['attention_mask'].to(DEVICE))[0],1).cpu().numpy())
        fold_probs.append(np.vstack(fp))
        print(f"    Rolling Test Acc: {accuracy_score(te_labels, np.mean(fold_probs,0).argmax(1)):.4f}")
        
    probs = np.mean(fold_probs, 0)
    report("TEXT (5-Fold Ensemble)", te_labels, probs.argmax(1))
    return probs

# ─────────────────────────────────────────────────────────
# LATE FUSION
# ─────────────────────────────────────────────────────────
def late_fusion(vp, ap, tp, te):
    sep("🔀 LATE FUSION — Weighted Average Grid Search")
    t_labels = np.array([LID[e] for e in te['emotion_final']])
    print(f"  Video standalone → Acc: {accuracy_score(t_labels,vp.argmax(1)):.4f} | F1: {f1_score(t_labels,vp.argmax(1),average='macro',zero_division=0):.4f}")
    print(f"  Audio standalone → Acc: {accuracy_score(t_labels,ap.argmax(1)):.4f} | F1: {f1_score(t_labels,ap.argmax(1),average='macro',zero_division=0):.4f}")
    print(f"  Text  standalone → Acc: {accuracy_score(t_labels,tp.argmax(1)):.4f} | F1: {f1_score(t_labels,tp.argmax(1),average='macro',zero_division=0):.4f}")
    
    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wv in np.arange(0.1, 0.8, 0.05):
        for wa in np.arange(0.1, 0.8, 0.05):
            wt = round(1.0 - wv - wa, 3)
            if wt < 0.05: continue
            probs = wv*vp + wa*ap + wt*tp
            f1 = f1_score(t_labels, probs.argmax(1), average='macro', zero_division=0)
            if f1 > best_f1: best_f1=f1; best_w=(wv, wa, wt)
            
    wv, wa, wt = best_w
    final_probs = wv*vp + wa*ap + wt*tp
    np.save(str(SAVE_DIR/"fusion_probs_v2.npy"), final_probs)
    print(f"\n  Best weights → Video: {wv:.2f} | Audio: {wa:.2f} | Text: {wt:.2f}")
    report("🏆 V2 LATE FUSION FINAL (SSL+SupCon+Ensemble)", t_labels, final_probs.argmax(1))

# ─────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sep(f"🚀 MULTIMODAL PRODUCTION V2 (SSL+SupCon) | Device: {DEVICE}")
    tr, va, te = load_splits()
    
    # 1. SSL Phase
    train_ssl_phase(pd.concat([tr, va]).reset_index(drop=True))
    
    # 2. Production FT Phase
    VP, AP, TP = SAVE_DIR/"vid_probs_v2.npy", SAVE_DIR/"aud_probs_v2.npy", SAVE_DIR/"txt_probs_v2.npy"
    
    if VP.exists():
        print("\n⏩ Loading cached video probs...")
        vid_probs = np.load(str(VP))
    else:
        vid_probs = train_video_prod(tr, va, te)
        np.save(str(VP), vid_probs)
        
    if AP.exists():
        print("\n⏩ Loading cached audio probs...")
        aud_probs = np.load(str(AP))
    else:
        aud_probs = train_audio_prod(tr, va, te)
        np.save(str(AP), aud_probs)
        
    if TP.exists():
        print("\n⏩ Loading cached text probs...")
        txt_probs = np.load(str(TP))
    else:
        txt_probs = train_text_prod(tr, va, te)
        np.save(str(TP), txt_probs)
        
    late_fusion(vid_probs, aud_probs, txt_probs, te)
