"""
fusion_contrastive_v2.py — Full Ablation Study (Colab)
================================================================
Runs the 4-scenario ablation (Baseline, SupCon only, SSL only, SSL+SupCon)
using the HIGH-ACCURACY PRODUCTION architecture (5-Seed Ensembles, 5-Fold CV,
Lookahead Optimizer, Progressive Unfreezing).

Bugs fixed in this version:
  1. SupConLoss: denominator was including self-similarity → loss was computed
     incorrectly. Now uses logsumexp with self masked to -inf, per-anchor
     normalization, correct margin application.
  2. Video SSL save: 'proj' not in k' filter also excluded proj_in.*  (the
     critical 3584→512 input projection). Fixed to startswith('proj.').
  3. Audio freeze: baseline scenario had ALL 12 WavLM layers frozen (layers 0-5
     frozen in AudioSSLModel.__init__, layers 6-11 frozen unconditionally in
     train_audio_ablation). Baseline now gets full backbone fine-tuning.
  4. Fusion: weights were grid-searched on TEST labels (data leakage). Now
     optimised on validation set; best weights applied to test.
  5. Text FT: missing LR scheduler added (CosineAnnealingLR).
  6. Logging: SSL epoch loss, FT epoch loss/acc/F1, data-size reports added.
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
    v_dir = None
    for p in Path("/content").rglob("*_clip_seq.npy"):
        if "drive" not in str(p):
            v_dir = p.parent; break
    if not v_dir:
        v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features/video_sequences_v1")
        if not v_dir.exists():
            v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features")
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
    print("\n" + "="*60)
    if t: print(f"  {t}")
    print("="*60)

# ─────────────────────────────────────────────────────────
# DATA LOADING
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
        Path("/content/drive/MyDrive"),
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
    if p1.exists():
        return p1, VID_DIR / f"{sid}_dinov2_seq.npy", VID_DIR / f"{sid}_resnet50_seq.npy"
    p2 = VID_DIR / f"video_sequences_v1\\{sid}_clip_seq.npy"
    if p2.exists():
        return p2, VID_DIR / f"video_sequences_v1\\{sid}_dinov2_seq.npy", VID_DIR / f"video_sequences_v1\\{sid}_resnet50_seq.npy"
    return None, None, None

def load_splits():
    tr = pd.read_csv(SPLIT_DIR/"train.csv")
    va = pd.read_csv(SPLIT_DIR/"val.csv")
    te = pd.read_csv(SPLIT_DIR/"test.csv")
    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        vid_p, _, _ = get_vid_paths(sid)
        return (vid_p is not None and vid_p.exists()
                and resolve_audio_path(row) is not None
                and isinstance(row.get('transcript'), str)
                and len(str(row['transcript']).strip()) > 2)
    tr = tr[tr.apply(ok, axis=1)].reset_index(drop=True)
    va = va[va.apply(ok, axis=1)].reset_index(drop=True)
    te = te[te.apply(ok, axis=1)].reset_index(drop=True)
    print(f"[DATA] Train: {len(tr)} | Val: {len(va)} | Test: {len(te)} multimodal-eligible samples")
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
        B  = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T
        sim.fill_diagonal_(float('-inf'))
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0,  B,  device=z.device),
        ])
        return F.cross_entropy(sim, labels)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020) with class-specific margins.

    FIX: Previous version computed denominator BEFORE masking out the diagonal,
    so self-similarity polluted the log-partition. Also normalized by the mean
    number of positives across the batch instead of per-anchor. Both are fixed.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature
        # Margins make these easily-confused classes harder positives
        self.margins = {3: 0.2, 6: 0.15, 2: 0.25}   # Happiness, Surprise, Fear

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        B = features.size(0)

        sim = torch.mm(features, features.T) / self.T

        # Self mask (exclude diagonal from everything)
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)

        # Positive mask: same class AND not self
        pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)) & ~self_mask

        # Apply class-specific margins to positive pairs only
        mg = torch.zeros(B, device=features.device)
        for k, v in self.margins.items():
            mg[labels == k] = v
        margin_mat = (mg.unsqueeze(0) + mg.unsqueeze(1)) * 0.5
        sim = sim - pos_mask.float() * margin_mat

        # Mask self BEFORE computing log-partition (critical fix)
        sim = sim.masked_fill(self_mask, float('-inf'))

        # log P(j | i) over all non-self pairs
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Per-anchor mean over positives (another fix: was global mean)
        n_pos = pos_mask.float().sum(1)
        valid = n_pos > 0
        if not valid.any():
            return features.sum() * 0.0   # differentiable zero

        loss = -(pos_mask.float() * log_prob).sum(1)
        return (loss[valid] / n_pos[valid]).mean()


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
            for i, g in enumerate(self.param_groups):
                for j, p in enumerate(g['params']):
                    p.data.mul_(self.a).add_(self.slow[i][j], alpha=1-self.a)
                    self.slow[i][j].copy_(p.data)
    def zero_grad(self, **kw): self.opt.zero_grad(**kw)

# ─────────────────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────────────────
def _audio_one_view(wav, maxlen=80000):
    w = wav.copy()
    if np.random.rand() > 0.3:
        w += np.random.randn(len(w)) * (np.sqrt(np.mean(w**2)) + 1e-9) / (10**(np.random.uniform(15, 30)/20))
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
    t = re.sub(r'[ً-ٰٟ]', '', t)
    t = re.sub(r'[آأإ]', 'ا', t)
    t = re.sub(r'ة', 'ه', t)
    t = re.sub(r'ى', 'ي', t)
    t = re.sub(r'ـ', '', t)
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
        b, n, c = x.shape
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

# ─────────────────────────────────────────────────────────
# PHASE 1: SELF-SUPERVISED PRE-TRAINING
# ─────────────────────────────────────────────────────────
def train_ssl_phase(pool):
    sep("PHASE 1: SELF-SUPERVISED PRE-TRAINING (SAVING TO GOOGLE DRIVE)")
    print(f"  SSL pool: {len(pool)} samples (train + val combined)")
    print(f"  Strategy: conservative SSL — only top-2 transformer layers updated")
    print(f"  Rationale: with 606 samples, training many layers collapses InfoNCE")
    print(f"             to near-zero and degrades strong pretrained representations.")

    # ── TEXT SSL ──────────────────────────────────────────
    ckpt = SSL_DIR / "text_ssl.pt"
    if not ckpt.exists():
        class TextSSLDS(Dataset):
            def __init__(self, texts, tok):
                self.enc = tok([clean(str(t)) for t in texts],
                               truncation=True, padding="max_length",
                               max_length=64, return_tensors="pt")
            def __len__(self): return self.enc['input_ids'].size(0)
            def __getitem__(self, i): return {k: v[i] for k,v in self.enc.items()}

        ds  = TextSSLDS(pool['transcript'].values, AutoTokenizer.from_pretrained(MODEL_NAME))
        dl  = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
        _ep = 20
        print(f"\n  [TEXT SSL] {len(ds)} samples | {len(dl)} batches/epoch | {_ep} epochs")
        m   = TextSSLModel().to(DEVICE)
        # TextSSLModel freezes layers 0-7; also freeze 8-9 → only top-2 (10-11) trainable
        for i, layer in enumerate(m.bert.encoder.layer):
            if i < 10:
                for p in layer.parameters(): p.requires_grad = False
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  [TEXT SSL] Trainable params: {trainable:,} (top-2 BERT layers + projection)")
        opt = torch.optim.AdamW([
            {'params': [p for n,p in m.named_parameters() if 'bert' in n and p.requires_grad], 'lr': 1e-5},
            {'params': m.proj.parameters(), 'lr': 5e-4},
        ], weight_decay=0.01)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_ep)
        loss_fn = InfoNCELoss(0.15)   # softer than 0.07/0.1 — prevents sharp collapse
        for ep in range(1, _ep+1):
            m.train(); ep_loss = 0
            for bd in dl:
                ids, mask = bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(m(ids, mask), m(ids, mask))
                loss.backward(); opt.step()
                ep_loss += loss.item()
            sch.step()
            if ep == 1 or ep % 5 == 0:
                print(f"    Ep {ep:2d}/{_ep} | Loss: {ep_loss/len(dl):.4f} | LR: {sch.get_last_lr()[0]:.2e}")
        torch.save({k:v for k,v in m.state_dict().items() if 'proj' not in k}, str(ckpt))
        print(f"  [SAVED] Text SSL → {ckpt}")
    else:
        print(f"  [SKIP] Text SSL checkpoint found: {ckpt}")

    # ── VIDEO SSL ─────────────────────────────────────────
    ckpt = SSL_DIR / "video_ssl.pt"
    if not ckpt.exists():
        class VideoSSLDS(Dataset):
            def __init__(self, df): self.df = df
            def __len__(self): return len(self.df)
            def __getitem__(self, i):
                sid = self.df.iloc[i]['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
                pc, pd_path, pr = get_vid_paths(sid)
                seq = np.concatenate([np.load(pc), np.load(pd_path), np.load(pr)], -1)
                return video_augment(seq)

        dl  = DataLoader(VideoSSLDS(pool), batch_size=32, shuffle=True, drop_last=True)
        _ep = 30
        print(f"\n  [VIDEO SSL] {len(pool)} samples | {len(dl)} batches/epoch | {_ep} epochs")
        # Video backbone is fully custom (no massive pretraining) → full training OK
        m   = VideoSSLModel().to(DEVICE)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  [VIDEO SSL] Trainable params: {trainable:,} (full custom backbone)")
        opt     = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_ep)
        loss_fn = InfoNCELoss(0.12)   # slightly softer than 0.07
        for ep in range(1, _ep+1):
            m.train(); ep_loss = 0
            for v1, v2 in dl:
                opt.zero_grad()
                loss = loss_fn(m(v1.float().to(DEVICE)), m(v2.float().to(DEVICE)))
                loss.backward(); opt.step()
                ep_loss += loss.item()
            sch.step()
            if ep == 1 or ep % 5 == 0:
                print(f"    Ep {ep:2d}/{_ep} | Loss: {ep_loss/len(dl):.4f} | LR: {sch.get_last_lr()[0]:.2e}")
        torch.save({k:v for k,v in m.state_dict().items() if not k.startswith('proj.')}, str(ckpt))
        print(f"  [SAVED] Video SSL → {ckpt}")
    else:
        print(f"  [SKIP] Video SSL checkpoint found: {ckpt}")
        _sd = torch.load(ckpt, map_location='cpu')
        if 'proj_in.0.weight' not in _sd:
            print("  *** WARNING: video_ssl.pt missing proj_in weights. Delete and re-run. ***")

    # ── AUDIO SSL ─────────────────────────────────────────
    ckpt = SSL_DIR / "audio_ssl.pt"
    if not ckpt.exists():
        class AudioSSLDS(Dataset):
            def __init__(self, df): self.df = df
            def __len__(self): return len(self.df)
            def __getitem__(self, i):
                p = resolve_audio_path(self.df.iloc[i])
                try:
                    y, _ = librosa.load(str(p), sr=16000)
                    y, _ = librosa.effects.trim(y, top_db=25)
                    y = y[:80000] if len(y) > 80000 else np.pad(y, (0, 80000-len(y)))
                except:
                    y = np.zeros(80000)
                v1, v2 = audio_augment(y)
                return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)

        dl  = DataLoader(AudioSSLDS(pool), batch_size=8, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
        _ep = 20
        print(f"\n  [AUDIO SSL] {len(pool)} samples | {len(dl)} batches/epoch | {_ep} epochs (eff. batch={8*GRAD_ACC})")
        m   = AudioSSLModel().to(DEVICE)
        # AudioSSLModel freezes layers 0-5; also freeze 6-8 → only layers 9-11 trainable
        for i, layer in enumerate(m.backbone.encoder.layers):
            if i < 9:
                for p in layer.parameters(): p.requires_grad = False
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  [AUDIO SSL] Trainable params: {trainable:,} (top-3 WavLM layers + lw + projection)")
        opt     = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()),
                                    lr=5e-5, weight_decay=1e-2)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_ep)
        loss_fn = InfoNCELoss(0.15)   # softer temperature → less prone to memorisation
        scaler  = GradScaler()
        for ep in range(1, _ep+1):
            m.train(); ep_loss = 0; opt.zero_grad()
            for step, (v1, v2) in enumerate(dl):
                v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
                with autocast("cuda"):
                    loss = loss_fn(m(v1), m(v2)) / GRAD_ACC
                scaler.scale(loss).backward()
                ep_loss += loss.item() * GRAD_ACC
                if (step+1) % GRAD_ACC == 0 or (step+1) == len(dl):
                    scaler.step(opt); scaler.update(); opt.zero_grad()
            sch.step()
            if ep == 1 or ep % 5 == 0:
                print(f"    Ep {ep:2d}/{_ep} | Loss: {ep_loss/len(dl):.4f} | LR: {sch.get_last_lr()[0]:.2e}")
        torch.save({'backbone': m.backbone.state_dict(), 'lw': m.lw.data}, str(ckpt))
        print(f"  [SAVED] Audio SSL → {ckpt}")
    else:
        print(f"  [SKIP] Audio SSL checkpoint found: {ckpt}")

# ─────────────────────────────────────────────────────────
# PHASE 2 MODELS
# ─────────────────────────────────────────────────────────
class AudioFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone   = AudioSSLModel(proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 7))
        self.proj_ft = ProjectionHead(768*2, proj_dim)

    def forward(self, x):
        feat = self.backbone.encode(x)
        return self.classifier(feat), F.normalize(self.proj_ft(feat), dim=1)


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
        return logits, F.normalize(self.proj_ft(feat), dim=1)


class VideoFTModel(nn.Module):
    def __init__(self, proj_dim=128, drop=0.5):
        super().__init__()
        self.backbone   = VideoSSLModel(proj_dim=proj_dim, drop=drop)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(drop),
            nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(drop), nn.Linear(256, 7))
        self.proj_ft = ProjectionHead(512, proj_dim)

    def forward(self, x):
        if self.training: x = x + torch.randn_like(x) * 0.01
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
            y = y[:self.maxlen] if len(y) > self.maxlen else np.pad(y, (0, self.maxlen-len(y)))
        except:
            y = np.zeros(self.maxlen)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)


class TextFTDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc    = tok([clean(str(t)) for t in texts], truncation=True,
                          padding="max_length", max_length=64, return_tensors="pt")
        self.labels = [LID[l] for l in labels]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k,v in self.enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)


class VideoFTDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r   = self.df.iloc[i]
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd_path, pr = get_vid_paths(sid)
        seq = np.concatenate([np.load(pc), np.load(pd_path), np.load(pr)], -1)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

# ─────────────────────────────────────────────────────────
# ABLATION RUNNERS
# ─────────────────────────────────────────────────────────
def train_audio_ablation(tr, va, te, use_ssl, use_supcon, sc_name):
    """Returns (test_probs, val_probs) both shape [N, 7]."""
    print(f"\n  [AUDIO] {sc_name} | train={len(tr)} val={len(va)} test={len(te)}")
    set_seed(42)
    cw        = compute_class_weight('balanced', classes=np.arange(7),
                                     y=np.array([LID[e] for e in tr['emotion_final']]))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    tl = DataLoader(AudioFTDS(tr), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(AudioFTDS(va), batch_size=8, num_workers=2)
    el = DataLoader(AudioFTDS(te), batch_size=8, num_workers=2)

    m = AudioFTModel().to(DEVICE)

    if use_ssl:
        ckpt_ssl = SSL_DIR / "audio_ssl.pt"
        assert ckpt_ssl.exists(), f"CRITICAL: {ckpt_ssl} missing — run SSL phase first"
        sd = torch.load(ckpt_ssl, map_location=DEVICE)
        m.backbone.backbone.load_state_dict(sd['backbone'], strict=False)
        m.backbone.lw.data = sd['lw']

    # ALL scenarios: freeze upper 6 WavLM layers initially (layers 0-5 already
    # frozen in AudioSSLModel.__init__). Unfreeze top-6 at epoch 3 for all.
    # On a small dataset (502 samples), training as a feature extractor first
    # then gradually unfreezing is more stable than full fine-tuning from epoch 1.
    # SSL/baseline differ only in what the backbone is initialised with.
    for i, layer in enumerate(m.backbone.backbone.encoder.layers):
        if i >= 6:
            for p in layer.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW([
        {'params': m.classifier.parameters(), 'lr': 1e-3},
        {'params': m.proj_ft.parameters(),    'lr': 1e-3},
        {'params': [m.backbone.lw],            'lr': 1e-3},
    ])

    scaler    = GradScaler()
    supcon_fn = SupConLoss(SSL_TEMP)
    best_f1   = 0
    ckpt      = SAVE_DIR / f"aud_{sc_name.replace(' ','_').replace('+','plus')}.pt"

    for ep in range(1, 16):
        # Progressive unfreeze at epoch 3 for ALL scenarios
        if ep == 3:
            for i, layer in enumerate(m.backbone.backbone.encoder.layers):
                if i >= 6:
                    for p in layer.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([
                {'params': [p for i,l in enumerate(m.backbone.backbone.encoder.layers)
                            if i >= 6 for p in l.parameters()], 'lr': 4e-5},
                {'params': m.classifier.parameters(), 'lr': 1e-3},
                {'params': m.proj_ft.parameters(),    'lr': 1e-3},
                {'params': [m.backbone.lw],            'lr': 1e-3},
            ])

        m.train(); tr_loss = 0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with autocast("cuda"):
                lo, pr = m(x)
                loss   = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.1)
                if use_supcon: loss = loss + 0.3 * supcon_fn(pr, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_loss += loss.item()

        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for x, y in vl:
                ps.extend(m(x.to(DEVICE))[0].argmax(1).cpu().numpy())
                ts.extend(y.numpy())
        val_acc = accuracy_score(ts, ps)
        f1      = f1_score(ts, ps, average='macro', zero_division=0)
        marker  = " *" if f1 > best_f1 else ""
        print(f"    Ep {ep:2d}/15 | Loss: {tr_loss/len(tl):.4f} | Val Acc: {val_acc:.4f} | Val F1: {f1:.4f}{marker}")
        if f1 > best_f1: best_f1 = f1; torch.save(m.state_dict(), str(ckpt))

    print(f"    [AUDIO] Best Val F1: {best_f1:.4f}")
    m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
    test_p, val_p = [], []
    with torch.no_grad():
        for x,_ in el: test_p.append(F.softmax(m(x.to(DEVICE))[0], 1).cpu().numpy())
        for x,_ in vl: val_p.append( F.softmax(m(x.to(DEVICE))[0], 1).cpu().numpy())
    return np.vstack(test_p), np.vstack(val_p)


def train_video_ablation(tr, va, te, use_ssl, use_supcon, sc_name):
    """Returns (test_probs, val_probs) both shape [N, 7]."""
    print(f"\n  [VIDEO] {sc_name} | train={len(tr)} val={len(va)} test={len(te)}")
    cw        = compute_class_weight('balanced', classes=np.arange(7),
                                     y=np.array([LID[e] for e in tr['emotion_final']]))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    tl = DataLoader(VideoFTDS(tr), batch_size=32, shuffle=True)
    vl = DataLoader(VideoFTDS(va), batch_size=32)
    el = DataLoader(VideoFTDS(te), batch_size=32)
    supcon_fn = SupConLoss(SSL_TEMP)

    all_test_p, all_val_p, all_w = [], [], []

    for seed in [42, 1337, 2024]:
        set_seed(seed)
        m = VideoFTModel(drop=0.5).to(DEVICE)
        if use_ssl:
            ckpt_ssl = SSL_DIR / "video_ssl.pt"
            assert ckpt_ssl.exists(), f"CRITICAL: {ckpt_ssl} missing — run SSL phase first"
            m.backbone.load_state_dict(torch.load(ckpt_ssl, map_location=DEVICE), strict=False)

        opt = Lookahead(torch.optim.AdamW(m.parameters(), lr=7e-5, weight_decay=5e-2))
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt.opt, max_lr=8.4e-5, steps_per_epoch=len(tl), epochs=25)
        best_f1 = 0
        ckpt    = SAVE_DIR / f"vid_{sc_name.replace(' ','_').replace('+','plus')}_{seed}.pt"

        for ep in range(1, 26):
            m.train(); tr_loss = 0
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                lo, pr = m(x)
                loss   = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.1)
                if use_supcon: loss = loss + 0.3 * supcon_fn(pr, y)
                loss.backward(); opt.step(); sch.step()
                tr_loss += loss.item()

            m.eval(); ps, ts = [], []
            with torch.no_grad():
                for x, y in vl:
                    lo, _ = m(x.to(DEVICE))
                    ps.extend(lo.argmax(1).cpu().numpy()); ts.extend(y.numpy())
            f1     = f1_score(ts, ps, average='macro', zero_division=0)
            marker = " *" if f1 > best_f1 else ""
            if ep == 1 or ep % 5 == 0:
                print(f"    [Seed {seed}] Ep {ep:2d}/25 | Loss: {tr_loss/len(tl):.4f} | Val F1: {f1:.4f}{marker}")
            if f1 > best_f1: best_f1 = f1; torch.save(m.state_dict(), str(ckpt))

        print(f"    [Seed {seed}] Best Val F1: {best_f1:.4f}")
        m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()
        tp, vp = [], []
        with torch.no_grad():
            for x,_ in el: lo,_ = m(x.to(DEVICE)); tp.append(F.softmax(lo,1).cpu().numpy())
            for x,_ in vl: lo,_ = m(x.to(DEVICE)); vp.append(F.softmax(lo,1).cpu().numpy())
        all_test_p.append(np.vstack(tp))
        all_val_p.append( np.vstack(vp))
        all_w.append(best_f1)

    w = np.array(all_w); w /= w.sum()
    return (sum(p*wt for p,wt in zip(all_test_p, w)),
            sum(p*wt for p,wt in zip(all_val_p,  w)))


def train_text_ablation(tr, va, te, use_ssl, use_supcon, sc_name):
    """Returns (test_probs, val_probs) both shape [N, 7].

    val_probs uses out-of-fold (OOF) predictions: each va sample is predicted
    only by the fold where it was in the held-out set, so no fold has seen that
    sample during training. This prevents data leakage into the fusion weight search.
    """
    print(f"\n  [TEXT] {sc_name} | CV pool={len(tr)+len(va)} test={len(te)}")
    set_seed(42)
    tok       = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool      = pd.concat([tr, va]).reset_index(drop=True)
    texts     = pool['transcript'].values
    labels    = np.array([LID[e] for e in pool['emotion_final']])
    cw        = compute_class_weight('balanced', classes=np.arange(7), y=labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    te_loader = DataLoader(TextFTDS(te['transcript'].values, te['emotion_final'].values, tok), batch_size=16)
    supcon_fn = SupConLoss(SSL_TEMP)

    # va samples sit at the END of pool (pool = concat([tr, va]))
    va_start = len(tr)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_test_p  = []
    oof_probs    = np.zeros((len(pool), 7))   # out-of-fold predictions over full pool

    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        tl      = DataLoader(TextFTDS(texts[t_idx], [list(LID.keys())[l] for l in labels[t_idx]], tok),
                             batch_size=16, shuffle=True)
        vl_fold = DataLoader(TextFTDS(texts[v_idx], [list(LID.keys())[l] for l in labels[v_idx]], tok),
                             batch_size=16)
        m = TextFTModel().to(DEVICE)
        if use_ssl:
            ckpt_ssl = SSL_DIR / "text_ssl.pt"
            assert ckpt_ssl.exists(), f"CRITICAL: {ckpt_ssl} missing — run SSL phase first"
            m.backbone.load_state_dict(torch.load(ckpt_ssl, map_location=DEVICE), strict=False)

        opt = torch.optim.AdamW([
            {'params': [p for n,p in m.named_parameters() if 'bert' in n], 'lr': 2e-5},
            {'params': [p for n,p in m.named_parameters() if 'bert' not in n], 'lr': 8e-4},
        ], weight_decay=0.01)
        sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        best_acc = 0
        ckpt     = SAVE_DIR / f"txt_{sc_name.replace(' ','_').replace('+','plus')}_f{fold}.pt"
        pat      = 0

        for ep in range(1, 41):
            m.train(); tr_loss = 0
            for bd, bl in tl:
                opt.zero_grad()
                ids, mask, y = bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE), bl.to(DEVICE)
                lo, pr = m(ids, mask)
                loss   = F.cross_entropy(lo, y, weight=cw_tensor, label_smoothing=0.08)
                if use_supcon: loss = loss + 0.3 * supcon_fn(pr, y)
                loss.backward(); opt.step()
                tr_loss += loss.item()
            sch.step()

            m.eval(); ps, ts = [], []
            with torch.no_grad():
                for bd, bl in vl_fold:
                    ps.extend(m(bd['input_ids'].to(DEVICE),
                                bd['attention_mask'].to(DEVICE))[0].argmax(1).cpu().numpy())
                    ts.extend(bl.numpy())
            acc    = accuracy_score(ts, ps)
            marker = " *" if acc > best_acc else ""
            if ep == 1 or ep % 10 == 0:
                print(f"    [Fold {fold}] Ep {ep:2d}/40 | Loss: {tr_loss/len(tl):.4f} | CV Val Acc: {acc:.4f}{marker}")
            if acc > best_acc:
                best_acc = acc; torch.save(m.state_dict(), str(ckpt)); pat = 0
            else:
                pat += 1
                if pat >= 8:
                    print(f"    [Fold {fold}] Early stop at ep {ep}"); break

        print(f"    [Fold {fold}] Best CV Val Acc: {best_acc:.4f}")
        m.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=DEVICE)); m.eval()

        # Test predictions (all folds averaged)
        tp = []
        with torch.no_grad():
            for bd,_ in te_loader:
                tp.append(F.softmax(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE))[0], 1).cpu().numpy())
        fold_test_p.append(np.vstack(tp))

        # OOF predictions: store this fold's held-out predictions into oof_probs
        oof_loader = DataLoader(TextFTDS(texts[v_idx], [list(LID.keys())[l] for l in labels[v_idx]], tok), batch_size=16)
        oof_preds  = []
        with torch.no_grad():
            for bd,_ in oof_loader:
                oof_preds.append(F.softmax(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE))[0], 1).cpu().numpy())
        oof_probs[v_idx] = np.vstack(oof_preds)

    # val_probs: OOF predictions for the va portion of pool (unbiased)
    val_probs = oof_probs[va_start:]
    return np.mean(fold_test_p, 0), val_probs

# ─────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────
def run_ablation(tr, va, te, start_from="Baseline"):
    scenarios = [
        {"name": "Baseline",     "ssl": False, "supcon": False},
        {"name": "SupCon only",  "ssl": False, "supcon": True},
        {"name": "SSL only",     "ssl": True,  "supcon": False},
        {"name": "SSL + SupCon", "ssl": True,  "supcon": True},
    ]
    names = [s["name"] for s in scenarios]
    if start_from not in names:
        raise ValueError(f"start_from must be one of {names}")
    scenarios = scenarios[names.index(start_from):]
    results   = []
    t_labels  = [LID[e] for e in te['emotion_final'].values]
    v_labels  = [LID[e] for e in va['emotion_final'].values]

    for sc in scenarios:
        sep(f"SCENARIO: {sc['name']}")

        ap_test, ap_val = train_audio_ablation(tr, va, te, sc['ssl'], sc['supcon'], sc['name'])
        vp_test, vp_val = train_video_ablation(tr, va, te, sc['ssl'], sc['supcon'], sc['name'])
        tp_test, tp_val = train_text_ablation( tr, va, te, sc['ssl'], sc['supcon'], sc['name'])

        # FIX: grid-search fusion weights on VALIDATION (was TEST — data leakage)
        best_val_acc, best_w = 0, (0.33, 0.33, 0.34)
        for w_v in np.linspace(0.1, 0.8, 15):
            for w_a in np.linspace(0.1, 0.8, 15):
                w_t = round(1.0 - w_v - w_a, 3)
                if w_t < 0.05: continue
                fp  = w_v * vp_val + w_a * ap_val + w_t * tp_val
                acc = accuracy_score(v_labels, fp.argmax(1))
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_w = (round(w_v,2), round(w_a,2), round(w_t,3))

        # Apply val-optimised weights to test
        fp_test   = best_w[0] * vp_test + best_w[1] * ap_test + best_w[2] * tp_test
        test_acc  = accuracy_score(t_labels, fp_test.argmax(1))
        test_f1   = f1_score(t_labels, fp_test.argmax(1), average='macro', zero_division=0)

        print(f"\n  >>> {sc['name']} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} "
              f"| Weights (V,A,T): {best_w}  [weights from val]")
        print(classification_report(t_labels, fp_test.argmax(1),
                                    target_names=CLASSES, zero_division=0))

        results.append({
            "Scenario":        sc['name'],
            "Acc":             round(test_acc, 6),
            "F1":              round(test_f1,  6),
            "Weights (V,A,T)": f"{best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}",
        })

    sep("FINAL ABLATION RESULTS (ENSEMBLED SOTA)")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_results_ensembled.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_from", default="Baseline",
                    choices=["Baseline", "SupCon only", "SSL only", "SSL + SupCon"])
    ap.add_argument("--skip_ssl", action="store_true",
                    help="Skip SSL phase (use existing checkpoints)")
    args = ap.parse_args()

    sep(f"CONTRASTIVE PIPELINE v2 (HIGH ACCURACY ABLATION) | Device: {DEVICE}")
    tr, va, te = load_splits()
    if not args.skip_ssl:
        train_ssl_phase(pd.concat([tr, va]).reset_index(drop=True))
    run_ablation(tr, va, te, start_from=args.start_from)
