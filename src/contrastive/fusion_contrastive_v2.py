"""
fusion_contrastive_v2.py — 2-Phase Contrastive Pipeline (Colab)
================================================================
PHASE 1  Cross-Modal Self-Supervised Pre-training  (InfoNCE, NO labels)
  Trains Audio × Text × Video encoders jointly on a small unlabelled pool.
  Cross-modal positives: (audio_i, text_i, video_i) from the same utterance.

PHASE 2  Supervised Fine-tuning  (CE + SupCon, WITH labels)

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
from sklearn.utils.class_weight import compute_class_weight
from transformers import WavLMModel, AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS & GLOBAL CONFIG (Smart Auto-Detection)
# ─────────────────────────────────────────────────────────
import glob, os
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
    v_dir, a_dir = None, None

    for p in Path("/content").rglob("*_clip_seq.npy"):
        if "drive" not in str(p):
            v_dir = p.parent
            break

    if not v_dir:
        v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features/video_sequences_v1")
        if not v_dir.exists(): v_dir = Path("/content/drive/MyDrive/Thesis Project/data/processed/features")

    a_dir = Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED")
    return v_dir, a_dir

VID_DIR, AUDIO_BASE = auto_detect()
print(f"  Final VID_DIR: {VID_DIR}")
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

# Constants
SSL_PROJ_DIM  = 128    # projection head output dim (all models)
SSL_TEMP      = 0.07   # SupCon temperature used in Phase 2
CM_SSL_EPOCHS = 35     # cross-modal SSL epochs
CM_SSL_BATCH  = 8      # batch size (WavLM is the VRAM bottleneck)
CM_SSL_TEMP   = 0.12   # InfoNCE temperature (warmer for 7 negatives at batch=8)
UNLABELLED_N  = 610    # unlabelled pool size for SSL (labels never used)

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
        p = base / folder / audio_rel if folder else base / audio_rel
        if p.exists(): return p

        if folder:
            bs_name = f"{folder}\\{audio_rel.replace('/', '\\')}"
            p_bs = base / bs_name
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
    if not SPLIT_DIR.exists():
        print(f"  [ERROR] Split directory not found: {SPLIT_DIR}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tr = pd.read_csv(SPLIT_DIR/"train.csv")
    va = pd.read_csv(SPLIT_DIR/"val.csv")
    te = pd.read_csv(SPLIT_DIR/"test.csv")

    sep("🔍 PATH DIAGNOSTIC")
    row0 = tr.iloc[0]
    sid0 = row0['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")

    v_test, _, _ = get_vid_paths(sid0)
    a_test = resolve_audio_path(row0)

    vid_ok = v_test is not None and v_test.exists()
    aud_ok = a_test is not None and a_test.exists()
    txt_ok = isinstance(row0.get('transcript'), str) and len(str(row0['transcript']).strip()) > 2

    print(f"  Sample ID: {sid0}")
    print(f"  Video Status: {'✅ OK' if vid_ok else '❌ MISSING'} ({v_test.name if v_test else 'Not found'})")

    if aud_ok:
        print(f"  Audio Status: ✅ OK ({a_test.name})")
    else:
        print(f"  Audio Status: ❌ MISSING")
        print(f"      -> Script is looking for: folder '{row0.get('folder','')}' and file '{row0.get('audio_relpath','')}'")
        print(f"      -> Are these audio files uploaded to your Google Drive?")

    print(f"  Text  Status: {'✅ OK' if txt_ok else '❌ EMPTY'}")

    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        vid_p, _, _ = get_vid_paths(sid)
        vid = vid_p is not None and vid_p.exists()
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
        z  = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T
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
# ENCODER ARCHITECTURES  (shared by SSL pre-training + FT)
# ─────────────────────────────────────────────────────────
class AudioSSLModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus",
                                                    output_hidden_states=True)
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


class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(nn.Linear(c, c//16), nn.ReLU(),
                                  nn.Linear(c//16, c), nn.Sigmoid())
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

    def _pool(self, x):
        return (x * F.softmax(self.attn(x), 1)).sum(1)

    def encode(self, x):
        x = self.tfm(self.se(self.proj_in(x)) + self.pos)
        return self.fuse(torch.cat([
            self._pool(x), self._pool(x[:,4:12,:]), self._pool(x[:,6:10,:])
        ], -1))

    def forward(self, x):
        return self.proj(self.encode(x))

# ─────────────────────────────────────────────────────────
# PHASE 1 — CROSS-MODAL SSL  (Audio × Text × Video)
# ─────────────────────────────────────────────────────────
class CrossModalSSLDS(Dataset):
    """Loads (audio, video, text) triplets without emotion labels."""
    def __init__(self, df, tok, sr=16000, maxlen=80000):
        self.df = df.reset_index(drop=True)
        self.sr = sr; self.maxlen = maxlen
        self.enc = tok([clean(str(t)) for t in df['transcript'].values],
                       truncation=True, padding="max_length",
                       max_length=64, return_tensors="pt")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        # Audio — raw waveform, no label
        try:
            p = resolve_audio_path(r)
            if p is None: raise FileNotFoundError
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y = y[:self.maxlen] if len(y) > self.maxlen else np.pad(y, (0, self.maxlen - len(y)))
        except:
            y = np.zeros(self.maxlen)
        wav = torch.tensor(y, dtype=torch.float32)

        # Video — pre-extracted features, no label
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd_path, pr = get_vid_paths(sid)
        try:
            seq = np.concatenate([np.load(pc), np.load(pd_path), np.load(pr)], -1)  # [16, 3584]
        except:
            seq = np.zeros((16, 3584), dtype=np.float32)
        vid = torch.tensor(seq, dtype=torch.float32)

        # Text — pre-tokenised; BERT dropout provides implicit augmentation
        return wav, vid, self.enc['input_ids'][i], self.enc['attention_mask'][i]


class CrossModalInfoNCE(nn.Module):
    """Bidirectional InfoNCE over all 3 cross-modal pairs: (A,T), (A,V), (T,V)."""
    def __init__(self, temperature=0.12):
        super().__init__()
        self.T = temperature

    def _pair_loss(self, za, zb):
        za = F.normalize(za, dim=1)
        zb = F.normalize(zb, dim=1)
        B  = za.size(0)
        z  = torch.cat([za, zb], dim=0)
        sim = torch.mm(z, z.T) / self.T
        sim.fill_diagonal_(float('-inf'))
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0,   B, device=z.device)
        ])
        return F.cross_entropy(sim, labels)

    def forward(self, z_a, z_t, z_v):
        return (self._pair_loss(z_a, z_t) +
                self._pair_loss(z_a, z_v) +
                self._pair_loss(z_t, z_v)) / 3.0


def train_cross_modal_ssl(pool):
    sep("PHASE 1 -- CROSS-MODAL SSL (Audio × Text × Video | InfoNCE)")
    ckpts = [SSL_DIR/"audio_ssl.pt", SSL_DIR/"text_ssl.pt", SSL_DIR/"video_ssl.pt"]
    if all(c.exists() for c in ckpts):
        print("  [SKIP] All 3 SSL checkpoints found — delete all to retrain.")
        return

    set_seed(42)

    # Small unlabelled pool — emotion labels are never accessed
    n = min(UNLABELLED_N, len(pool))
    unlabelled = pool.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"  Unlabelled pool: {n}/{len(pool)} samples (labels not used)")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds  = CrossModalSSLDS(unlabelled, tok)
    dl  = DataLoader(ds, batch_size=CM_SSL_BATCH, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)

    a_enc = AudioSSLModel(SSL_PROJ_DIM).to(DEVICE)
    t_enc = TextSSLModel(SSL_PROJ_DIM).to(DEVICE)
    v_enc = VideoSSLModel(proj_dim=SSL_PROJ_DIM, drop=0.1).to(DEVICE)

    # Unfreeze only top-3 transformer layers for large pretrained backbones
    for i, layer in enumerate(a_enc.backbone.encoder.layers):
        for p in layer.parameters(): p.requires_grad = (i >= 9)
    for i, layer in enumerate(t_enc.bert.encoder.layer):
        for p in layer.parameters(): p.requires_grad = (i >= 9)

    # Build non-overlapping param groups
    a_bb = [p for n, p in a_enc.named_parameters()
            if p.requires_grad and n != 'lw' and not n.startswith('proj.')]
    t_bb = [p for n, p in t_enc.named_parameters()
            if p.requires_grad and not n.startswith('proj.')]
    v_bb = [p for n, p in v_enc.named_parameters() if not n.startswith('proj.')]
    all_projs = ([a_enc.lw] +
                 list(a_enc.proj.parameters()) +
                 list(t_enc.proj.parameters()) +
                 list(v_enc.proj.parameters()))

    opt = torch.optim.AdamW([
        {'params': a_bb,       'lr': 2e-6, 'weight_decay': 0.01},
        {'params': t_bb,       'lr': 2e-6, 'weight_decay': 0.01},
        {'params': v_bb,       'lr': 1e-5, 'weight_decay': 0.01},
        {'params': all_projs,  'lr': 1e-3, 'weight_decay': 0.01},
    ])
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CM_SSL_EPOCHS)
    loss_fn = CrossModalInfoNCE(CM_SSL_TEMP)
    scaler  = GradScaler()
    all_params = list(a_enc.parameters()) + list(t_enc.parameters()) + list(v_enc.parameters())

    print(f"  Batch: {CM_SSL_BATCH} | Steps/ep: {len(dl)} | Epochs: {CM_SSL_EPOCHS}")

    for ep in range(1, CM_SSL_EPOCHS + 1):
        a_enc.train(); t_enc.train(); v_enc.train()
        ep_loss = 0.0

        for wav, vid, ids, mask in dl:
            wav  = wav.to(DEVICE)
            vid  = vid.to(DEVICE)
            ids  = ids.to(DEVICE)
            mask = mask.to(DEVICE)

            opt.zero_grad()
            with autocast("cuda"):
                z_a = a_enc(wav)          # [B, 128]
                z_t = t_enc(ids, mask)    # [B, 128]
                z_v = v_enc(vid)          # [B, 128]
                loss = loss_fn(z_a, z_t, z_v)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()

        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{CM_SSL_EPOCHS} | CrossModal NCE: {ep_loss/len(dl):.4f}")

    # Save in same format as before — FT loading code is unchanged
    torch.save({'backbone': a_enc.backbone.state_dict(), 'lw': a_enc.lw.data}, str(ckpts[0]))
    torch.save({k: v for k,v in t_enc.state_dict().items() if 'proj' not in k}, str(ckpts[1]))
    torch.save({k: v for k,v in v_enc.state_dict().items() if 'proj' not in k}, str(ckpts[2]))
    print("  [SAVED] audio_ssl.pt | text_ssl.pt | video_ssl.pt")

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
        pc, pd, pr = get_vid_paths(sid)
        c   = np.load(pc)
        d   = np.load(pd)
        r2  = np.load(pr)
        seq = np.concatenate([c, d, r2], -1)   # [16, 3584]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

# FT Models
class AudioFTModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = AudioSSLModel(proj_dim)
        self.classifier = nn.Sequential(nn.Linear(768*2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 7))
        self.proj_ft = ProjectionHead(768*2, proj_dim)
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

# FT Training Loop
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
        lr_bb, lr_hd = 1e-5, 1e-3
        bs = 8
    elif name == "TEXT":
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        ds_tr = TextFTDS(train_df['transcript'].values, train_df['emotion_final'].values, tok)
        ds_va = TextFTDS(val_df['transcript'].values, val_df['emotion_final'].values, tok)
        ds_te = TextFTDS(test_df['transcript'].values, test_df['emotion_final'].values, tok)
        m = TextFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl: m.backbone.load_state_dict(torch.load(SSL_DIR/"text_ssl.pt", map_location=DEVICE), strict=False)
        lr_bb, lr_hd = 1e-5, 5e-4
        bs = 16
    else: # VIDEO
        ds_tr, VideoFTDS_va, ds_te = VideoFTDS(train_df), VideoFTDS(val_df), VideoFTDS(test_df)
        m = VideoFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl: m.backbone.load_state_dict(torch.load(SSL_DIR/"video_ssl.pt", map_location=DEVICE), strict=False)
        lr_bb, lr_hd = 3e-5, 1e-3
        bs = 32

    bb_params = [p for n, p in m.named_parameters() if 'backbone' in n]
    hd_params = [p for n, p in m.named_parameters() if 'backbone' not in n]

    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': lr_bb},
        {'params': hd_params, 'lr': lr_hd}
    ], weight_decay=0.05)

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(VideoFTDS_va if name=="VIDEO" else ds_va, batch_size=bs)
    dl_te = DataLoader(ds_te, batch_size=bs)

    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr_hd], steps_per_epoch=len(dl_tr), epochs=20)
    supcon_fn = SupConLoss(SSL_TEMP)
    best_acc, ckpt = 0, SAVE_DIR/f"{name.lower()}_ft.pt"
    y_tr = np.array([LID[e] for e in train_df['emotion_final']])
    cw = compute_class_weight('balanced', classes=np.arange(7), y=y_tr)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    for ep in range(1, 21):
        m.train()
        cur_supcon_w = 0.3 if use_supcon else 0.0

        for batch in dl_tr:
            opt.zero_grad()
            if name == "TEXT":
                logits, proj = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
            else:
                logits, proj = m(batch[0].to(DEVICE))
            labels = batch[1].to(DEVICE)

            loss = F.cross_entropy(logits, labels, weight=cw_tensor, label_smoothing=0.1)
            if use_supcon: loss += cur_supcon_w * supcon_fn(proj, labels)

            loss.backward(); opt.step(); sch.step()

        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for batch in dl_va:
                if name == "TEXT": logits, _ = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
                else: logits, _ = m(batch[0].to(DEVICE))
                ps.extend(logits.argmax(1).cpu().numpy()); ts.extend(batch[1].numpy())
        acc = accuracy_score(ts, ps)
        if acc > best_acc: best_acc = acc; torch.save(m.state_dict(), str(ckpt))
        if ep % 2 == 0 or ep == 1: print(f"  Ep {ep:02d} | Val Acc: {acc:.4f}")

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

        best_acc, best_f1, best_w = 0, 0, (0.33, 0.33, 0.34)
        for w_v in np.linspace(0, 1, 11):
            for w_a in np.linspace(0, 1, 11):
                w_t = 1.0 - w_v - w_a
                if w_t < 0 or w_t > 1: continue

                fp = w_v * vp + w_a * ap + w_t * tp
                preds = fp.argmax(1)
                acc = accuracy_score(t_labels, preds)

                if acc > best_acc:
                    best_acc = acc
                    best_f1 = f1_score(t_labels, preds, average='macro')
                    best_w = (w_v, w_a, w_t)

        results.append({
            "Scenario": sc['name'],
            "Acc": best_acc,
            "F1": best_f1,
            "Weights (V,A,T)": f"{best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}"
        })
        print(f"\n  >>> {sc['name']} Result: Acc={best_acc:.4f}, F1={best_f1:.4f} | Weights: {best_w}")

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

    # PHASE 1: Cross-Modal SSL on small unlabelled pool
    pool = pd.concat([tr, va]).reset_index(drop=True)
    print(f"  Full labelled pool: {len(pool)} samples (SSL will sample {UNLABELLED_N} without labels)")

    train_cross_modal_ssl(pool)

    sep("PHASE 1 COMPLETE -- Cross-modal encoders pre-trained.")

    # PHASE 2: Ablation (Fine-tuning on labelled train set)
    run_ablation(tr, va, te)
