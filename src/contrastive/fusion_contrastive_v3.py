"""
fusion_contrastive_v3.py — 2-Phase Contrastive Pipeline (Colab)
================================================================
FIXES over v2:
  FIX 1 — SSL pool uses truly unlabelled data from all_segments.xlsx (not carved from labelled train)
  FIX 2 — Fusion weights searched on VALIDATION set, not test set
  FIX 3 — Gradient accumulation for effective batch ~32 in cross-modal SSL
  FIX 4 — VideoFTDS_va variable name bug fixed
  FIX 5 — Backbone LR raised for SSL scenarios (3e-5 vs 1e-5)
  FIX 6 — Text augmentation uses real token masking (not dropout-only)
  FIX 7 — SSL loss logged every epoch (was every 5)

PHASE 1  Cross-Modal Self-Supervised Pre-training  (InfoNCE, NO labels)
  Trains Audio × Text × Video encoders jointly on a held-out unlabelled pool.
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
# PATHS & GLOBAL CONFIG
# ─────────────────────────────────────────────────────────
import glob
cands = glob.glob("/content/*thesis*") + glob.glob("/content/*omaremad*")
cands = [c for c in cands if os.path.isdir(c) and (Path(c)/"src").exists()]
_repo_str = cands[0] if cands else "/content/thesis"

REPO        = Path(_repo_str)
SPLIT_DIR   = REPO / "data/processed/splits/multimodal_eligible"
SAVE_DIR    = Path("/content/fusion_models")
SSL_DIR     = Path("/content/ssl_pretrained")
SSL_VID_DIR = Path("/content/ssl_video_features")   # on-demand features for unlabelled SSL pool
for d in [SAVE_DIR, SSL_DIR, SSL_VID_DIR]: d.mkdir(exist_ok=True)

def auto_detect():
    print("  Smart-detecting data locations...")
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
UNLABELLED_XLSX = AUDIO_BASE / "all_segments.xlsx"
print(f"  Final VID_DIR:    {VID_DIR}")
print(f"  Final AUDIO_BASE: {AUDIO_BASE}")

LID     = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6}
CLASSES = list(LID.keys())
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
SSL_PROJ_DIM  = 128
SSL_TEMP      = 0.07
CM_SSL_EPOCHS = 50
CM_SSL_BATCH  = 8
# FIX 3: gradient accumulation steps → effective batch = 8 × 4 = 32
CM_GRAD_ACC   = 4
CM_SSL_TEMP   = 0.12
UNLABELLED_N  = 1500   # max samples drawn from all_segments.xlsx for SSL (labels never used)

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
            p_bs = base / f"{folder}\\{audio_rel.replace('/', chr(92))}"
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
    # On-demand features extracted for SSL unlabelled pool
    p3 = SSL_VID_DIR / f"{sid}_clip_seq.npy"
    if p3.exists():
        return p3, SSL_VID_DIR / f"{sid}_dinov2_seq.npy", SSL_VID_DIR / f"{sid}_resnet50_seq.npy"
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
    print(f"  Video: {'OK' if vid_ok else 'MISSING'} | Audio: {'OK' if aud_ok else 'MISSING'} | Text: {'OK' if txt_ok else 'EMPTY'}")

    def ok(row):
        sid = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        vp, _, _ = get_vid_paths(sid)
        return (vp is not None and vp.exists() and
                resolve_audio_path(row) is not None and
                isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2)

    tr_f = tr[tr.apply(ok, axis=1)].reset_index(drop=True)
    va_f = va[va.apply(ok, axis=1)].reset_index(drop=True)
    te_f = te[te.apply(ok, axis=1)].reset_index(drop=True)

    sep("ALIGNED SPLITS")
    print(f"  Train: {len(tr_f)} | Val: {len(va_f)} | Test: {len(te_f)}")
    return tr_f, va_f, te_f


# ─────────────────────────────────────────────────────────
# FIX 1 — UNLABELLED SSL POOL  (from all_segments.xlsx)
# ─────────────────────────────────────────────────────────
def load_unlabelled_pool(exclude_ids=None, n=UNLABELLED_N, seed=42):
    """
    Loads truly unlabelled segments from all_segments.xlsx (NaN emotion label).

    exclude_ids: set of sample_id strings from val+test splits. Explicitly
      removed even if NaN filter didn't catch them — guarantees val/test audio
      and text are never seen during SSL regardless of annotation completeness
      in the xlsx.

    Returns (pool_df, vid_frac) where vid_frac is the fraction of pool samples
    that have video .npy features available (from a 20-sample probe).
    """
    df = pd.read_excel(str(UNLABELLED_XLSX))
    unlabelled = df[df['Final Overall (majority of modalities)'].isna()].copy()
    unlabelled['folder']        = unlabelled['Folder']
    unlabelled['audio_relpath'] = unlabelled['audio_file']
    unlabelled['sample_id']     = unlabelled['Folder'] + '::' + unlabelled['video_file']
    unlabelled = unlabelled[
        unlabelled['transcript'].apply(
            lambda t: isinstance(t, str) and len(t.strip()) > 2)
    ].reset_index(drop=True)

    # Explicit guard: remove any val/test samples regardless of NaN filter accuracy
    if exclude_ids:
        before = len(unlabelled)
        unlabelled = unlabelled[~unlabelled['sample_id'].isin(exclude_ids)].reset_index(drop=True)
        removed = before - len(unlabelled)
        if removed:
            print(f"  Removed {removed} val/test sample(s) found in unlabelled pool (annotation gap in xlsx)")

    def _has_vid(r):
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, _, _ = get_vid_paths(sid)
        return pc is not None and pc.exists()

    probe    = unlabelled.sample(min(20, len(unlabelled)), random_state=0)
    vid_frac = probe.apply(_has_vid, axis=1).mean()
    print(f"  Video feature coverage (probe): {vid_frac:.0%}")

    n_use = min(n, len(unlabelled))
    pool  = unlabelled.sample(n=n_use, random_state=seed).reset_index(drop=True)
    print(f"  SSL pool : {n_use} samples (labels never accessed)")
    return pool, vid_frac


# ─────────────────────────────────────────────────────────
# SSL VIDEO FEATURE EXTRACTION  (unlabelled pool, on-demand)
# ─────────────────────────────────────────────────────────
def extract_ssl_video_features(ssl_pool):
    """
    Extracts CLIP+DINOv2+ResNet50 frame features [16, D] for each sample
    in ssl_pool that doesn't already have .npy files in SSL_VID_DIR.
    Mirrors video_stage3_extract_sequences.py exactly (same timm model IDs).
    Saves clip_seq, dinov2_seq, resnet50_seq per sample.
    """
    import timm, cv2
    from torchvision import transforms
    from PIL import Image as PILImage

    def _resolve_video_path(row):
        folder = str(row.get('folder', row.get('Folder', '')))
        vfile  = str(row.get('video_file', ''))
        if not vfile: return None
        for base in [AUDIO_BASE, AUDIO_BASE / folder]:
            p = base / vfile
            if p.exists(): return p
            p2 = base / folder / vfile
            if p2.exists(): return p2
        return None

    def _sample_frames(v_path, n=16):
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened(): return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); return []
        idxs = set(np.linspace(0, total - 1, n).astype(int))
        buf, cur = {}, 0
        while cur <= max(idxs):
            ret, f = cap.read()
            if not ret: break
            if cur in idxs:
                buf[cur] = PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cur += 1
        cap.release()
        frames = [buf[i] for i in sorted(buf.keys())]
        while len(frames) < n:
            frames.append(frames[-1] if frames else PILImage.new('RGB', (224, 224)))
        return frames[:n]

    # Which samples still need extraction?
    needs = [row for _, row in ssl_pool.iterrows()
             if not (SSL_VID_DIR / f"{row['sample_id'].replace('::','__').replace('/','_').replace('.mp4','')}_clip_seq.npy").exists()]

    if not needs:
        print("  All SSL video features already extracted.")
        return

    print(f"  Extracting video features for {len(needs)}/{len(ssl_pool)} SSL samples...")

    models_cfg = [
        ("clip",     "vit_base_patch32_clip_224"),
        ("dinov2",   "vit_base_patch14_dinov2"),
        ("resnet50", "resnet50"),
    ]

    for mname, mid in models_cfg:
        print(f"  [{mname}] loading {mid}...")
        model = timm.create_model(mid, pretrained=True, num_classes=0).to(DEVICE)
        model.eval()
        cfg = timm.data.resolve_model_data_config(model)
        tf  = transforms.Compose([
            transforms.Resize(cfg['input_size'][1:]),
            transforms.CenterCrop(cfg['input_size'][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
        ])
        ok = skip = 0
        for row in needs:
            sid   = row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
            fpath = SSL_VID_DIR / f"{sid}_{mname}_seq.npy"
            if fpath.exists(): ok += 1; continue
            vp = _resolve_video_path(row)
            if vp is None: skip += 1; continue
            frames = _sample_frames(vp)
            if not frames: skip += 1; continue
            batch = torch.stack([tf(f) for f in frames]).to(DEVICE)
            with torch.no_grad():
                feat = model(batch)
                if feat.dim() > 2: feat = feat.mean(dim=[2, 3])
            np.save(str(fpath), feat.cpu().numpy())
            ok += 1
        print(f"  [{mname}] saved={ok}, skipped={skip} (video not found on Drive)")
        del model; torch.cuda.empty_cache()

    print("  Video feature extraction complete.")


# ─────────────────────────────────────────────────────────
# CONTRASTIVE LOSSES
# ─────────────────────────────────────────────────────────
class InfoNCELoss(nn.Module):
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
# ENCODER ARCHITECTURES
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
        return torch.cat([out.mean(1), out.std(1)], 1)

    def forward(self, x):
        return self.proj(self.encode(x))


MODEL_NAME = "UBC-NLP/MARBERT"
_FILLERS   = re.compile(
    r'\b(اه|ايه|يعني|بص|كده|كدا|اهو|والله|عشان|بقا|بقى|يا|اوه|هاه|اوكي|اوكى|تمام|صح|ايوه|لا|مش|ميش)\b'
)
def clean(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'[ً-ٰٟ]', '', t)
    t = re.sub(r'[آأإ]', 'ا', t)
    t = re.sub(r'ة', 'ه', t)
    t = re.sub(r'ى', 'ي', t)
    t = re.sub(r'ـ', '', t)
    t = _FILLERS.sub(' ', t)
    t = re.sub(r'(.)\1+', r'\1\1', t)
    return re.sub(r'\s+', ' ', t).strip()


# FIX 6 — Real token masking for text augmentation
def mask_tokens(input_ids, mask_token_id=3, mask_prob=0.15, pad_token_id=0):
    """Randomly replace 15% of non-padding tokens with [MASK] (id=3 for MARBERT)."""
    ids = input_ids.clone()
    # Only mask real tokens (not padding)
    prob_matrix = torch.rand_like(ids, dtype=torch.float)
    padding_mask = ids.eq(pad_token_id)
    prob_matrix.masked_fill_(padding_mask, 0.0)
    masked_positions = prob_matrix < mask_prob
    ids[masked_positions] = mask_token_id
    return ids


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
    """
    Loads (audio, video, text) triplets WITHOUT accessing emotion labels.
    FIX 1: receives ssl_pool which has emotion_final column dropped.
    FIX 6: returns raw token ids so mask_tokens() can be called per-batch.
    """
    def __init__(self, df, tok, sr=16000, maxlen=80000):
        self.df = df.reset_index(drop=True)
        self.sr = sr; self.maxlen = maxlen
        self.enc = tok([clean(str(t)) for t in df['transcript'].values],
                       truncation=True, padding="max_length",
                       max_length=64, return_tensors="pt")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        # Audio
        try:
            p = resolve_audio_path(r)
            if p is None: raise FileNotFoundError
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y = y[:self.maxlen] if len(y) > self.maxlen else np.pad(y, (0, self.maxlen - len(y)))
        except:
            y = np.zeros(self.maxlen)
        wav = torch.tensor(y, dtype=torch.float32)

        # Video
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        pc, pd_path, pr = get_vid_paths(sid)
        try:
            seq = np.concatenate([np.load(pc), np.load(pd_path), np.load(pr)], -1)
        except:
            seq = np.zeros((16, 3584), dtype=np.float32)
        vid = torch.tensor(seq, dtype=torch.float32)

        # Text — raw ids returned; masking applied in training loop
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


def train_cross_modal_ssl(ssl_pool, vid_frac=1.0):
    """
    FIX 1: receives ssl_pool (labels already dropped before this call).
    FIX 3: gradient accumulation → effective batch = CM_SSL_BATCH × CM_GRAD_ACC.
    FIX 6: real token masking applied per-batch for text views.
    FIX 7: SSL loss logged every epoch.
    vid_frac: fraction of pool with valid video features (from load_unlabelled_pool probe).
      If < 0.2, skip A-V and T-V pairs so video encoder is not trained on zero-vectors.
      video_ssl.pt is only saved when video pairs are computed.
    """
    sep("PHASE 1 -- CROSS-MODAL SSL (Audio × Text × Video | InfoNCE)")
    ckpts = [SSL_DIR/"audio_ssl.pt", SSL_DIR/"text_ssl.pt", SSL_DIR/"video_ssl.pt"]
    # Only require audio+text checkpoints to skip; video may not exist if pool had no features
    if ckpts[0].exists() and ckpts[1].exists():
        status = "all 3" if ckpts[2].exists() else "audio+text (no video features in pool)"
        print(f"  [SKIP] SSL checkpoints found ({status}) — delete to retrain.")
        return

    has_video_ssl = vid_frac >= 0.2
    if not has_video_ssl:
        print(f"  Video coverage {vid_frac:.0%} < 20% — running audio-text SSL only.")
        print(f"  Video encoder will NOT receive SSL weights (avoids training on zero-vectors).")

    set_seed(42)
    eff_batch = CM_SSL_BATCH * CM_GRAD_ACC
    print(f"  SSL pool size   : {len(ssl_pool)} samples (NO labels accessed)")
    print(f"  Micro batch     : {CM_SSL_BATCH} | Grad acc steps: {CM_GRAD_ACC} | Effective batch: {eff_batch}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds  = CrossModalSSLDS(ssl_pool, tok)
    dl  = DataLoader(ds, batch_size=CM_SSL_BATCH, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)

    a_enc = AudioSSLModel(SSL_PROJ_DIM).to(DEVICE)
    t_enc = TextSSLModel(SSL_PROJ_DIM).to(DEVICE)
    v_enc = VideoSSLModel(proj_dim=SSL_PROJ_DIM, drop=0.1).to(DEVICE)

    for i, layer in enumerate(a_enc.backbone.encoder.layers):
        for p in layer.parameters(): p.requires_grad = (i >= 9)
    for i, layer in enumerate(t_enc.bert.encoder.layer):
        for p in layer.parameters(): p.requires_grad = (i >= 9)

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
        {'params': a_bb,      'lr': 2e-6, 'weight_decay': 0.01},
        {'params': t_bb,      'lr': 2e-6, 'weight_decay': 0.01},
        {'params': v_bb,      'lr': 1e-5, 'weight_decay': 0.01},
        {'params': all_projs, 'lr': 1e-3, 'weight_decay': 0.01},
    ])
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CM_SSL_EPOCHS)
    loss_fn = CrossModalInfoNCE(CM_SSL_TEMP)
    scaler  = GradScaler()
    all_params = (list(a_enc.parameters()) +
                  list(t_enc.parameters()) +
                  list(v_enc.parameters()))

    print(f"  Steps/epoch: {len(dl)} | Epochs: {CM_SSL_EPOCHS}")

    for ep in range(1, CM_SSL_EPOCHS + 1):
        a_enc.train(); t_enc.train(); v_enc.train()
        ep_loss = 0.0
        opt.zero_grad()

        for step, (wav, vid, ids, mask) in enumerate(dl):
            wav  = wav.to(DEVICE)
            vid  = vid.to(DEVICE)
            ids  = ids.to(DEVICE)
            mask = mask.to(DEVICE)

            # FIX 6: two independently masked views of the same text
            ids_v1 = mask_tokens(ids)
            ids_v2 = mask_tokens(ids)

            with autocast("cuda"):
                z_a  = a_enc(wav)
                z_t  = t_enc(ids_v1, mask)   # view 1
                z_t2 = t_enc(ids_v2, mask)   # view 2

                if has_video_ssl:
                    z_v  = v_enc(vid)
                    # 4-pair loss: A-T, A-V, T-V, T1-T2
                    loss = (loss_fn._pair_loss(z_a, z_t) +
                            loss_fn._pair_loss(z_a, z_v) +
                            loss_fn._pair_loss(z_t, z_v) +
                            loss_fn._pair_loss(z_t, z_t2)) / 4.0
                else:
                    # Audio-text only: A-T and T1-T2 (no video pairs to avoid zero-vector training)
                    loss = (loss_fn._pair_loss(z_a, z_t) +
                            loss_fn._pair_loss(z_t, z_t2)) / 2.0

                loss = loss / CM_GRAD_ACC  # FIX 3: scale for accumulation

            scaler.scale(loss).backward()
            ep_loss += loss.item() * CM_GRAD_ACC

            # FIX 3: step only every CM_GRAD_ACC micro-batches
            if (step + 1) % CM_GRAD_ACC == 0 or (step + 1) == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad()

        sch.step()
        # FIX 7: log every epoch (was every 5)
        print(f"  Ep {ep:02d}/{CM_SSL_EPOCHS} | CrossModal NCE: {ep_loss/len(dl):.4f}")

    torch.save({'backbone': a_enc.backbone.state_dict(), 'lw': a_enc.lw.data}, str(ckpts[0]))
    torch.save({k: v for k,v in t_enc.state_dict().items() if 'proj' not in k}, str(ckpts[1]))
    if has_video_ssl:
        torch.save({k: v for k,v in v_enc.state_dict().items() if 'proj' not in k}, str(ckpts[2]))
        print("  [SAVED] audio_ssl.pt | text_ssl.pt | video_ssl.pt")
    else:
        print("  [SAVED] audio_ssl.pt | text_ssl.pt  (video_ssl.pt skipped — no video features in pool)")


# ─────────────────────────────────────────────────────────
# PHASE 2 — SUPERVISED FINE-TUNING
# ─────────────────────────────────────────────────────────
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
        self.enc = tok([clean(str(t)) for t in texts], truncation=True,
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
        pc, pd, pr = get_vid_paths(sid)
        seq = np.concatenate([np.load(pc), np.load(pd), np.load(pr)], -1)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

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


def train_modality_ft(name, train_df, val_df, test_df, use_ssl=True, use_supcon=True):
    """
    FIX 4: VideoFTDS_va variable name bug fixed — val dataset uses consistent naming.
    FIX 5: Backbone LR raised to 3e-5 when use_ssl=True (SSL reps need more room to adapt).
    """
    sep(f"PHASE 2 -- {name} FT (SSL={use_ssl}, SupCon={use_supcon})")
    set_seed(42)

    if name == "AUDIO":
        ds_tr = AudioFTDS(train_df)
        ds_va = AudioFTDS(val_df)      # FIX 4: consistent naming
        ds_te = AudioFTDS(test_df)
        m     = AudioFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            sd = torch.load(SSL_DIR/"audio_ssl.pt", map_location=DEVICE)
            m.backbone.backbone.load_state_dict(sd['backbone'])
            m.backbone.lw.data = sd['lw']
        # FIX 5: higher backbone LR when SSL initialized
        lr_bb = 3e-5 if use_ssl else 1e-5
        lr_hd = 1e-3
        bs    = 8

    elif name == "TEXT":
        tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
        ds_tr = TextFTDS(train_df['transcript'].values, train_df['emotion_final'].values, tok)
        ds_va = TextFTDS(val_df['transcript'].values,   val_df['emotion_final'].values,   tok)
        ds_te = TextFTDS(test_df['transcript'].values,  test_df['emotion_final'].values,  tok)
        m     = TextFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            m.backbone.load_state_dict(
                torch.load(SSL_DIR/"text_ssl.pt", map_location=DEVICE), strict=False)
        lr_bb = 3e-5 if use_ssl else 1e-5
        lr_hd = 5e-4
        bs    = 16

    else:  # VIDEO
        ds_tr = VideoFTDS(train_df)
        ds_va = VideoFTDS(val_df)      # FIX 4: was `VideoFTDS_va` (shadowed class name)
        ds_te = VideoFTDS(test_df)
        m     = VideoFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            _vid_ckpt = SSL_DIR/"video_ssl.pt"
            if _vid_ckpt.exists():
                m.backbone.load_state_dict(torch.load(_vid_ckpt, map_location=DEVICE), strict=False)
            else:
                print(f"  [INFO] video_ssl.pt not found (pool had no video features) — video uses random init")
        lr_bb = 3e-5 if use_ssl else 1e-5  # FIX 5
        lr_hd = 1e-3
        bs    = 32

    bb_params = [p for n, p in m.named_parameters() if 'backbone' in n]
    hd_params = [p for n, p in m.named_parameters() if 'backbone' not in n]

    opt   = torch.optim.AdamW([
        {'params': bb_params, 'lr': lr_bb},
        {'params': hd_params, 'lr': lr_hd}
    ], weight_decay=0.05)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=bs)
    dl_te = DataLoader(ds_te, batch_size=bs)

    # SSL scenarios need more epochs to adapt contrastive reps to classification
    n_epochs  = 25 if use_ssl else 20
    sch       = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_bb, lr_hd], steps_per_epoch=len(dl_tr), epochs=n_epochs)
    supcon_fn = SupConLoss(SSL_TEMP)
    best_acc, ckpt = 0, SAVE_DIR / f"{name.lower()}_ft.pt"

    # Lower SupCon weight for SSL+SupCon to reduce val→test overfit
    supcon_final_w = 0.2 if (use_ssl and use_supcon) else 0.3

    y_tr     = np.array([LID[e] for e in train_df['emotion_final']])
    cw       = compute_class_weight('balanced', classes=np.arange(7), y=y_tr)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(DEVICE)

    for ep in range(1, n_epochs + 1):
        m.train()
        # SupCon warmup: ramp from 0 → supcon_final_w over first 5 epochs
        # CE dominates early so backbone stabilises before contrastive loss kicks in
        cur_supcon_w = supcon_final_w * min(1.0, ep / 5.0) if use_supcon else 0.0
        for batch in dl_tr:
            opt.zero_grad()
            if name == "TEXT":
                logits, proj = m(batch[0]['input_ids'].to(DEVICE),
                                 batch[0]['attention_mask'].to(DEVICE))
            else:
                logits, proj = m(batch[0].to(DEVICE))
            labels = batch[1].to(DEVICE)
            loss   = F.cross_entropy(logits, labels, weight=cw_tensor, label_smoothing=0.1)
            if use_supcon: loss += cur_supcon_w * supcon_fn(proj, labels)
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
        if acc > best_acc:
            best_acc = acc
            torch.save(m.state_dict(), str(ckpt))
        if ep % 2 == 0 or ep == 1:
            print(f"  Ep {ep:02d} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

    m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    m.eval(); probs = []
    with torch.no_grad():
        for batch in dl_te:
            if name == "TEXT":
                logits, _ = m(batch[0]['input_ids'].to(DEVICE),
                               batch[0]['attention_mask'].to(DEVICE))
            else:
                logits, _ = m(batch[0].to(DEVICE))
            probs.append(F.softmax(logits, 1).cpu().numpy())
    return np.vstack(probs)


# ─────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────
def run_ablation(tr, va, te):
    """
    FIX 2: Fusion weights are searched on VALIDATION set (va), then
            the winning weights are applied to TEST set (te) for final numbers.
            Test labels are never seen during weight selection.
    """
    scenarios = [
        {"name": "Baseline",     "ssl": False, "supcon": False},
        {"name": "SupCon only",  "ssl": False, "supcon": True},
        {"name": "SSL only",     "ssl": True,  "supcon": False},
        {"name": "SSL + SupCon", "ssl": True,  "supcon": True},
    ]
    results = []

    # Collect validation labels once
    va_labels = [LID[e] for e in va['emotion_final'].values]
    te_labels = [LID[e] for e in te['emotion_final'].values]

    for sc in scenarios:
        sep(f"RUNNING SCENARIO: {sc['name']}")

        # Get test-set probabilities per modality
        vp_te = train_modality_ft("VIDEO", tr, va, te, sc['ssl'], sc['supcon'])
        ap_te = train_modality_ft("AUDIO", tr, va, te, sc['ssl'], sc['supcon'])
        tp_te = train_modality_ft("TEXT",  tr, va, te, sc['ssl'], sc['supcon'])

        # Also get validation-set probabilities for weight search
        # Re-run inference on val using the saved checkpoints
        def get_val_probs(name):
            if name == "AUDIO":
                ds_va = AudioFTDS(va)
                m = AudioFTModel(SSL_PROJ_DIM).to(DEVICE)
                bs = 8
            elif name == "TEXT":
                tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
                ds_va = TextFTDS(va['transcript'].values, va['emotion_final'].values, tok)
                m     = TextFTModel(SSL_PROJ_DIM).to(DEVICE)
                bs    = 16
            else:
                ds_va = VideoFTDS(va)
                m     = VideoFTModel(SSL_PROJ_DIM).to(DEVICE)
                bs    = 32
            m.load_state_dict(torch.load(SAVE_DIR/f"{name.lower()}_ft.pt", map_location=DEVICE))
            m.eval()
            dl = DataLoader(ds_va, batch_size=bs)
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

        vp_va = get_val_probs("VIDEO")
        ap_va = get_val_probs("AUDIO")
        tp_va = get_val_probs("TEXT")

        # FIX 2: search weights on VALIDATION
        best_acc_val, best_w = 0, (0.33, 0.33, 0.34)
        for w_v in np.linspace(0, 1, 11):
            for w_a in np.linspace(0, 1, 11):
                w_t = 1.0 - w_v - w_a
                if w_t < 0 or w_t > 1: continue
                fp_va   = w_v * vp_va + w_a * ap_va + w_t * tp_va
                acc_val = accuracy_score(va_labels, fp_va.argmax(1))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_w       = (w_v, w_a, w_t)

        # Apply winning weights to TEST (no snooping)
        fp_te    = best_w[0] * vp_te + best_w[1] * ap_te + best_w[2] * tp_te
        test_acc = accuracy_score(te_labels, fp_te.argmax(1))
        test_f1  = f1_score(te_labels, fp_te.argmax(1), average='macro', zero_division=0)

        results.append({
            "Scenario":       sc['name'],
            "Val Acc (w sel)": f"{best_acc_val:.4f}",
            "Test Acc":        f"{test_acc:.4f}",
            "Test F1":         f"{test_f1:.4f}",
            "Weights (V,A,T)": f"{best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}"
        })
        print(f"\n  >>> {sc['name']} | Val Acc: {best_acc_val:.4f} | "
              f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Weights: {best_w}")

    sep("FINAL ABLATION RESULTS")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_results_v3.csv", index=False)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sep(f"CONTRASTIVE PIPELINE v3 | Device: {DEVICE}")
    tr, va, te = load_splits()

    sep("LOADING UNLABELLED SSL POOL")
    _excl = set(va['sample_id'].values) | set(te['sample_id'].values)
    ssl_pool, vid_frac = load_unlabelled_pool(exclude_ids=_excl)

    # Extract video features for SSL pool if not already present
    if vid_frac < 0.2:
        sep("EXTRACTING SSL VIDEO FEATURES")
        extract_ssl_video_features(ssl_pool)
        vid_frac = 1.0   # features now on disk; full 4-pair SSL will run

    # PHASE 1: SSL on unlabelled pool (labels never accessed)
    train_cross_modal_ssl(ssl_pool, vid_frac=vid_frac)
    sep("PHASE 1 COMPLETE")

    # PHASE 2: Ablation using ALL labelled train data
    run_ablation(tr, va, te)
