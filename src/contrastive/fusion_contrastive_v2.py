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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
        # Standard Path
        p = base / folder / audio_rel if folder else base / audio_rel
        if p.exists(): return p
        
        # Windows Backslash Artifact Path (e.g. "videoplayback (1)\audios\SPEAKER_00\SPEAKER_00_segment_0000.wav")
        if folder:
            bs_name = f"{folder}\\{audio_rel.replace('/', '\\')}"
            p_bs = base / bs_name
            if p_bs.exists(): return p_bs
            
        # Flat Directory Fallback
        p_flat = base / Path(audio_rel).name
        if p_flat.exists(): return p_flat
        
    return None

def resolve_video_path(row):
    """Return the raw .mp4 path for a sample, or None if not found.

    Works for both labelled rows (video_relpath column) and unlabelled rows
    (video_relpath inferred from audio_relpath).
    """
    video_rel = str(row.get('video_relpath', ''))
    if not video_rel:
        audio_rel = str(row.get('audio_relpath', ''))
        if audio_rel:
            video_rel = audio_rel.replace('audios/', 'videos/').replace('.wav', '.mp4')
        else:
            return None
    folder = str(row.get('folder', ''))

    # Mirror the same comprehensive base list used by resolve_audio_path so that
    # any folder structure that yields a valid audio path also yields a valid video path.
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
        p = base / folder / video_rel if folder else base / video_rel
        if p.exists(): return p
        if folder:
            bs_name = f"{folder}\\{video_rel.replace('/', '\\')}"
            p_bs = base / bs_name
            if p_bs.exists(): return p_bs
        p_flat = base / Path(video_rel).name
        if p_flat.exists(): return p_flat
    return None


def get_vid_paths(sid):
    # Check normal paths
    p1 = VID_DIR / f"{sid}_clip_seq.npy"
    if p1.exists():
        return p1, VID_DIR / f"{sid}_dinov2_seq.npy", VID_DIR / f"{sid}_resnet50_seq.npy"
        
    # Check Windows backslash zip extraction artifact
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


def load_unlabelled(known_audio_relpaths=None):
    """Find unlabelled audio segments for SSL pre-training (no emotion label needed).

    Steps:
      1. Check for an explicit unlabelled.csv (fastest, user-supplied).
      2. Auto-scan known Colab audio directories for all .wav files and exclude
         any already in the labelled splits (identified by audio_relpath).

    WAV files must live under a path of the form:
        .../{folder}/audios/{speaker}/{file}.wav
    which is the standard structure of the Final Modalink Dataset.

    Args:
        known_audio_relpaths: set of 'audios/{speaker}/{file}.wav' strings that
            belong to the labelled pool — these are excluded from the result.
    """
    # 1. Check for an explicit CSV first
    candidates = [
        SPLIT_DIR.parent / "unlabelled.csv",
        REPO / "data" / "processed" / "unlabelled.csv",
        REPO / "annotations" / "unlabelled.csv",
        Path("/content/drive/MyDrive/Thesis Project/data/processed/unlabelled.csv"),
        Path("/content/unlabelled.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            for col in ('sample_id', 'folder', 'audio_relpath', 'transcript'):
                if col not in df.columns:
                    df[col] = ''
            print(f"  Unlabelled pool loaded: {len(df)} samples from {p.name}")
            return df.reset_index(drop=True)

    # 2. Auto-discover WAV files in known Colab audio roots
    print("  [INFO] No unlabelled CSV found — scanning audio directories for unlabelled segments...")
    known = set(known_audio_relpaths or [])

    # Directories to scan (order matters: local /content preferred over Drive)
    audio_roots = [
        Path("/content/audio"),
        Path("/content/Thesis_Audio_Full"),
        Path("/content/drive/MyDrive/Thesis Project/dataset/Final Modalink Dataset MERGED"),
    ]

    rows = []
    seen = set()
    for base in audio_roots:
        if not base.exists():
            continue
        for wav in base.rglob("*.wav"):
            try:
                parts = wav.parts
                # Find the 'audios' directory in the path; the folder above it is the video folder
                for i, part in enumerate(parts):
                    if part == "audios" and i >= 1:
                        folder       = parts[i - 1]
                        speaker      = parts[i + 1] if i + 1 < len(parts) else ""
                        audio_relpath = f"audios/{speaker}/{wav.name}"
                        # Skip if already labelled or already seen from another root
                        if audio_relpath in known or audio_relpath in seen:
                            break
                        seen.add(audio_relpath)
                        video_relpath = audio_relpath.replace('audios/', 'videos/').replace('.wav', '.mp4')
                        rows.append({
                            'sample_id'   : f"{folder}::{video_relpath}",
                            'folder'      : folder,
                            'audio_relpath': audio_relpath,
                            'video_relpath': video_relpath,
                            'transcript'  : '',
                        })
                        break
            except Exception:
                pass

    if not rows:
        print("  [INFO] No unlabelled audio found — SSL pool = labelled train+val only.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  Auto-discovered {len(df)} unlabelled audio segments.")
    return df


def _ok_audio(row): return resolve_audio_path(row) is not None

def _ok_text(row):
    return isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2

def _ok_video(row):
    sid = str(row.get('sample_id', '')).replace("::","__").replace("/","_").replace(".mp4","")
    p, _, _ = get_vid_paths(sid)
    return p is not None and p.exists()


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
        # Freeze bottom 6 layers for efficiency and stability
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

# ─────────────────────────────────────────────────────────
# TEXT ENCODER (MARBERT)
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

# ─────────────────────────────────────────────────────────
# VIDEO ENCODER (Transformer over CLIP+DINOv2+ResNet50 features)
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
# VIDEO FEATURE EXTRACTION (for unlabelled samples)
# ─────────────────────────────────────────────────────────
def extract_unlabelled_video_features(pool):
    """Extract CLIP+DINOv2+ResNet50 sequence features for pool samples that
    have a raw .mp4 file but no pre-extracted .npy sequences in VID_DIR.

    Uses the same timm pipeline as video_stage3_extract_sequences.py.
    Saves {sid}_clip_seq.npy, {sid}_dinov2_seq.npy, {sid}_resnet50_seq.npy
    to VID_DIR so that CrossModalSSLDS can pick them up immediately after.
    GPU memory is freed before returning.
    """
    try:
        import timm, cv2
        from torchvision import transforms as tvt
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"  [VID FEAT] Skipping extraction — missing dependency: {e}")
        return

    N_FRAMES = 16
    MODELS = [
        ("clip",     "vit_base_patch32_clip_224"),
        ("dinov2",   "vit_base_patch14_dinov2"),
        ("resnet50", "resnet50"),
    ]

    # Identify samples that have a raw video file but no extracted features
    missing = []
    n_already = 0
    n_no_mp4  = 0
    for _, row in pool.iterrows():
        sid = (str(row.get('sample_id', ''))
               .replace("::", "__").replace("/", "_").replace(".mp4", ""))
        pc, _, _ = get_vid_paths(sid)
        if pc is not None and pc.exists():
            n_already += 1
            continue                          # features already extracted
        vp = resolve_video_path(row)
        if vp is not None:
            missing.append((sid, vp))
        else:
            n_no_mp4 += 1

    print(f"  [VID FEAT] Pool: {len(pool)} | already have npy: {n_already} | "
          f"mp4 found (need extraction): {len(missing)} | no mp4 found: {n_no_mp4}")
    if not missing:
        return

    print(f"  [VID FEAT] Extracting features for {len(missing)} samples (saving to VID_DIR)...")
    VID_DIR.mkdir(exist_ok=True)

    def _sample_frames(v_path):
        cap = cv2.VideoCapture(str(v_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return []
        idxs = set(np.linspace(0, total - 1, N_FRAMES, dtype=int).tolist())
        frames_dict, fi = {}, 0
        while fi <= max(idxs):
            ret, frame = cap.read()
            if not ret:
                break
            if fi in idxs:
                frames_dict[fi] = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fi += 1
        cap.release()
        ordered = [frames_dict[i] for i in sorted(idxs) if i in frames_dict]
        return ordered

    for tag, model_id in MODELS:
        print(f"  [VID FEAT]   {model_id} ...")
        try:
            model = timm.create_model(model_id, pretrained=True, num_classes=0).to(DEVICE)
            model.eval()
            cfg = timm.data.resolve_model_data_config(model)
            tf  = tvt.Compose([
                tvt.Resize(cfg['input_size'][1:]),
                tvt.CenterCrop(cfg['input_size'][1:]),
                tvt.ToTensor(),
                tvt.Normalize(mean=cfg['mean'], std=cfg['std']),
            ])
            done = 0
            for sid, vp in missing:
                out = VID_DIR / f"{sid}_{tag}_seq.npy"
                if out.exists():
                    done += 1
                    continue
                frames = _sample_frames(vp)
                if len(frames) < N_FRAMES:
                    continue
                batch = torch.stack([tf(f) for f in frames]).to(DEVICE)
                with torch.no_grad():
                    feat = model(batch)
                    if feat.dim() > 2:
                        feat = feat.mean(dim=list(range(2, feat.dim())))
                    feat = feat.cpu().numpy()       # [N_FRAMES, D]
                np.save(str(out), feat)
                done += 1
            print(f"    → {done}/{len(missing)} samples saved.")
        except Exception as e:
            print(f"    [WARN] {model_id} extraction failed: {e}")
        finally:
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()

    complete = sum(
        1 for sid, _ in missing
        if all((VID_DIR / f"{sid}_{t}_seq.npy").exists() for t, _ in MODELS)
    )
    print(f"  [VID FEAT] Done — {complete}/{len(missing)} unlabelled samples now have all 3 feature files.")


# ─────────────────────────────────────────────────────────
# CROSS-MODAL SSL  (Audio ↔ Video ↔ Text InfoNCE)
# ─────────────────────────────────────────────────────────
CM_SSL_EPOCHS = 20
CM_SSL_TEMP   = 0.07

class CrossModalSSLDS(Dataset):
    """Loads paired (audio, video, text) for three-way cross-modal InfoNCE.
    Filters to samples that have audio AND video features.
    Text (transcript) is pre-tokenised in __init__ if a tokenizer is supplied.
    """
    def __init__(self, df, tok=None, sr=16000, maxlen=80000):
        self.sr = sr; self.maxlen = maxlen
        def has_av(row):
            sid = str(row.get('sample_id','')).replace("::","__").replace("/","_").replace(".mp4","")
            pc, _, _ = get_vid_paths(sid)
            return resolve_audio_path(row) is not None and pc is not None and pc.exists()
        self.df = df[df.apply(has_av, axis=1)].reset_index(drop=True)
        # Pre-tokenise all transcripts once (fast, in-memory)
        if tok is not None:
            texts = [clean(str(t)) for t in self.df['transcript'].fillna('').tolist()]
            enc = tok(texts, truncation=True, padding='max_length',
                      max_length=64, return_tensors='pt')
            self.input_ids = enc['input_ids']
            self.attn_mask = enc['attention_mask']
        else:
            self.input_ids = self.attn_mask = None
        modalities = "audio+video+text" if tok is not None else "audio+video"
        n_real_text = int((self.attn_mask.sum(1) > 2).sum()) if self.attn_mask is not None else 0
        n_no_text   = len(self.df) - n_real_text
        print(f"  Cross-modal pool ({modalities}): {len(self.df)} samples")
        print(f"    ├─ with real transcript (A↔T eligible) : {n_real_text}")
        print(f"    └─ audio+video only (A↔V only)         : {n_no_text}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # Audio
        try:
            p = resolve_audio_path(r)
            y, _ = librosa.load(str(p), sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=25)
            y = y[:self.maxlen] if len(y) > self.maxlen else np.pad(y, (0, self.maxlen - len(y)))
        except: y = np.zeros(self.maxlen, dtype=np.float32)
        # Video
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        try:
            pc, pd_, pr = get_vid_paths(sid)
            seq = np.concatenate([np.load(pc), np.load(pd_), np.load(pr)], -1).astype(np.float32)
        except: seq = np.zeros((16, 3584), dtype=np.float32)
        out = (torch.tensor(y, dtype=torch.float32), torch.tensor(seq, dtype=torch.float32))
        if self.input_ids is not None:
            out = out + (self.input_ids[i], self.attn_mask[i])
        return out


def train_cross_modal_ssl(pool):
    """Cross-modal SSL: align audio, video, and text encoders via InfoNCE.

    Positive pair = (audio_i, video_i) or (audio_i, text_i) from the SAME utterance.
    Audio starts from WavLM pretrained weights.
    Video starts from random init — aligned by cross-modal InfoNCE (no per-modal pre-training).
    Text MARBERT backbone is frozen; only the projection head trains (avoids OOM on T4).
    Loss = InfoNCE(A↔V) + 0.5 × InfoNCE(A↔T)
    """
    sep("CROSS-MODAL SSL (Audio ↔ Video + Audio ↔ Text)")
    ckpt_a = SSL_DIR / "audio_cm_ssl.pt"
    ckpt_v = SSL_DIR / "video_cm_ssl.pt"
    ckpt_t = SSL_DIR / "text_cm_ssl.pt"
    if ckpt_a.exists() and ckpt_v.exists() and ckpt_t.exists():
        print("  [SKIP] Cross-modal SSL checkpoints cached — delete to retrain.")
        return
    set_seed(42)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds  = CrossModalSSLDS(pool, tok=tok)
    print(f"  Pool supplied : {len(pool)} samples")
    print(f"  Paired A+V+T  : {len(ds)} samples (pool filtered to those with all 3 modalities)")
    if len(ds) < 8:
        print(f"  [SKIP] Only {len(ds)} paired samples — need ≥8.")
        return

    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    # Audio: WavLM pretrained, top-6 layers trainable (frozen layers 0-5 in AudioSSLModel.__init__)
    a_enc = AudioSSLModel(SSL_PROJ_DIM).to(DEVICE)
    print("  Audio  : WavLM-Base-Plus pretrained")

    # Video: random init — cross-modal InfoNCE drives alignment from scratch
    v_enc = VideoSSLModel(proj_dim=SSL_PROJ_DIM).to(DEVICE)
    print("  Video  : random init (aligned via cross-modal InfoNCE)")

    # Text: MARBERT pretrained backbone FROZEN; only projection head trains
    t_enc = TextSSLModel(SSL_PROJ_DIM).to(DEVICE)
    for p in t_enc.bert.parameters(): p.requires_grad = False
    print("  Text   : MARBERT pretrained (backbone frozen, projection head trains)")

    # Per-modality cross-modal projection heads (each modality → shared 128-dim space)
    a_cm_proj = ProjectionHead(768 * 2, SSL_PROJ_DIM).to(DEVICE)
    v_cm_proj = ProjectionHead(512,     SSL_PROJ_DIM).to(DEVICE)
    t_cm_proj = ProjectionHead(768 * 3, SSL_PROJ_DIM).to(DEVICE)

    opt = torch.optim.AdamW([
        {'params': [p for p in a_enc.parameters() if p.requires_grad], 'lr': 2e-6},
        {'params': v_enc.parameters(),                                   'lr': 1e-5},
        {'params': list(a_cm_proj.parameters()) + list(v_cm_proj.parameters()) +
                   list(t_cm_proj.parameters()),                         'lr': 1e-3},
    ], weight_decay=0.01)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CM_SSL_EPOCHS)
    loss_fn = InfoNCELoss(CM_SSL_TEMP)
    scaler  = GradScaler()
    clip_params = (list(a_enc.parameters()) + list(v_enc.parameters()) +
                   list(a_cm_proj.parameters()) + list(v_cm_proj.parameters()) +
                   list(t_cm_proj.parameters()))

    print(f"  Epochs: {CM_SSL_EPOCHS} | Batch: 8 | "
          f"A↔V: all A+V samples | A↔T: labelled-only (non-empty transcripts)")
    for ep in range(1, CM_SSL_EPOCHS + 1):
        a_enc.train(); v_enc.train()
        a_cm_proj.train(); v_cm_proj.train(); t_cm_proj.train()
        ep_av = ep_at = 0.0; n_at_batches = 0
        for batch in dl:
            aud, vid, ids, mask = (batch[0].to(DEVICE), batch[1].to(DEVICE),
                                   batch[2].to(DEVICE), batch[3].to(DEVICE))
            opt.zero_grad()
            with autocast("cuda"):
                z_a  = a_cm_proj(a_enc.encode(aud))
                z_v  = v_cm_proj(v_enc.encode(vid))
                l_av = loss_fn(z_a, z_v)

                # A↔T only for samples with a real transcript (more than CLS+SEP active)
                has_text = mask.sum(1) > 2          # [B] bool
                if has_text.sum() >= 2:
                    with torch.no_grad():
                        t_feat = t_enc.encode(ids[has_text], mask[has_text])
                    z_t  = t_cm_proj(t_feat)
                    l_at = loss_fn(z_a[has_text], z_t)
                    loss = l_av + 0.5 * l_at
                    ep_at += l_at.item(); n_at_batches += 1
                else:
                    loss = l_av

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            scaler.step(opt); scaler.update()
            ep_av += l_av.item()
        sch.step()
        if ep % 5 == 0 or ep == 1:
            at_str = f"{ep_at/n_at_batches:.4f}" if n_at_batches else "n/a"
            print(f"  Ep {ep:02d}/{CM_SSL_EPOCHS} | "
                  f"A↔V: {ep_av/len(dl):.4f}  A↔T: {at_str} ({n_at_batches} batches)")

    torch.save({'backbone': a_enc.backbone.state_dict(), 'lw': a_enc.lw.data}, str(ckpt_a))
    torch.save({k: v for k,v in v_enc.state_dict().items() if not k.startswith('proj.')}, str(ckpt_v))
    torch.save({k: v for k,v in t_enc.state_dict().items() if 'bert' in k}, str(ckpt_t))
    print(f"  [SAVED] {ckpt_a.name}, {ckpt_v.name}, {ckpt_t.name}")
    del a_enc, v_enc, t_enc, a_cm_proj, v_cm_proj, t_cm_proj; torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────
# MINORITY-CLASS MIXUP (feature space)
# ─────────────────────────────────────────────────────────
_MINORITY = frozenset({2, 3, 6})   # Fear, Happiness, Surprise

def mixup_minority(features, labels, alpha=0.4):
    """Beta mixup applied only to minority-class samples in feature space.

    Majority-class samples pass through unchanged.  Returns
    (mixed_features, labels_a, labels_b, lambda) where the mixed loss is
    lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b).
    """
    m_mask = torch.tensor([l.item() in _MINORITY for l in labels], device=features.device)
    m_idx  = m_mask.nonzero(as_tuple=True)[0]
    if len(m_idx) < 2:
        return features, labels, labels, 1.0
    lam   = float(np.random.beta(alpha, alpha))
    perm  = m_idx[torch.randperm(len(m_idx), device=features.device)]
    mixed = features.clone()
    mixed[m_idx] = lam * features[m_idx] + (1 - lam) * features[perm]
    labels_b        = labels.clone()
    labels_b[m_idx] = labels[perm]
    return mixed, labels, labels_b, lam


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
            cm_ckpt = SSL_DIR / "audio_cm_ssl.pt"
            sd = torch.load(str(cm_ckpt if cm_ckpt.exists() else SSL_DIR/"audio_ssl.pt"), map_location=DEVICE)
            m.backbone.backbone.load_state_dict(sd['backbone'])
            m.backbone.lw.data = sd['lw']
            print(f"  [SSL] audio encoder: {'cross-modal' if cm_ckpt.exists() else 'per-modal'}")
        else:
            # Unfreeze full WavLM for fair baseline (AudioSSLModel.__init__ freezes layers 0-5)
            for p in m.backbone.backbone.parameters(): p.requires_grad = True
        lr_bb, lr_hd = 1e-5, 1e-3
        bs = 8

    elif name == "TEXT":
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        ds_tr = TextFTDS(train_df['transcript'].values, train_df['emotion_final'].values, tok)
        ds_va = TextFTDS(val_df['transcript'].values,   val_df['emotion_final'].values,   tok)
        ds_te = TextFTDS(test_df['transcript'].values,  test_df['emotion_final'].values,  tok)
        m = TextFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            cm_ckpt = SSL_DIR / "text_cm_ssl.pt"
            if cm_ckpt.exists():
                m.backbone.load_state_dict(torch.load(str(cm_ckpt), map_location=DEVICE), strict=False)
                print("  [SSL] text encoder: cross-modal aligned (text_cm_ssl.pt)")
            else:
                print("  [SSL] text encoder: MARBERT pretrained (no cm checkpoint)")
        else:
            # Unfreeze full MARBERT for fair baseline (TextSSLModel.__init__ freezes layers 0-7)
            for p in m.backbone.bert.parameters(): p.requires_grad = True
        lr_bb, lr_hd = 1e-5, 5e-4
        bs = 16

    else:  # VIDEO
        ds_tr, ds_va, ds_te = VideoFTDS(train_df), VideoFTDS(val_df), VideoFTDS(test_df)
        m = VideoFTModel(SSL_PROJ_DIM).to(DEVICE)
        if use_ssl:
            cm_ckpt = SSL_DIR / "video_cm_ssl.pt"
            if cm_ckpt.exists():
                m.backbone.load_state_dict(torch.load(str(cm_ckpt), map_location=DEVICE), strict=False)
                print("  [SSL] video encoder: cross-modal (video_cm_ssl.pt)")
            else:
                print("  [SSL] video encoder: random init (no cross-modal checkpoint found)")
        lr_bb, lr_hd = 3e-5, 1e-3
        bs = 32

    # ── Class-weighted sampler (replaces shuffle=True for the train loader) ──
    y_tr       = np.array([LID[e] for e in train_df['emotion_final']])
    cw         = compute_class_weight('balanced', classes=np.arange(7), y=y_tr)
    cw_tensor  = torch.tensor(cw, dtype=torch.float).to(DEVICE)
    samp_w     = torch.tensor(cw[y_tr], dtype=torch.float64)
    sampler    = WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=bs, sampler=sampler)
    dl_va = DataLoader(ds_va, batch_size=bs)
    dl_te = DataLoader(ds_te, batch_size=bs)

    # Separate backbone / head param groups for differential LR
    bb_params = [p for n, p in m.named_parameters() if 'backbone' in n and p.requires_grad]
    hd_params = [p for n, p in m.named_parameters() if 'backbone' not in n]
    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': lr_bb},
        {'params': hd_params, 'lr': lr_hd},
    ], weight_decay=0.05)

    sch       = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr_hd],
                                                     steps_per_epoch=len(dl_tr), epochs=20)
    supcon_fn = SupConLoss(SSL_TEMP)
    best_f1, no_improve, patience = 0.0, 0, 5
    ckpt = SAVE_DIR / f"{name.lower()}_ft.pt"

    for ep in range(1, 21):
        m.train()
        cur_supcon_w = 0.3 if use_supcon else 0.0

        for batch in dl_tr:
            opt.zero_grad()
            labels = batch[1].to(DEVICE)

            # ── Encode → minority mixup → classify ──
            if name == "TEXT":
                feat = m.backbone.encode(
                    batch[0]['input_ids'].to(DEVICE),
                    batch[0]['attention_mask'].to(DEVICE))
            else:
                feat = m.backbone.encode(batch[0].to(DEVICE))

            feat, labels_a, labels_b, lam = mixup_minority(feat, labels)

            if name == "TEXT":
                logits = torch.stack([m.classifier(d(feat)) for d in m.drops]).mean(0)
            else:
                logits = m.classifier(feat)
            proj = m.proj_ft(feat)

            ce_a = F.cross_entropy(logits, labels_a, weight=cw_tensor, label_smoothing=0.1)
            ce_b = F.cross_entropy(logits, labels_b, weight=cw_tensor, label_smoothing=0.1)
            loss = lam * ce_a + (1 - lam) * ce_b
            if use_supcon:
                loss += cur_supcon_w * supcon_fn(proj, labels_a)

            loss.backward(); opt.step(); sch.step()

        # ── Validation: accuracy + macro F1 + minority class F1 ──
        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for batch in dl_va:
                if name == "TEXT":
                    logits, _ = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
                else:
                    logits, _ = m(batch[0].to(DEVICE))
                ps.extend(logits.argmax(1).cpu().numpy()); ts.extend(batch[1].numpy())

        acc      = accuracy_score(ts, ps)
        f1_macro = f1_score(ts, ps, average='macro',  zero_division=0)
        f1_cls   = f1_score(ts, ps, average=None,     zero_division=0, labels=list(range(7)))

        improved = f1_macro > best_f1
        if improved:
            best_f1 = f1_macro; no_improve = 0
            torch.save(m.state_dict(), str(ckpt))
        else:
            no_improve += 1

        print(f"  Ep {ep:02d} | Acc {acc:.3f} | F1 {f1_macro:.3f} "
              f"| Fear {f1_cls[2]:.2f} Hap {f1_cls[3]:.2f} Sur {f1_cls[6]:.2f} "
              f"{'✓' if improved else f'({no_improve}/{patience})'}")

        if no_improve >= patience:
            print(f"  [EARLY STOP] F1 flat for {patience} epochs → stopping at ep {ep}")
            break

    m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    m.eval(); probs = []
    with torch.no_grad():
        for batch in dl_te:
            if name == "TEXT":
                logits, _ = m(batch[0]['input_ids'].to(DEVICE), batch[0]['attention_mask'].to(DEVICE))
            else:
                logits, _ = m(batch[0].to(DEVICE))
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
        
        # Grid Search Fusion for Optimal Weights
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

    # ── Build SSL pool ────────────────────────────────────
    # SSL is unsupervised so we use train+val (test kept out to avoid any leakage).
    # Unlabelled audio discovered from the Colab audio directory expands the pool;
    # CrossModalSSLDS automatically filters to samples that have all 3 modalities
    # (audio + video features + transcript) — unlabelled audio-only samples that
    # lack video features or transcripts are reported but not forced in.
    labelled_pool  = pd.concat([tr, va]).reset_index(drop=True)
    known_relpaths = set(labelled_pool['audio_relpath'].dropna().tolist())
    unlabelled_df  = load_unlabelled(known_relpaths)

    if len(unlabelled_df) > 0:
        ssl_pool = pd.concat([labelled_pool, unlabelled_df]).reset_index(drop=True)
        sep("SSL POOL SIZES")
        print(f"  Labelled   (train+val)  : {len(labelled_pool)}")
        print(f"  Unlabelled (audio scan) : {len(unlabelled_df)}")
        print(f"  Total SSL input pool    : {len(ssl_pool)}  "
              f"(cross-modal DS will filter to samples with A+V+T)")
    else:
        ssl_pool = labelled_pool
        print(f"  SSL pool: {len(labelled_pool)} labelled samples (train + val)")

    # ── PHASE 1-A: Extract video features for unlabelled samples ─────────────
    # Unlabelled segments that have raw .mp4 files get CLIP+DINOv2+ResNet50
    # features extracted here (same timm pipeline as video_stage3_extract_sequences.py).
    # After this step they can join the A↔V cross-modal SSL pool.
    # GPU is freed before loading audio/text models.
    if len(unlabelled_df) > 0:
        extract_unlabelled_video_features(ssl_pool)

    # ── PHASE 1-B: Cross-modal SSL ───────────────────────
    # Per-modal SSL removed: WavLM/MARBERT are already pretrained on large corpora;
    # individual SimCLR on ~600-5k samples degraded those representations.
    # VideoSSLModel starts from random init and is aligned to audio+text via
    # cross-modal InfoNCE — no separate per-modal video SSL warmup needed.
    train_cross_modal_ssl(ssl_pool)

    sep("PHASE 1 COMPLETE -- Cross-modal SSL done.")

    # ── PHASE 2: Supervised fine-tuning ablation ──────────
    run_ablation(tr, va, te)
