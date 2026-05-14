"""
Late Fusion — Track A Multimodal Emotion Recognition (Colab Version)
=====================================================================
Run on: Google Colab (T4 GPU recommended)
Drive layout expected:
  /content/drive/MyDrive/Thesis Project/
    data/processed/splits/multimodal_eligible/  (train.csv, val.csv, test.csv)
    data/processed/features/video_sequences_v1/ (*.npy files)
    data/raw/  (audio files, referenced by audio_relpath in CSVs)
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
# PATHS (Colab)
# ─────────────────────────────────────────────────────────
DRIVE       = Path("/content/drive/MyDrive/Thesis Project")
REPO        = Path("/content/thesis")
SPLIT_DIR   = REPO / "data/processed/splits/multimodal_eligible"
VID_DIR     = Path("/content/video_features/video_sequences_v1")   # extracted locally
AUDIO_BASE  = Path("/content/audio/Thesis Project/dataset/Final Modalink Dataset MERGED")
SAVE_DIR    = Path("/content/fusion_models")
SAVE_DIR.mkdir(exist_ok=True)

LID     = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6}
CLASSES = list(LID.keys())
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

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
    """Resolve audio path using folder + audio_relpath under AUDIO_BASE."""
    folder      = str(row.get('folder', ''))
    audio_rel   = str(row.get('audio_relpath', ''))

    # Strategy 1: AUDIO_BASE / folder / audio_relpath  (from Thesis_Audio_Full.zip)
    if folder and audio_rel:
        p = AUDIO_BASE / folder / audio_rel
        if p.exists(): return p

    # Strategy 2: AUDIO_BASE / audio_relpath  (flat structure in zip)
    if audio_rel:
        p = AUDIO_BASE / audio_rel
        if p.exists(): return p

    # Strategy 3: Drive fallbacks
    for base in [DRIVE/"dataset/Final Modalink Dataset MERGED", DRIVE/"data/raw", DRIVE]:
        if folder:
            p = base / folder / audio_rel
            if p.exists(): return p
        p = base / audio_rel
        if p.exists(): return p

    return None


def load_splits():
    tr_raw = pd.read_csv(SPLIT_DIR / "train.csv")
    va_raw = pd.read_csv(SPLIT_DIR / "val.csv")
    te_raw = pd.read_csv(SPLIT_DIR / "test.csv")

    def sid(row): return row['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
    def has_video(row): return (VID_DIR / f"{sid(row)}_clip_seq.npy").exists()
    def has_audio(row): return resolve_audio_path(row) is not None
    def has_text(row):  return isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2
    def all_ok(row):    return has_video(row) and has_audio(row) and has_text(row)

    sep("🔍 DIAGNOSTIC — Path Resolution Check")
    row0 = tr_raw.iloc[0]
    vpath = VID_DIR / f"{sid(row0)}_clip_seq.npy"
    print(f"  [VIDEO] {vpath} → exists: {vpath.exists()}")
    print(f"  [AUDIO] resolved: {resolve_audio_path(row0) is not None}")
    print(f"  [TEXT]  present: {has_text(row0)}")

    tr = tr_raw[tr_raw.apply(all_ok, axis=1)].reset_index(drop=True)
    va = va_raw[va_raw.apply(all_ok, axis=1)].reset_index(drop=True)
    te = te_raw[te_raw.apply(all_ok, axis=1)].reset_index(drop=True)

    sep("ALIGNED SPLITS (all 3 modalities)")
    print(f"  Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
    return tr, va, te, tr_raw, va_raw   # tr_raw/va_raw used by text for full pool




# ─────────────────────────────────────────────────────────
# MODALITY 1 — VIDEO  (Multi-Scale Window Transformer)
# ─────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(nn.Linear(c, c//16), nn.ReLU(), nn.Linear(c//16, c), nn.Sigmoid())
    def forward(self, x):
        b,n,c = x.shape
        return x * self.fc(self.pool(x.transpose(1,2)).view(b,c)).view(b,1,c)

class MSWModel(nn.Module):
    def __init__(self, d=512, drop=0.5):
        super().__init__()
        self.proj    = nn.Sequential(nn.Linear(3584,d), nn.LayerNorm(d), nn.GELU())
        self.se      = SEBlock(d)
        self.pos     = nn.Parameter(torch.randn(1,16,d))
        enc          = nn.TransformerEncoderLayer(d, 8, d*2, drop, batch_first=True)
        self.tfm     = nn.TransformerEncoder(enc, 2)
        self.attn    = nn.Linear(d,1)
        self.fuse    = nn.Linear(d*3, d)
        self.cls     = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d,256), nn.GELU(), nn.Dropout(drop), nn.Linear(256,7))
        self.scl     = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,128))
    def _pool(self, x):
        return (x * F.softmax(self.attn(x),1)).sum(1)
    def forward(self, x):
        if self.training: x = x + torch.randn_like(x)*0.01
        x = self.tfm(self.se(self.proj(x)) + self.pos)
        f = self.fuse(torch.cat([self._pool(x), self._pool(x[:,4:12,:]), self._pool(x[:,6:10,:])], -1))
        return self.cls(f), F.normalize(self.scl(f), 1)

class VideoDS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r   = self.df.iloc[i]
        sid = r['sample_id'].replace("::","__").replace("/","_").replace(".mp4","")
        c,d,r2 = [np.load(VID_DIR/f"{sid}_{m}_seq.npy") for m in ['clip','dinov2','resnet50']]
        return torch.tensor(np.concatenate([c,d,r2],-1), dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

def scl_loss(f, y, T=0.07):
    m = torch.zeros(7, device=f.device)
    m[LID['Happiness']],m[LID['Surprise']],m[LID['Fear']] = 0.2,0.15,0.25
    sim  = f@f.T/T
    mask = y.unsqueeze(1).eq(y.unsqueeze(0)).float()
    sim  -= mask * m[y].unsqueeze(1)
    lm   = torch.ones_like(mask).scatter_(1, torch.arange(len(y),device=f.device).view(-1,1), 0)
    return -(mask*lm*(sim - torch.log(sim.exp().sum(1,keepdim=True)+1e-6))).sum(1).mean() / (mask.sum(1).mean()+1e-6)

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

def train_video(tr, va, te):
    sep("🎥  VIDEO MODALITY — MSW Transformer (5-seed ensemble)")
    tl = DataLoader(VideoDS(tr), batch_size=32, shuffle=True)
    vl = DataLoader(VideoDS(va), batch_size=32)
    el = DataLoader(VideoDS(te), batch_size=32)
    all_probs, all_w = [], []
    for seed in [42, 1337, 2024, 777, 999]:
        print(f"\n  ── Seed {seed} ──")
        set_seed(seed)
        m   = MSWModel().to(DEVICE)
        opt = Lookahead(torch.optim.AdamW(m.parameters(), lr=7e-5, weight_decay=5e-2))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt.opt, max_lr=8.4e-5, steps_per_epoch=len(tl), epochs=25)
        best_f1, ckpt = 0, SAVE_DIR/f"vid_{seed}.pt"
        for ep in range(1, 26):
            m.train()
            for x,y in tl:
                x,y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                lo,pr = m(x)
                (F.cross_entropy(lo,y,label_smoothing=0.1) + 0.4*scl_loss(pr,y)).backward()
                opt.step(); sch.step()
            m.eval(); ps,ts = [],[]
            with torch.no_grad():
                for x,y in vl: lo,_ = m(x.to(DEVICE)); ps.extend(lo.argmax(1).cpu().numpy()); ts.extend(y.numpy())
            acc = accuracy_score(ts, ps)
            f1  = f1_score(ts, ps, average='macro', zero_division=0)
            marker = " ⭐" if f1 > best_f1 else ""
            print(f"    Ep {ep:02d} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}{marker}")
            if f1 > best_f1: best_f1=f1; torch.save(m.state_dict(), str(ckpt))
        m.load_state_dict(torch.load(str(ckpt), weights_only=True)); m.eval()
        tp = []
        with torch.no_grad():
            for x,_ in el: lo,_ = m(x.to(DEVICE)); tp.append(F.softmax(lo,1).cpu().numpy())
        all_probs.append(np.vstack(tp)); all_w.append(best_f1)
    w = np.array(all_w); w /= w.sum()
    probs = sum(p*wt for p,wt in zip(all_probs,w))
    t_labels = [LID[e] for e in te['emotion_final'].values]
    report("VIDEO ENSEMBLE", t_labels, probs.argmax(1))
    return probs


# ─────────────────────────────────────────────────────────
# MODALITY 2 — AUDIO  (WavLM-Base-Plus, Layer-Weighted)
# ─────────────────────────────────────────────────────────
class WavLMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
        self.lw   = nn.Parameter(torch.ones(13))
        self.head = nn.Sequential(nn.Linear(768*2,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,7))
    def forward(self, x):
        hs  = torch.stack(self.backbone(x).hidden_states, 0)
        out = (hs * F.softmax(self.lw,0).view(-1,1,1,1)).sum(0)
        return self.head(torch.cat([out.mean(1), out.std(1)], 1))

class AudioDS(Dataset):
    def __init__(self, df, sr=16000, maxlen=80000):
        self.df=df.reset_index(drop=True); self.sr=sr; self.maxlen=maxlen
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            apath = resolve_audio_path(r)
            if apath is None: raise FileNotFoundError
            y,_ = librosa.load(str(apath), sr=self.sr)
            y,_ = librosa.effects.trim(y, top_db=25)
            y   = y[:self.maxlen] if len(y)>self.maxlen else np.pad(y,(0,self.maxlen-len(y)))
        except: y = np.zeros(self.maxlen)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(LID[r['emotion_final']], dtype=torch.long)

def train_audio(tr, va, te):
    sep("🎙️  AUDIO MODALITY — WavLM-Base-Plus (Layer-Weighted)")
    set_seed(42)
    tl = DataLoader(AudioDS(tr), batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(AudioDS(va), batch_size=8, num_workers=2)
    el = DataLoader(AudioDS(te), batch_size=8, num_workers=2)
    m  = WavLMClassifier().to(DEVICE)
    # Warmup: freeze backbone
    for p in m.backbone.parameters(): p.requires_grad=False
    opt    = torch.optim.AdamW([{'params':m.head.parameters(),'lr':1e-3},{'params':m.lw,'lr':1e-3}])
    scaler = GradScaler()
    best_f1, ckpt = 0, SAVE_DIR/"aud_best.pt"
    for ep in range(1, 16):
        if ep == 3:   # Unfreeze backbone
            for p in m.backbone.parameters(): p.requires_grad=True
            opt = torch.optim.AdamW([
                {'params':m.backbone.parameters(),'lr':4e-5},
                {'params':m.head.parameters(),'lr':1e-3},
                {'params':m.lw,'lr':1e-3}
            ])
            print("  [Ep 3] Backbone unfrozen — full fine-tuning begins")
        m.train(); tl_s=0
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            with autocast("cuda"):
                loss = F.cross_entropy(m(x), y, label_smoothing=0.1)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl_s += loss.item()
        m.eval(); ps,ts=[],[]
        with torch.no_grad():
            for x,y in vl: ps.extend(m(x.to(DEVICE)).argmax(1).cpu().numpy()); ts.extend(y.numpy())
        acc = accuracy_score(ts,ps); f1 = f1_score(ts,ps,average='macro',zero_division=0)
        marker = " ⭐" if f1 > best_f1 else ""
        print(f"  Ep {ep:02d} | Loss: {tl_s/len(tl):.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}{marker}")
        if f1>best_f1: best_f1=f1; torch.save(m.state_dict(), str(ckpt))
    m.load_state_dict(torch.load(str(ckpt), weights_only=True)); m.eval()
    tp=[]
    with torch.no_grad():
        for x,_ in el: tp.append(F.softmax(m(x.to(DEVICE)),1).cpu().numpy())
    probs = np.vstack(tp)
    t_labels = [LID[e] for e in te['emotion_final'].values]
    report("AUDIO", t_labels, probs.argmax(1))
    return probs

# ─────────────────────────────────────────────────────────
# MODALITY 3 — TEXT  (MARBERT, 5-Fold CV, Full Text Pool)
# ─────────────────────────────────────────────────────────
MODEL_NAME = "UBC-NLP/MARBERT"

EGYPTIAN_FILLERS = re.compile(
    r'\b(اه|ايه|يعني|بص|كده|كدا|اهو|والله|عشان|بقا|بقى|يا|اوه|هاه|اوكي|اوكى|تمام|صح|ايوه|لا|اه|مش|ميش)\b'
)

def clean(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)   # diacritics
    t = re.sub(r'[\u0622\u0623\u0625]', '\u0627', t)  # Alef variants → bare Alef
    t = re.sub(r'\u0629', '\u0647', t)              # Ta marbuta → Ha
    t = re.sub(r'\u0649', '\u064A', t)              # Alef maqsura → Ya
    t = re.sub(r'\u0640', '', t)                    # Tatweel
    t = EGYPTIAN_FILLERS.sub(' ', t)                # dialect fillers
    t = re.sub(r'(.)\1+', r'\1\1', t)              # repeated chars
    return re.sub(r'\s+', ' ', t).strip()

class MARBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        bert     = AutoModel.from_pretrained(MODEL_NAME)
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1, bias="none"
        )
        self.bert = get_peft_model(bert, lora_cfg)
        self.cls  = nn.Linear(768*3, 7)
        self.drops= nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
    def forward(self, ids, mask):
        lh  = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        msk = mask.unsqueeze(-1).expand(lh.size()).float()
        mp  = (lh*msk).sum(1) / msk.sum(1).clamp(min=1e-9)
        xp  = (lh*msk - (1-msk)*1e9).max(1)[0]
        cat = torch.cat([lh[:,0,:], mp, xp], 1)
        return torch.stack([self.cls(d(cat)) for d in self.drops]).mean(0)

class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc    = tok([clean(str(t)) for t in texts],
                          truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {k:v[i] for k,v in self.enc.items()}, torch.tensor(self.labels[i], dtype=torch.long)

def train_text(tr, va, te, tr_raw, va_raw):
    from sklearn.model_selection import StratifiedKFold
    sep("📝  TEXT — MARBERT+LoRA | v30 Champion Config | 5-Fold CV")
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load training pool from final_sanitized (SAME data v30 used for its 64% result)
    # Fall back to multimodal_eligible raw pool if Drive path not available
    FS_DIR = DRIVE / "data/processed/splits/final_sanitized"
    if (FS_DIR / "train.csv").exists():
        fs_tr = pd.read_csv(FS_DIR / "train.csv")
        fs_va = pd.read_csv(FS_DIR / "val.csv")
        pool  = pd.concat([fs_tr, fs_va]).reset_index(drop=True)
        print(f"  Using final_sanitized pool: {len(pool)} samples")
    else:
        def has_text(row): return isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 2
        pool = pd.concat([tr_raw[tr_raw.apply(has_text, axis=1)],
                          va_raw[va_raw.apply(has_text, axis=1)]]).reset_index(drop=True)
        print(f"  final_sanitized not found on Drive — using multimodal pool: {len(pool)} samples")

    texts     = pool['transcript'].values
    labels    = np.array([LID[e] for e in pool['emotion_final'].values])
    te_labels = np.array([LID[e] for e in te['emotion_final'].values])
    te_loader = DataLoader(TextDS(te['transcript'].values, te_labels, tok), batch_size=16)
    print(f"  Aligned test: {len(te)} samples")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_probs = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n  ── Fold {fold+1}/5 ──")
        tl = DataLoader(TextDS(texts[t_idx], labels[t_idx], tok), batch_size=16, shuffle=True)
        vl = DataLoader(TextDS(texts[v_idx], labels[v_idx], tok), batch_size=16)

        m = MARBERTClassifier().to(DEVICE)
        # v30 exact optimizer — head first (higher LR), then bert
        opt = torch.optim.AdamW([
            {'params': [p for n,p in m.named_parameters() if 'classifier' in n or ('bert' not in n)], 'lr': 8e-4},
            {'params': [p for n,p in m.named_parameters() if 'bert' in n], 'lr': 2e-5}
        ], weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)   # v30 exact

        # v30 exact: monitor ACCURACY, patience=8
        best_acc, best_f1, ckpt, patience, pat = 0, 0, SAVE_DIR/f"txt_fold{fold}.pt", 8, 0
        for ep in range(1, 40):
            m.train()
            for bd, bl in tl:
                opt.zero_grad()
                criterion(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)),
                          bl.to(DEVICE)).backward()
                opt.step()
            m.eval(); ps, ts = [], []
            with torch.no_grad():
                for bd, bl in vl:
                    ps.extend(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)).argmax(1).cpu().numpy())
                    ts.extend(bl.numpy())
            acc = accuracy_score(ts, ps)
            f1  = f1_score(ts, ps, average='macro', zero_division=0)
            marker = " ⭐" if acc > best_acc else ""
            print(f"    Ep {ep:02d} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}{marker}")
            if acc > best_acc:
                best_acc, best_f1 = acc, f1
                torch.save(m.state_dict(), str(ckpt)); pat = 0
            else:
                pat += 1
                if pat >= patience: print(f"    Early stop. Best Acc: {best_acc:.4f}"); break

        m.load_state_dict(torch.load(str(ckpt), weights_only=True)); m.eval()
        fp = []
        with torch.no_grad():
            for bd, _ in te_loader:
                fp.append(F.softmax(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)), 1).cpu().numpy())
        fold_probs.append(np.vstack(fp))
        print(f"    Rolling Test Acc: {accuracy_score(te_labels, np.mean(fold_probs,0).argmax(1)):.4f}")

    probs = np.mean(fold_probs, 0)
    report("TEXT (v30 Champion Config)", te_labels, probs.argmax(1))
    return probs



# ─────────────────────────────────────────────────────────
# LATE FUSION
# ─────────────────────────────────────────────────────────
def late_fusion(vp, ap, tp, te):
    sep("🔀  LATE FUSION — Weighted Average Grid Search")
    t_labels = np.array([LID[e] for e in te['emotion_final'].values])

    # Per-modality standalone test accuracy summary
    print(f"  Video  standalone → Acc: {accuracy_score(t_labels,vp.argmax(1)):.4f} | F1: {f1_score(t_labels,vp.argmax(1),average='macro',zero_division=0):.4f}")
    print(f"  Audio  standalone → Acc: {accuracy_score(t_labels,ap.argmax(1)):.4f} | F1: {f1_score(t_labels,ap.argmax(1),average='macro',zero_division=0):.4f}")
    print(f"  Text   standalone → Acc: {accuracy_score(t_labels,tp.argmax(1)):.4f} | F1: {f1_score(t_labels,tp.argmax(1),average='macro',zero_division=0):.4f}")

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
    final_preds = final_probs.argmax(1)
    print(f"\n  Best weights → Video: {wv:.2f} | Audio: {wa:.2f} | Text: {wt:.2f}")
    np.save(str(SAVE_DIR/"fusion_probs.npy"), final_probs)
    report("🏆 LATE FUSION FINAL", t_labels, final_preds)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sep(f"🚀 MULTIMODAL LATE FUSION v2 — Track A | Device: {DEVICE}")
    tr, va, te, tr_raw, va_raw = load_splits()

    VP = SAVE_DIR / "vid_probs.npy"
    AP = SAVE_DIR / "aud_probs.npy"
    TP = SAVE_DIR / "txt_probs.npy"

    # TEXT FIRST — so you can stop early and check quality
    if TP.exists():
        print("\n⏩ Loading cached text probs..."); txt_probs = np.load(str(TP))
    else:
        txt_probs = train_text(tr, va, te, tr_raw, va_raw); np.save(str(TP), txt_probs)

    if VP.exists():
        print("\n⏩ Loading cached video probs..."); vid_probs = np.load(str(VP))
    else:
        vid_probs = train_video(tr, va, te); np.save(str(VP), vid_probs)

    if AP.exists():
        print("\n⏩ Loading cached audio probs..."); aud_probs = np.load(str(AP))
    else:
        aud_probs = train_audio(tr, va, te); np.save(str(AP), aud_probs)

    late_fusion(vid_probs, aud_probs, txt_probs, te)
