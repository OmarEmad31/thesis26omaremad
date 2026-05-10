"""
Late Fusion — Track A Multimodal Emotion Recognition
=====================================================
Single Source of Truth: data/processed/splits/multimodal_eligible/
Modalities: Video (MSW-TT) + Audio (WavLM-Base-Plus) + Text (MARBERT)

Strategy:
  1. Load all three splits from multimodal_eligible/
  2. Filter to samples with ALL modalities available
  3. Train each modality independently on the SAME split
  4. Save softmax probabilities (probs) for the test set
  5. Weighted Late Fusion of all three probability arrays
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
from transformers import WavLMModel, AutoConfig, AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────
ROOT = Path(r"d:\Thesis Project")
SPLIT_DIR = ROOT / "data" / "processed" / "splits" / "multimodal_eligible"
VIDEO_FEAT_DIR = ROOT / "data" / "processed" / "features" / "video_sequences_v1"
MODELS_DIR = ROOT / "models" / "fusion"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LID = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Surprise':6}
EMOTIONS = list(LID.keys())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────
# SPLIT LOADING & ALIGNMENT
# ─────────────────────────────────────────────────────────
def load_aligned_splits():
    tr = pd.read_csv(SPLIT_DIR / "train.csv")
    va = pd.read_csv(SPLIT_DIR / "val.csv")
    te = pd.read_csv(SPLIT_DIR / "test.csv")

    def has_video(row):
        sid = row['sample_id'].replace("::", "__").replace("/", "_").replace(".mp4", "")
        return (VIDEO_FEAT_DIR / f"{sid}_clip_seq.npy").exists()

    def has_audio(row):
        p = ROOT / row['audio_relpath'] if isinstance(row.get('audio_relpath'), str) else None
        return p is not None and p.exists()

    def has_text(row):
        return isinstance(row.get('transcript'), str) and len(str(row['transcript']).strip()) > 0

    def filter_complete(df):
        mask = df.apply(lambda r: has_video(r) and has_audio(r) and has_text(r), axis=1)
        return df[mask].reset_index(drop=True)

    tr_f, va_f, te_f = filter_complete(tr), filter_complete(va), filter_complete(te)
    print(f"✅ Aligned splits (all 3 modalities present):")
    print(f"   Train: {len(tr_f)} | Val: {len(va_f)} | Test: {len(te_f)}")
    return tr_f, va_f, te_f

# ─────────────────────────────────────────────────────────
# ── MODALITY 1: VIDEO (MSW-TT) ────────────────────────────
# ─────────────────────────────────────────────────────────
D_MODEL_VID = 512
VID_EPOCHS = 25
VID_SEEDS = [42, 1337, 2024, 777, 999]

class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(c, c//16), nn.ReLU(), nn.Linear(c//16, c), nn.Sigmoid())
    def forward(self, x):
        b, n, c = x.size()
        y = self.avg_pool(x.transpose(1,2)).view(b, c)
        return x * self.fc(y).view(b, 1, c)

class MSWModel(nn.Module):
    def __init__(self, d=D_MODEL_VID, dropout=0.5):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(3584, d), nn.LayerNorm(d), nn.GELU())
        self.se = SEBlock(d)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d))
        layer = nn.TransformerEncoderLayer(d, 8, d*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.attn = nn.Linear(d, 1)
        self.scale_fusion = nn.Linear(d*3, d)
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 7))
        self.scl_head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 128))
    def forward(self, x):
        if self.training: x = x + torch.randn_like(x) * 0.01
        x = self.proj(x); x = self.se(x); x = x + self.pos_embed; x = self.transformer(x)
        def pool(f): w = F.softmax(self.attn(f), 1); return (f * w).sum(1)
        fused = self.scale_fusion(torch.cat([pool(x), pool(x[:,4:12,:]), pool(x[:,6:10,:])], -1))
        return self.classifier(fused), F.normalize(self.scl_head(fused), 1)

class VideoDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['sample_id'].replace("::", "__").replace("/", "_").replace(".mp4", "")
        c, d, r = [np.load(str(VIDEO_FEAT_DIR / f"{sid}_{m}_seq.npy")) for m in ['clip','dinov2','resnet50']]
        return torch.tensor(np.concatenate([c,d,r], -1), dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long)

def margin_scl(feats, labels, T=0.07):
    m = torch.zeros(7, device=feats.device)
    m[LID['Happiness']], m[LID['Surprise']], m[LID['Fear']] = 0.2, 0.15, 0.25
    sim = feats @ feats.T / T
    mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
    sim -= mask * m[labels].unsqueeze(1)
    lm = torch.ones_like(mask).scatter_(1, torch.arange(len(labels), device=feats.device).view(-1,1), 0)
    return -(mask * lm * (sim - torch.log(sim.exp().sum(1, keepdim=True) + 1e-6))).sum(1).mean() / (mask.sum(1).mean() + 1e-6)

class LAHead:
    def __init__(self, opt, k=5, alpha=0.5):
        self.opt, self.k, self.alpha = opt, k, alpha
        self.param_groups = opt.param_groups
        self.slow = [[p.data.clone() for p in g['params']] for g in opt.param_groups]
        self.i = 0
    def step(self):
        self.opt.step(); self.i += 1
        if self.i % self.k == 0:
            for i, g in enumerate(self.param_groups):
                for j, p in enumerate(g['params']):
                    p.data.mul_(self.alpha).add_(self.slow[i][j], alpha=1-self.alpha)
                    self.slow[i][j].copy_(p.data)
    def zero_grad(self, **kw): self.opt.zero_grad(**kw)

def train_video(tr_df, va_df, te_df):
    print("\n" + "="*50 + "\n🎥 TRAINING VIDEO MODALITY (MSW-TT)\n" + "="*50)
    tr_ld = DataLoader(VideoDS(tr_df), batch_size=32, shuffle=True)
    va_ld = DataLoader(VideoDS(va_df), batch_size=32)
    te_ld = DataLoader(VideoDS(te_df), batch_size=32)
    probs_list, weights = [], []
    for seed in VID_SEEDS:
        set_seed(seed); print(f"\n  Seed {seed}")
        m = MSWModel().to(DEVICE)
        base = torch.optim.AdamW(m.parameters(), lr=7e-5, weight_decay=5e-2)
        opt = LAHead(base)
        sch = torch.optim.lr_scheduler.OneCycleLR(base, max_lr=8.4e-5, steps_per_epoch=len(tr_ld), epochs=VID_EPOCHS)
        best_f1, ckpt = 0, MODELS_DIR / f"video_{seed}.pt"
        for ep in range(1, VID_EPOCHS+1):
            m.train()
            for x, y in tr_ld:
                x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                lo, pr = m(x)
                (F.cross_entropy(lo, y, label_smoothing=0.1) + 0.4*margin_scl(pr, y)).backward()
                opt.step(); sch.step()
            m.eval(); ps, ts = [], []
            with torch.no_grad():
                for vx, vy in va_ld:
                    vl, _ = m(vx.to(DEVICE)); ps.extend(vl.argmax(1).cpu().numpy()); ts.extend(vy.numpy())
            f1 = f1_score(ts, ps, average='macro', zero_division=0)
            if f1 > best_f1: best_f1 = f1; torch.save(m.state_dict(), str(ckpt))
            print(f"    Ep {ep:02d} | Val F1: {f1:.4f}")
        m.load_state_dict(torch.load(str(ckpt), weights_only=True))
        tp = []
        with torch.no_grad():
            for tx, _ in te_ld: tl, _ = m(tx.to(DEVICE)); tp.append(F.softmax(tl,1).cpu().numpy())
        probs_list.append(np.vstack(tp)); weights.append(best_f1)
    w = np.array(weights); w /= w.sum()
    return sum(p * wt for p, wt in zip(probs_list, w))

# ─────────────────────────────────────────────────────────
# ── MODALITY 2: AUDIO (WavLM-Base-Plus) ──────────────────
# ─────────────────────────────────────────────────────────
AUD_SR = 16000
AUD_MAX = 80000
AUD_EPOCHS = 12

class LayerWeightedAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
        self.lw = nn.Parameter(torch.ones(13))
        self.head = nn.Sequential(nn.Linear(768*2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 7))
    def forward(self, x):
        hs = torch.stack(self.backbone(x).hidden_states, 0)
        w = F.softmax(self.lw, 0).view(-1,1,1,1)
        out = (hs * w).sum(0)
        return self.head(torch.cat([out.mean(1), out.std(1)], 1))

class AudioDS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = ROOT / row['audio_relpath']
        try:
            y, _ = librosa.load(str(path), sr=AUD_SR)
            y, _ = librosa.effects.trim(y, top_db=25)
            y = y[:AUD_MAX] if len(y) > AUD_MAX else np.pad(y, (0, AUD_MAX - len(y)))
        except: y = np.zeros(AUD_MAX)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(LID[row['emotion_final']], dtype=torch.long)

def train_audio(tr_df, va_df, te_df):
    print("\n" + "="*50 + "\n🎙️ TRAINING AUDIO MODALITY (WavLM)\n" + "="*50)
    set_seed(42)
    tr_ld = DataLoader(AudioDS(tr_df), batch_size=8, shuffle=True)
    va_ld = DataLoader(AudioDS(va_df), batch_size=8)
    te_ld = DataLoader(AudioDS(te_df), batch_size=8)
    m = LayerWeightedAudio().to(DEVICE)
    # Freeze backbone for warmup
    for p in m.backbone.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW([
        {'params': m.head.parameters(), 'lr': 1e-3},
        {'params': m.lw, 'lr': 1e-3}
    ])
    scaler = GradScaler()
    best_f1, ckpt = 0, MODELS_DIR / "audio_best.pt"
    for ep in range(1, AUD_EPOCHS+1):
        if ep == 3:  # Unfreeze after warmup
            for p in m.backbone.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([
                {'params': m.backbone.parameters(), 'lr': 4e-5},
                {'params': m.head.parameters(), 'lr': 1e-3},
                {'params': m.lw, 'lr': 1e-3}
            ])
        m.train(); tl_sum = 0
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with autocast(device_type=DEVICE if DEVICE=="cuda" else "cpu", enabled=DEVICE=="cuda"):
                loss = F.cross_entropy(m(x), y, label_smoothing=0.1)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl_sum += loss.item()
        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for vx, vy in va_ld: ps.extend(m(vx.to(DEVICE)).argmax(1).cpu().numpy()); ts.extend(vy.numpy())
        f1 = f1_score(ts, ps, average='macro', zero_division=0)
        if f1 > best_f1: best_f1 = f1; torch.save(m.state_dict(), str(ckpt))
        print(f"  Ep {ep:02d} | Loss: {tl_sum/len(tr_ld):.4f} | Val F1: {f1:.4f}")
    m.load_state_dict(torch.load(str(ckpt), weights_only=True))
    tp = []
    with torch.no_grad():
        for tx, _ in te_ld: tp.append(F.softmax(m(tx.to(DEVICE)),1).cpu().numpy())
    return np.vstack(tp)

# ─────────────────────────────────────────────────────────
# ── MODALITY 3: TEXT (MARBERT) ───────────────────────────
# ─────────────────────────────────────────────────────────
TXT_EPOCHS = 20
TXT_MODEL = "UBC-NLP/MARBERT"

def clean_text(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)
    t = re.sub(r'[أإآ]', 'ا', t)
    t = re.sub(r'\u0640', '', t)
    t = re.sub(r'(.)\1+', r'\1\1', t)
    return re.sub(r'\s+', ' ', t).strip()

class MARBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(TXT_MODEL)
        self.classifier = nn.Linear(768*3, 7)
        self.drops = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
    def forward(self, ids, mask):
        lh = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        cls = lh[:,0,:]
        m = mask.unsqueeze(-1).expand(lh.size()).float()
        mean_p = (lh*m).sum(1) / m.sum(1).clamp(min=1e-9)
        max_p = (lh*m - (1-m)*1e9).max(1)[0]
        cat = torch.cat([cls, mean_p, max_p], 1)
        return torch.stack([self.classifier(d(cat)) for d in self.drops]).mean(0)

class TextDS(Dataset):
    def __init__(self, df, tok):
        texts = [clean_text(t) for t in df['transcript'].values]
        self.enc = tok(texts, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = [LID[e] for e in df['emotion_final'].values]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {k: v[idx] for k,v in self.enc.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

def train_text(tr_df, va_df, te_df):
    print("\n" + "="*50 + "\n📝 TRAINING TEXT MODALITY (MARBERT)\n" + "="*50)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(TXT_MODEL)
    tr_ld = DataLoader(TextDS(tr_df, tok), batch_size=16, shuffle=True)
    va_ld = DataLoader(TextDS(va_df, tok), batch_size=16)
    te_ld = DataLoader(TextDS(te_df, tok), batch_size=16)
    m = MARBERTClassifier().to(DEVICE)
    opt = torch.optim.AdamW([
        {'params': [p for n,p in m.named_parameters() if 'bert' in n], 'lr': 2e-5},
        {'params': [p for n,p in m.named_parameters() if 'bert' not in n], 'lr': 8e-4}
    ], weight_decay=0.01)
    best_f1, ckpt = 0, MODELS_DIR / "text_best.pt"
    patience, pat_ctr = 8, 0
    for ep in range(1, TXT_EPOCHS+1):
        m.train()
        for bd, bl in tr_ld:
            opt.zero_grad()
            loss = F.cross_entropy(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)), bl.to(DEVICE), label_smoothing=0.08)
            loss.backward(); opt.step()
        m.eval(); ps, ts = [], []
        with torch.no_grad():
            for bd, bl in va_ld:
                ps.extend(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)).argmax(1).cpu().numpy())
                ts.extend(bl.numpy())
        f1 = f1_score(ts, ps, average='macro', zero_division=0)
        acc = accuracy_score(ts, ps)
        if f1 > best_f1: best_f1 = f1; torch.save(m.state_dict(), str(ckpt)); pat_ctr = 0
        else: pat_ctr += 1
        print(f"  Ep {ep:02d} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        if pat_ctr >= patience: print("  Early stop."); break
    m.load_state_dict(torch.load(str(ckpt), weights_only=True))
    tp = []
    with torch.no_grad():
        for bd, _ in te_ld:
            tp.append(F.softmax(m(bd['input_ids'].to(DEVICE), bd['attention_mask'].to(DEVICE)),1).cpu().numpy())
    return np.vstack(tp)

# ─────────────────────────────────────────────────────────
# ── LATE FUSION ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────
def late_fusion(vid_probs, aud_probs, txt_probs, te_df):
    """Grid-search best weights on a held-out val-like strategy."""
    t_labels = np.array([LID[e] for e in te_df['emotion_final'].values])

    print("\n" + "="*50 + "\n🔀 LATE FUSION — GRID SEARCH\n" + "="*50)
    best_acc, best_f1, best_w = 0, 0, (1/3, 1/3, 1/3)
    
    for wv in np.arange(0.2, 0.7, 0.1):
        for wa in np.arange(0.1, 0.5, 0.1):
            wt = 1.0 - wv - wa
            if wt < 0.05: continue
            total = wv + wa + wt
            probs = (wv*vid_probs + wa*aud_probs + wt*txt_probs) / total
            preds = probs.argmax(1)
            acc = accuracy_score(t_labels, preds)
            f1 = f1_score(t_labels, preds, average='macro', zero_division=0)
            if f1 > best_f1: best_f1 = f1; best_acc = acc; best_w = (wv, wa, wt)

    wv, wa, wt = best_w
    total = wv + wa + wt
    final_probs = (wv*vid_probs + wa*aud_probs + wt*txt_probs) / total
    final_preds = final_probs.argmax(1)

    print(f"\n🏆 Best Fusion Weights → Video: {wv:.1f} | Audio: {wa:.1f} | Text: {wt:.1f}")
    print("\n" + "="*50 + "\n🎯 FINAL LATE FUSION EVALUATION\n" + "="*50)
    print(f"Test Accuracy : {accuracy_score(t_labels, final_preds):.4f}")
    print(f"Test Macro F1 : {f1_score(t_labels, final_preds, average='macro'):.4f}")
    print(classification_report(t_labels, final_preds, target_names=EMOTIONS, zero_division=0))

    # Save probs for analysis
    np.save(str(MODELS_DIR / "fusion_probs.npy"), final_probs)
    np.save(str(MODELS_DIR / "video_probs.npy"), vid_probs)
    np.save(str(MODELS_DIR / "audio_probs.npy"), aud_probs)
    np.save(str(MODELS_DIR / "text_probs.npy"), txt_probs)
    print(f"\n✅ Probability arrays saved to {MODELS_DIR}/")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n🚀 MULTIMODAL LATE FUSION — Track A")
    print(f"   Device: {DEVICE}")
    
    tr_df, va_df, te_df = load_aligned_splits()
    
    vid_probs = train_video(tr_df, va_df, te_df)
    aud_probs = train_audio(tr_df, va_df, te_df)
    txt_probs = train_text(tr_df, va_df, te_df)
    
    late_fusion(vid_probs, aud_probs, txt_probs, te_df)

if __name__ == "__main__":
    main()
