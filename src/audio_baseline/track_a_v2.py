"""
Egyptian Arabic SER — Feature Fusion + SupCon (Track A)
======================================================
1. Handcrafted + Frozen WavLM Extraction
2. Supervised Contrastive Learning (SupCon) for Embedding Sharpening
3. Final Classifier Training
"""

import os, sys, time, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import WavLMModel

class TrackAConfig:
    SR = 16000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    BATCH_SIZE = 32
    SUPCON_EPOCHS = 30
    SUPCON_LR = 1e-3
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    LID = {e: i for i, e in enumerate(EMOTIONS)}
    CACHE_DIR = Path("features_cache")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────
# SUPCON LOSS & MODELS
# ─────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D], labels: [B]
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # L2 Normalize
        features = F.normalize(features, p=2, dim=1)
        
        # Similarity matrix
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Log-sum-exp trick for stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1571, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    def forward(self, x): return self.net(x)

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION (Same as v56)
# ─────────────────────────────────────────────────────────
# ... (extract_handcrafted, extract_wavlm_embeddings)

def extract_handcrafted(p):
    # (Same robust logic as v56)
    try:
        y, sr = librosa.load(p, sr=16000)
        yt, _ = librosa.effects.trim(y, top_db=40)
        if len(yt)<100: return np.zeros(35)
        mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=13)
        feat = [len(yt)/sr, np.mean(librosa.feature.rms(y=yt)), np.std(librosa.feature.rms(y=yt)), 
                np.mean(librosa.feature.zero_crossing_rate(y=yt)), np.std(librosa.feature.zero_crossing_rate(y=yt)), 0]
        feat.extend(np.mean(mfcc, axis=1)); feat.extend(np.std(mfcc, axis=1))
        return np.nan_to_num(feat)
    except: return np.zeros(32) # (Adjusted to 32 if 35 failed previously)

@torch.no_grad()
def extract_wavlm_embeddings(paths, model):
    model.eval()
    embs = []
    for p in tqdm(paths, desc="Extracting WavLM"):
        try:
            y, _ = librosa.load(p, sr=16000)
            if len(y)>80000: y=y[(len(y)-80000)//2 : (len(y)-80000)//2 + 80000]
            inp = torch.from_numpy(y).float().unsqueeze(0).to(TrackAConfig.DEVICE)
            out = model(inp).last_hidden_state
            embs.append(torch.cat([out.mean(1), out.std(1)], 1).cpu().numpy().squeeze())
        except: embs.append(np.zeros(1536))
    return np.array(embs)

# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def main():
    seed_everything(TrackAConfig.SEED)
    TrackAConfig.CACHE_DIR.mkdir(exist_ok=True)
    if os.path.exists("/content/drive/MyDrive"): root = Path("/content/drive/MyDrive/Thesis Project")
    else: root = Path(__file__).parent.parent.parent
    
    csv_r = root / "data/processed/splits/text_hc"
    df = pd.read_csv(root / "audio_manifest.csv")
    orig_val_ids = set(pd.read_csv(csv_r/"val.csv")['sample_id'])
    df['split'] = df['sample_id'].apply(lambda x: 'val' if x in orig_val_ids else 'train')
    tr_df, va_df = df[df['split'] == 'train'], df[df['split'] == 'val']
    y_tr, y_va = torch.tensor(tr_df['label_id'].values), torch.tensor(va_df['label_id'].values)

    # 1. Feature Retrieval
    hc_tr_p, hc_va_p = TrackAConfig.CACHE_DIR/"hc_tr.npy", TrackAConfig.CACHE_DIR/"hc_va.npy"
    wv_tr_p, wv_va_p = TrackAConfig.CACHE_DIR/"wv_tr.npy", TrackAConfig.CACHE_DIR/"wv_va.npy"

    if not hc_tr_p.exists():
        np.save(hc_tr_p, [extract_handcrafted(p) for p in tqdm(tr_df['resolved_path'])])
        np.save(hc_va_p, [extract_handcrafted(p) for p in tqdm(va_df['resolved_path'])])
    
    if not wv_tr_p.exists():
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(TrackAConfig.DEVICE)
        np.save(wv_tr_p, extract_wavlm_embeddings(tr_df['resolved_path'], wavlm))
        np.save(wv_va_p, extract_wavlm_embeddings(va_df['resolved_path'], wavlm))

    X_hc_tr, X_wv_tr = np.load(hc_tr_p), np.load(wv_tr_p)
    X_hc_va, X_wv_va = np.load(hc_va_p), np.load(wv_va_p)
    
    # 2. Early Fusion
    X_fused_tr = np.concatenate([X_hc_tr, X_wv_tr], axis=1)
    X_fused_va = np.concatenate([X_hc_va, X_wv_va], axis=1)
    
    # 3. 🚀 SUPCON SHARPENING (The "SSL" Stage)
    print("\n🚀 [STAGE 2] Applying SupCon Feature Sharpening...")
    scaler = StandardScaler()
    X_tr_s = torch.from_numpy(scaler.fit_transform(X_fused_tr)).float().to(TrackAConfig.DEVICE)
    X_va_s = torch.from_numpy(scaler.transform(X_fused_va)).float().to(TrackAConfig.DEVICE)
    
    head = ProjectionHead(input_dim=X_fused_tr.shape[1]).to(TrackAConfig.DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=TrackAConfig.SUPCON_LR)
    criterion = SupConLoss().to(TrackAConfig.DEVICE)
    
    for epoch in range(TrackAConfig.SUPCON_EPOCHS):
        head.train()
        optimizer.zero_grad()
        features = head(X_tr_s)
        loss = criterion(features, y_tr.to(TrackAConfig.DEVICE))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0: print(f"   Epoch {epoch+1}/{TrackAConfig.SUPCON_EPOCHS} | SupCon Loss: {loss.item():.4f}")

    # 4. Final Classification
    print("\n🏆 [STAGE 3] Final Classifier on Sharpened Embeddings...")
    head.eval()
    with torch.no_grad():
        X_tr_final = head(X_tr_s).cpu().numpy()
        X_va_final = head(X_va_s).cpu().numpy()
    
    clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_tr_final, y_tr.numpy())
    
    preds = clf.predict(X_va_final)
    print(f"\nFinal Accuracy: {accuracy_score(y_va, preds):.4f}")
    print(f"Macro F1: {f1_score(y_va, preds, average='macro'):.4f}")
    print(classification_report(y_va, preds, target_names=TrackAConfig.EMOTIONS, zero_division=0))

if __name__ == "__main__": main()
