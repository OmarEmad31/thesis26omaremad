"""
Method 11: Self-Supervised Learning (SSL) Pre-training.
Task: SimCLR (Contrastive Learning) on the full 5,702 file dataset.
Goal: Create "Egyptian-Native" ResNet weights before supervised fine-tuning.

Run: python -m src.audio_baseline.pretrain_ssl_spectrogram
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_EPOCHS   = 10  # SSL takes longer, but we do 10 to see effect
LR           = 3e-4
MAX_DURATION = 5
SR           = 16000
TEMP         = 0.1 # Temperature for Contrastive Loss

# ─────────────────────────────────────────────
# DATASET (UNLABELED)
# ─────────────────────────────────────────────
class SSLAudioDataset(Dataset):
    def __init__(self, audio_paths):
        self.paths = audio_paths

    def __len__(self): return len(self.paths)

    def _get_spec(self, path):
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except:
            audio = np.zeros(SR * MAX_DURATION)
        
        target_len = SR * MAX_DURATION
        if len(audio) > target_len: audio = audio[:target_len]
        else: audio = np.pad(audio, (0, target_len - len(audio)))
        
        # Static + Delta + Delta-Delta
        mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=SR, n_mels=128, n_fft=400, hop_length=160)
        static = librosa.power_to_db(mel, ref=np.max)
        d1 = librosa.feature.delta(static, order=1)
        d2 = librosa.feature.delta(static, order=2)
        
        img = np.stack([static, d1, d2], axis=0)
        
        # Random SpecAugment for Contrastive "View"
        f_mask = random.randint(5, 15)
        f0 = random.randint(0, 128 - f_mask)
        img[:, f0:f0+f_mask, :] = img.min()
        
        t_mask = random.randint(10, 30)
        t0 = random.randint(0, img.shape[2] - t_mask)
        img[:, :, t0:t0+t_mask] = img.min()

        # Normalize
        img_min = img.min(axis=(1, 2), keepdims=True)
        img_max = img.max(axis=(1, 2), keepdims=True)
        img = (img - img_min) / (img_max - img_min + 1e-9)
        return torch.tensor(img, dtype=torch.float32)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # Return two different augmented views of the same clip
        view1 = self._get_spec(path)
        view2 = self._get_spec(path)
        return view1, view2

# ─────────────────────────────────────────────
# SSL MODEL (SimCLR)
# ─────────────────────────────────────────────
class SimCLRResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        dim_mlp = self.backbone.fc.in_features
        # Replace projection head for SSL
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 128)
        )

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2

def contrastive_loss(z1, z2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    bs = z1.size(0)
    
    # Cosine similarity matrix
    # [2B, 2B]
    features = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(features, features.T) / TEMP
    
    # Mask out self-similarities
    mask = torch.eye(2 * bs, device=z1.device).bool()
    logits = logits.masked_fill(mask, -1e9)
    
    # Labels for positives (z1_i matches z2_i)
    labels = torch.cat([torch.arange(bs, 2*bs), torch.arange(bs)], dim=0).to(z1.device)
    return F.cross_entropy(logits, labels)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("🧠  METHOD 11: SSL PRE-TRAINING (EGYPTIAN SPEECH DOMAIN)")
    print(f"{'='*70}\n")

    # 1. Collect all 5,702 paths
    data_dir = Path("dataset/Final Modalink Dataset MERGED")
    all_paths = list(data_dir.rglob("*.wav"))
    print(f"Loaded {len(all_paths)} unlabeled Egyptian audio clips.")

    dataset = SSLAudioDataset(all_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SimCLRResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    ckpt_path = config.CHECKPOINT_DIR / "egyptian_ssl_resnet18.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}")
        for v1, v2 in pbar:
            optimizer.zero_grad()
            z1, z2 = model(v1.to(device), v2.to(device))
            loss = contrastive_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} Complete | Avg SSL Loss: {avg_loss:.4f}")
        
        # Save "Backbone-only" weights
        torch.save(model.backbone.state_dict(), ckpt_path)
    
    print(f"\n🚀 SSL Pre-training Complete! weights saved to {ckpt_path}")
    print("Next step: Run Method 12 (Supervised Fine-tuning with these weights).")

if __name__ == "__main__":
    main()
