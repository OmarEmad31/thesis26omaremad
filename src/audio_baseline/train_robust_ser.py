"""
Method 8: Robust-ER Pivot.
Backbone: audeering/wav2vec2-large-robust-24-ft-emotion-msp-corpus
Strategy: Pre-trained "Emotion-First" brain, SCL, and Mask-Aware Pooling.
Goal: 58%+ Individual Accuracy.

Run: python -m src.audio_baseline.train_robust_ser
"""

import os, sys, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import librosa
from tqdm import tqdm
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
MODEL_ID        = "audeering/wav2vec2-large-robust-24-ft-emotion-msp-corpus"
MAX_WAV_SAMPLES = 6 * 16000
BATCH_SIZE      = 4            # Memory intensive model
ACCUM_STEPS     = 8            # Virtual batch 32
UNFREEZE_LAYERS = 4            # Keep elite emotion weights frozen, tune last 4
LR_ENCODER      = 8e-6         # Very low to protect emotion brain
LR_HEAD         = 3e-4         # Aggressive for classification
NUM_EPOCHS      = 25
SCL_TEMP        = 0.1
SCL_WEIGHT      = 0.3          # High importance on clustering

# ─────────────────────────────────────────────
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        bs = labels.size(0)
        diag = torch.ones_like(mask) - torch.eye(bs, device=mask.device)
        mask = mask * diag
        max_s, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - max_s.detach()
        exp_s = torch.exp(sim) * diag
        log_p = sim - torch.log(exp_s.sum(1, keepdim=True) + 1e-8)
        valid = mask.sum(1) > 0
        if valid.any():
            return -(mask[valid] * log_p[valid]).sum(1).div(mask[valid].sum(1) + 1e-8).mean()
        return torch.tensor(0.0, device=features.device, requires_grad=True)

# ─────────────────────────────────────────────
class RobustAudioDataset(Dataset):
    def __init__(self, df, label2id, audio_map, fe, augment=False):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.audio_map = audio_map
        self.fe = fe
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(basename)
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
        except Exception:
            audio = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)

        if self.augment:
            if random.random() > 0.5:
                audio = audio + 0.005 * np.random.uniform() * np.max(np.abs(audio)+1e-9) * np.random.normal(size=audio.shape)
            if random.random() > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-1.5, 1.5))

        if len(audio) > MAX_WAV_SAMPLES: audio = audio[:MAX_WAV_SAMPLES]
        else: audio = np.pad(audio, (0, MAX_WAV_SAMPLES - len(audio)))

        inputs = self.fe(audio.astype(np.float32), sampling_rate=16000, return_tensors="pt", 
                         padding="max_length", max_length=MAX_WAV_SAMPLES, truncation=True)
        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)
        }

# ─────────────────────────────────────────────
def build_model_robust(num_labels, device):
    print(f"🚀 Loading {MODEL_ID}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    # Freeze everything
    for p in model.parameters(): p.requires_grad = False
    # Unfreeze top N layers
    total = len(model.wav2vec2.encoder.layers)
    for i in range(total - UNFREEZE_LAYERS, total):
        for p in model.wav2vec2.encoder.layers[i].parameters(): p.requires_grad = True
    # Heads
    for p in model.projector.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True
    return model.to(device)

def main():
    torch.manual_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"🔥  ROBUST-ER PIVOT (MSP-CORPUS BACKBONE) + SCL")
    print(f"    Target: 58% | Backbone: {MODEL_ID.split('/')[-1]}")
    print(f"{'='*70}\n")

    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p

    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label2id = {l: i for i, l in enumerate(sorted(all_df["emotion_final"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    test_loader = DataLoader(RobustAudioDataset(test_df, label2id, audio_map, fe), batch_size=BATCH_SIZE)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    X, y = np.arange(len(all_df)), np.array([label2id[l] for l in all_df["emotion_final"]])

    ckpt_dir = config.CHECKPOINT_DIR / "robust_ser_scl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n🌀 FOLD {fold_idx+1}")
        tr_loader = DataLoader(RobustAudioDataset(all_df.iloc[tr_idx], label2id, audio_map, fe, augment=True), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(RobustAudioDataset(all_df.iloc[va_idx], label2id, audio_map, fe, augment=False), batch_size=BATCH_SIZE)

        model = build_model_robust(num_labels, device)
        cw = compute_class_weight("balanced", classes=np.unique(y[tr_idx]), y=y[tr_idx])
        ce_fn  = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
        scl_fn = SupervisedContrastiveLoss(SCL_TEMP)
        optimizer = torch.optim.AdamW([
            {"params": model.wav2vec2.encoder.layers[-UNFREEZE_LAYERS:].parameters(), "lr": LR_ENCODER},
            {"params": model.projector.parameters(), "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ])

        best_f1, no_improve = 0, 0
        ckpt = ckpt_dir / f"best_fold_{fold_idx}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train(); total_loss = 0; optimizer.zero_grad()
            for b_idx, batch in enumerate(tqdm(tr_loader, desc=f"Ep{epoch}", leave=False)):
                iv, mk, lb = batch["input_values"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                out = model(iv, attention_mask=mk, output_hidden_states=True)
                
                # Mask-Aware Pooling for SCL
                last_hidden = out.hidden_states[-1]
                mask_float = mk.unsqueeze(1).float()
                mask_resized = F.interpolate(mask_float, size=last_hidden.size(1), mode='nearest').squeeze(1)
                mask_expanded = mask_resized.unsqueeze(-1).expand_as(last_hidden)
                emb = (last_hidden * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                
                # In robust model, logits are calculated differently usually, but we forced SequenceClassification logic
                loss = (1 - SCL_WEIGHT) * ce_fn(out.logits, lb) + SCL_WEIGHT * scl_fn(emb, lb)
                (loss / ACCUM_STEPS).backward()
                
                if (b_idx+1) % ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()
                total_loss += loss.item()

            model.eval(); preds, truth = [], []
            with torch.no_grad():
                for batch in va_loader:
                    logits = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                    preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    truth.extend(batch["label"].numpy())
            
            va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
            print(f"   Ep {epoch}/{NUM_EPOCHS} | Loss {total_loss/len(tr_loader):.4f} | Val Acc {va_acc:.3f} | Val F1 {va_f1:.3f}")

            if va_f1 > best_f1: best_f1 = va_f1; torch.save(model.state_dict(), ckpt); no_improve = 0
            else: no_improve += 1
            if no_improve >= 6: break

        model.load_state_dict(torch.load(ckpt)); model.eval()
        p_t, t_t = [], []
        with torch.no_grad():
            for b in test_loader:
                ls = model(b["input_values"].to(device), attention_mask=b["attention_mask"].to(device)).logits
                p_t.extend(torch.argmax(ls, 1).cpu().numpy()); t_t.extend(b["label"].numpy())
        te_acc, te_f1 = accuracy_score(t_t, p_t), f1_score(t_t, p_t, average="macro")
        fold_results.append({"acc": te_acc, "f1": te_f1})
        print(f"📊 FOLD {fold_idx+1} Result: Acc {te_acc*100:.1f}% | F1 {te_f1*100:.1f}%")
        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\nFinal: Mean Acc {np.mean([r['acc'] for r in fold_results])*100:.1f}%")

if __name__ == "__main__":
    main()
