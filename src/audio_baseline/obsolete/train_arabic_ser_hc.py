"""
Egyptian Specialized Fine-Tuning (High Capacity).
Backbone: jonatasgrosman/wav2vec2-large-xlsr-53-arabic
Capacity: Unfreeze 12 layers (Half the model).
Stability: Gradient Accumulation (Effective Batch 32).
Diversity: Advanced Augmentation (Stretch, Pitch, Noise).
Imbalance: Focal Loss.

Run: python -m src.audio_baseline.train_arabic_ser_hc
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
MODEL_ID        = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
MAX_WAV_SAMPLES = 5 * 16000
BATCH_SIZE      = 8            # Physical batch
ACCUM_STEPS     = 4            # Effective Batch = 32
UNFREEZE_LAYERS = 6            # Optimized capacity (6 of 24 blocks)
LR_ENCODER      = 1e-5         # Conservatory for the foundation
LR_HEAD         = 4e-4         # Aggressive for the decision maker
NUM_EPOCHS      = 30
PATIENCE        = 7

# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

# ─────────────────────────────────────────────
class EgyptianAudioDataset(Dataset):
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
        audio_path = self.audio_map.get(basename)
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            audio = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)

        if self.augment:
            # 1. Random White Noise
            if random.random() > 0.5:
                noise = 0.005 * np.random.uniform() * np.max(np.abs(audio) + 1e-9)
                audio = audio + noise * np.random.normal(size=audio.shape)
            # 2. Time Stretch (0.8x to 1.2x)
            if random.random() > 0.5:
                rate = random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=rate)
            # 3. Pitch Shift (-2 to +2 semitones)
            if random.random() > 0.5:
                steps = random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=steps)

        if len(audio) > MAX_WAV_SAMPLES:
            audio = audio[:MAX_WAV_SAMPLES]
        else:
            audio = np.pad(audio, (0, MAX_WAV_SAMPLES - len(audio)))

        inputs = self.fe(audio.astype(np.float32), sampling_rate=16000,
                         return_tensors="pt", padding="max_length",
                         max_length=MAX_WAV_SAMPLES, truncation=True)
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)
        }

# ─────────────────────────────────────────────
def build_model_hc(num_labels, device):
    print(f"  Building High-Capacity model from {MODEL_ID}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    
    # Freeze CNN foundation
    for p in model.wav2vec2.feature_extractor.parameters():
        p.requires_grad = False
    
    # Selectively unfreeze Transformer Blocks
    # Wav2Vec2-Large has 24 layers. We unfreeze the top 12.
    total_layers = len(model.wav2vec2.encoder.layers)
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        if i >= (total_layers - UNFREEZE_LAYERS):
            for p in layer.parameters(): p.requires_grad = True
        else:
            for p in layer.parameters(): p.requires_grad = False

    # Always train the head
    for p in model.projector.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {trainable/1e6:.1f}M")
    return model.to(device)

# ─────────────────────────────────────────────
def train_epoch_hc(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        iv = batch["input_values"].to(device)
        mk = batch["attention_mask"].to(device)
        lb = batch["label"].to(device)
        
        out = model(input_values=iv, attention_mask=mk, output_hidden_states=True)
        
        # 🛠️ Fix: Ensure we use the Masked Pooled Embedding for the decider
        last_hidden = out.hidden_states[-1]
        mask_float = mk.unsqueeze(1).float()
        mask_resized = F.interpolate(mask_float, size=last_hidden.size(1), mode='nearest').squeeze(1)
        mask_expanded = mask_resized.unsqueeze(-1).expand_as(last_hidden)
        emb = (last_hidden * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        proj = model.projector(emb)
        logits = model.classifier(proj)
        
        loss = loss_fn(logits, lb) / ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * ACCUM_STEPS
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_hc(model, loader, device):
    model.eval()
    preds, truth = [], []
    for batch in loader:
        iv = batch["input_values"].to(device)
        mk = batch["attention_mask"].to(device)
        out = model(input_values=iv, attention_mask=mk, output_hidden_states=True)
        
        # Must replicate masked pooling at inference
        last_hidden = out.hidden_states[-1]
        mask_float = mk.unsqueeze(1).float()
        mask_resized = F.interpolate(mask_float, size=last_hidden.size(1), mode='nearest').squeeze(1)
        mask_expanded = mask_resized.unsqueeze(-1).expand_as(last_hidden)
        emb = (last_hidden * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        proj = model.projector(emb)
        logits = model.classifier(proj)
        
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        truth.extend(batch["label"].numpy())
    return accuracy_score(truth, preds), f1_score(truth, preds, average="macro"), preds, truth

# ─────────────────────────────────────────────
def main():
    torch.manual_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"🔥  EGYPTIAN SPECIALIZED SER (HI-CAPACITY)")
    print(f"    Unfreeze: {UNFREEZE_LAYERS} Layers | Effective Batch: {BATCH_SIZE*ACCUM_STEPS}")
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
    test_loader = DataLoader(EgyptianAudioDataset(test_df, label2id, audio_map, fe), batch_size=BATCH_SIZE)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    X, y = np.arange(len(all_df)), np.array([label2id[l] for l in all_df["emotion_final"]])

    fold_results = []
    ckpt_dir = config.CHECKPOINT_DIR / "egyptian_ser_hc"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- FOLD {fold_idx+1} ---")
        
        tr_ds = EgyptianAudioDataset(all_df.iloc[tr_idx], label2id, audio_map, fe, augment=True)
        va_ds = EgyptianAudioDataset(all_df.iloc[va_idx], label2id, audio_map, fe, augment=False)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # Loss and Optimizer
        cw = compute_class_weight("balanced", classes=np.unique(y[tr_idx]), y=y[tr_idx])
        loss_fn = FocalLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device), gamma=2.0)
        
        model = build_model_hc(num_labels, device)
        optimizer = torch.optim.AdamW([
            {"params": list(model.wav2vec2.encoder.layers[-(UNFREEZE_LAYERS):].parameters()), "lr": LR_ENCODER},
            {"params": list(model.projector.parameters()), "lr": LR_HEAD},
            {"params": list(model.classifier.parameters()), "lr": LR_HEAD},
        ], weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_f1, no_improve = 0.0, 0
        ckpt = ckpt_dir / f"best_fold_{fold_idx}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            avg_loss = train_epoch_hc(model, tr_loader, optimizer, loss_fn, device)
            tr_acc, tr_f1, _, _ = evaluate_hc(model, tr_loader, device)
            va_acc, va_f1, _, _ = evaluate_hc(model, va_loader, device)
            scheduler.step(va_f1)
            
            print(f"  Ep {epoch:2d}/{NUM_EPOCHS} | Loss {avg_loss:.4f} | "
                  f"Train {tr_acc:.2f}/{tr_f1:.2f} | Val {va_acc:.2f}/{va_f1:.2f}")
            
            if va_f1 > best_f1:
                best_f1 = va_f1; torch.save(model.state_dict(), ckpt); no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE: break

        model.load_state_dict(torch.load(ckpt))
        te_acc, te_f1, te_preds, te_truth = evaluate_hc(model, test_loader, device)
        fold_results.append({"acc": te_acc, "f1": te_f1})
        print(f"  ✅ Fold Result: Acc {te_acc*100:.2f}% | F1 {te_f1*100:.2f}%")
        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"🏆  FINAL RESULTS: Mean Test Acc {np.mean([r['acc'] for r in fold_results])*100:.2f}%")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
