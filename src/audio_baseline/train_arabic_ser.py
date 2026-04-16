"""
Arabic-Native SER Fine-Tuning.
Backbone: jonatasgrosman/wav2vec2-large-xlsr-53-arabic
 — explicitly fine-tuned on Arabic speech (CommonVoice AR + MGB-2).
Strategy: Freeze CNN + first 20 of 24 transformer blocks.
           Unfreeze last 4 blocks + new 7-class classification head.
           Run: python -m src.audio_baseline.train_arabic_ser
"""

import os, sys, random
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
MAX_WAV_SAMPLES = 5 * 16000   # 5 seconds
BATCH_SIZE      = 8
UNFREEZE_LAYERS = 4            # unfreeze last N transformer blocks
LR              = 2e-5         # lower than WavLM — model is already Arabic-specialized
NUM_EPOCHS      = 25
PATIENCE        = 6
SCL_WEIGHT      = 0.1
SCL_TEMP        = 0.1

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
class ArabicAudioDataset(Dataset):
    def __init__(self, df, label2id, audio_map, fe):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.audio_map = audio_map
        self.fe = fe

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        audio_path = self.audio_map.get(basename)
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            audio = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)
        if len(audio) > MAX_WAV_SAMPLES:
            audio = audio[:MAX_WAV_SAMPLES]
        else:
            audio = np.pad(audio, (0, MAX_WAV_SAMPLES - len(audio)))
        inputs = self.fe(audio.astype(np.float32), sampling_rate=16000,
                         return_tensors="pt", padding=False,
                         max_length=MAX_WAV_SAMPLES, truncation=True)
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)
        }


# ─────────────────────────────────────────────
def build_model(num_labels, device):
    print(f"  Loading {MODEL_ID}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last UNFREEZE_LAYERS transformer blocks
    total = len(model.wav2vec2.encoder.layers)
    for i in range(total - UNFREEZE_LAYERS, total):
        for p in model.wav2vec2.encoder.layers[i].parameters():
            p.requires_grad = True

    # Always train the classification head
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total_p/1e6:.1f}M  ({100*trainable/total_p:.1f}%)")
    return model.to(device)


# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, ce_fn, scl_fn, device, scheduler=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        iv = batch["input_values"].to(device)
        lb = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(input_values=iv, output_hidden_states=True)
        emb  = out.hidden_states[-1].mean(dim=1)
        loss = (1 - SCL_WEIGHT) * ce_fn(out.logits, lb) + SCL_WEIGHT * scl_fn(emb, lb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, truth = [], []
    for batch in loader:
        logits = model(input_values=batch["input_values"].to(device)).logits
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        truth.extend(batch["label"].numpy())
    return accuracy_score(truth, preds), f1_score(truth, preds, average="macro"), preds, truth


# ─────────────────────────────────────────────
def main():
    torch.manual_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"🔥  Arabic-Native SER  |  {MODEL_ID.split('/')[-1]}")
    print(f"    Unfreeze last {UNFREEZE_LAYERS} blocks  |  LR={LR}  |  {device.upper()}")
    print(f"{'='*60}\n")

    # Audio map
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"):
        audio_map[p.name] = p
    print(f"Mapped {len(audio_map)} audio tracks.\n")

    # Data
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    all_df   = pd.concat([train_df, val_df], ignore_index=True)

    label2id   = {l: i for i, l in enumerate(sorted(all_df["emotion_final"].unique()))}
    id2label   = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {label2id}\n")

    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    test_loader = DataLoader(
        ArabicAudioDataset(test_df, label2id, audio_map, fe),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    X = np.arange(len(all_df))
    y = np.array([label2id[l] for l in all_df["emotion_final"]])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    fold_results = []
    ckpt_dir = config.CHECKPOINT_DIR / "arabic_ser"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx+1} / 5")
        print(f"{'='*60}")

        fold_tr = all_df.iloc[tr_idx].reset_index(drop=True)
        fold_va = all_df.iloc[va_idx].reset_index(drop=True)

        tr_loader = DataLoader(ArabicAudioDataset(fold_tr, label2id, audio_map, fe),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(ArabicAudioDataset(fold_va, label2id, audio_map, fe),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        cw    = compute_class_weight("balanced", classes=np.unique(y[tr_idx]), y=y[tr_idx])
        ce_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
        scl_fn = SupervisedContrastiveLoss(SCL_TEMP)

        model = build_model(num_labels, device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, steps_per_epoch=len(tr_loader),
            epochs=NUM_EPOCHS, pct_start=0.1, anneal_strategy="cos"
        )

        best_f1, no_improve = 0.0, 0
        ckpt = ckpt_dir / f"fold_{fold_idx}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train_epoch(model, tr_loader, optimizer, ce_fn, scl_fn, device, scheduler)
            tr_acc, tr_f1, _, _ = evaluate(model, tr_loader, device)
            va_acc, va_f1, _, _ = evaluate(model, va_loader, device)
            print(f"  Ep {epoch:2d}/{NUM_EPOCHS} | Loss {loss:.4f} | "
                  f"Train {tr_acc:.3f}/{tr_f1:.3f} | Val {va_acc:.3f}/{va_f1:.3f}")
            if va_f1 > best_f1:
                best_f1 = va_f1; torch.save(model.state_dict(), ckpt); no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  ⏹  Early stop — best Val F1: {best_f1:.4f}"); break

        model.load_state_dict(torch.load(ckpt))
        te_acc, te_f1, te_preds, te_truth = evaluate(model, test_loader, device)
        fold_results.append({"val_f1": best_f1, "test_acc": te_acc, "test_f1": te_f1})
        print(f"\n  ✅ FOLD {fold_idx+1} | Val F1 {best_f1*100:.1f}% | Test Acc {te_acc*100:.1f}% | Test F1 {te_f1*100:.1f}%")
        del model; torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"🏆  Arabic-Native SER  —  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Mean Val  F1 : {np.mean([r['val_f1']  for r in fold_results])*100:.2f}%")
    print(f"  Mean Test Acc: {np.mean([r['test_acc'] for r in fold_results])*100:.2f}%")
    print(f"  Mean Test F1 : {np.mean([r['test_f1']  for r in fold_results])*100:.2f}%")
    print(f"{'='*60}")
    print("\n📊  Per-Class (last fold):")
    print(classification_report(te_truth, te_preds, target_names=[id2label[i] for i in range(num_labels)]))


if __name__ == "__main__":
    main()
