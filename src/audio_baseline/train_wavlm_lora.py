"""
LoRA Fine-Tuning of WavLM-base-plus for Egyptian Arabic Speech Emotion Recognition
Run: python -m src.audio_baseline.train_wavlm_lora

Key design choices:
  - WavLM-base-plus (96M params) frozen except for LoRA adapters (~2M trainable)
  - Audio strictly truncated to 4 seconds to prevent A100 OOM
  - SCL loss on pooled hidden states + weighted CrossEntropy
  - 5-Fold Stratified CV with blind test evaluation
"""

import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, AutoFeatureExtractor
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
WAVLM_MODEL    = "microsoft/wavlm-base-plus"
MAX_WAV_SAMPLES = 4 * 16000   # strictly 4 seconds — 7x smaller than previous 30-sec window
BATCH_SIZE     = 8            # safe on A100 with 4-sec sequences + LoRA
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.1
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 20
PATIENCE       = 5

# ──────────────────────────────────────────────────────────────────────────────
# SUPERVISED CONTRASTIVE LOSS
# ──────────────────────────────────────────────────────────────────────────────
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        bs = labels.size(0)
        diag_mask = torch.ones_like(mask) - torch.eye(bs, device=mask.device)
        mask = mask * diag_mask

        max_sim, _ = torch.max(sim, dim=1, keepdim=True)
        sim_stable = sim - max_sim.detach()
        exp_sim = torch.exp(sim_stable) * diag_mask
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        valid = mask.sum(1) > 0
        if valid.any():
            mean_pos = (mask[valid] * log_prob[valid]).sum(1) / (mask[valid].sum(1) + 1e-8)
            return -mean_pos.mean()
        return torch.tensor(0.0, device=features.device, requires_grad=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
class AudioEmotionDataset(Dataset):
    def __init__(self, df, label2id, audio_map, feature_extractor):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.audio_map = audio_map
        self.fe = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name

        audio_path = self.audio_map.get(basename)
        try:
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            audio = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)

        # Hard truncate / pad to exactly 4 seconds
        if len(audio) > MAX_WAV_SAMPLES:
            audio = audio[:MAX_WAV_SAMPLES]
        else:
            audio = np.pad(audio, (0, MAX_WAV_SAMPLES - len(audio)))

        inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt",
                         padding=False, max_length=MAX_WAV_SAMPLES, truncation=True)

        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(self.label2id[row["emotion_final"]], dtype=torch.long)
        }


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────────────────────────────────────
def build_lora_model(num_labels, device):
    print(f"Loading {WAVLM_MODEL} and injecting LoRA adapters...")
    base_model = WavLMForSequenceClassification.from_pretrained(
        WAVLM_MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()   # shows how few params we actually train
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, ce_fn, scl_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Training", leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_values=input_values, output_hidden_states=True)
        logits = outputs.logits
        # Use last hidden state mean pool for SCL
        hidden = outputs.hidden_states[-1].mean(dim=1)

        loss_ce  = ce_fn(logits, labels)
        loss_scl = scl_fn(hidden, labels)
        loss = (1 - config.SCL_WEIGHT) * loss_ce + config.SCL_WEIGHT * loss_scl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        logits = model(input_values=batch["input_values"].to(device)).logits
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"🔥  WavLM-base-plus + LoRA  |  Device: {device.upper()}")
    print(f"{'='*60}\n")

    # ── Audio map ──────────────────────────────────────────────────────────────
    print("Mapping physical .wav files on disk...")
    audio_map = {}
    search_dir = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in search_dir.rglob("*.wav"):
        audio_map[p.name] = p
    print(f"Mapped {len(audio_map)} audio tracks.\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")

    all_df   = pd.concat([train_df, val_df], ignore_index=True)
    label2id = {lbl: i for i, lbl in enumerate(sorted(all_df["emotion_final"].unique()))}
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {label2id}\n")

    # ── Feature extractor ──────────────────────────────────────────────────────
    feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)

    # ── Fixed test loader ──────────────────────────────────────────────────────
    test_loader = DataLoader(
        AudioEmotionDataset(test_df, label2id, audio_map, feature_extractor),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # ── K-Fold ─────────────────────────────────────────────────────────────────
    X = np.arange(len(all_df))
    y = np.array([label2id[l] for l in all_df["emotion_final"]])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    fold_results = []
    lora_ckpt_dir = config.CHECKPOINT_DIR / "lora"
    lora_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx+1} / 5")
        print(f"{'='*60}")

        fold_train = all_df.iloc[train_idx].reset_index(drop=True)
        fold_val   = all_df.iloc[val_idx].reset_index(drop=True)

        train_loader = DataLoader(
            AudioEmotionDataset(fold_train, label2id, audio_map, feature_extractor),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            AudioEmotionDataset(fold_val, label2id, audio_map, feature_extractor),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Class weighting
        class_w = compute_class_weight("balanced", classes=np.unique(y[train_idx]), y=y[train_idx])
        ce_fn  = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(device))
        scl_fn = SupervisedContrastiveLoss(temperature=config.SCL_TEMP)

        model     = build_lora_model(num_labels, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_f1 = 0.0
        no_improve  = 0
        ckpt_path   = lora_ckpt_dir / f"fold_{fold_idx}_best.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, ce_fn, scl_fn, device)
            scheduler.step()
            train_acc, train_f1 = evaluate(model, train_loader, device)
            val_acc,   val_f1   = evaluate(model, val_loader,   device)

            print(f"  Ep {epoch:2d}/{NUM_EPOCHS} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                  f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), ckpt_path)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"  ⏹  Early stop (patience={PATIENCE}). Best Val F1: {best_val_f1:.4f}")
                break

        # ── Test evaluation on best checkpoint ──────────────────────────────
        model.load_state_dict(torch.load(ckpt_path))
        test_acc, test_f1 = evaluate(model, test_loader, device)
        fold_results.append({"val_f1": best_val_f1, "test_acc": test_acc, "test_f1": test_f1})

        print(f"\n  ✅ FOLD {fold_idx+1} | Best Val F1: {best_val_f1*100:.2f}% "
              f"| Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1*100:.2f}%")

        del model; torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏆  WavLM LoRA  —  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Mean Val  F1 : {np.mean([r['val_f1']   for r in fold_results])*100:.2f}%")
    print(f"  Mean Test Acc: {np.mean([r['test_acc']  for r in fold_results])*100:.2f}%")
    print(f"  Mean Test F1 : {np.mean([r['test_f1']   for r in fold_results])*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
