"""
Method 18.1: The WavLM Titan (Graduation Edition).
Backbone: microsoft/wavlm-base-plus
Strategy: Direct High-Intensity Fine-tuning with Weighted Loss.
Goal: 60%+ Individual Audio Accuracy for the meeting.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_NAME     = "microsoft/wavlm-base-plus"
BATCH_SIZE     = 4     # Fits T4/A100
ACCUM_STEPS    = 16    # Virtual Batch 64
NUM_EPOCHS     = 20
LR             = 3e-5
SR             = 16000
MAX_DURATION   = 6

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class WavLMGraduationDataset(Dataset):
    def __init__(self, df, label2id, audio_map, processor):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.audio_map = audio_map
        self.processor = processor

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        basename = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(basename)
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except:
            audio = np.zeros(SR * MAX_DURATION)
        
        target_len = SR * MAX_DURATION
        if len(audio) > target_len: audio = audio[:target_len]
        else: audio = np.pad(audio, (0, target_len - len(audio)))
        
        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        label_id = self.label2id.get(row["emotion_final"], -1)
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# MAIN GRADUATION RUN
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 TITAN GRADUATION: LAUNCHING WAVLM DIRECT TUNE")
    
    # Load Gold Data
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    audio_map = {}
    src = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    # Balanced Loss Fix
    y_train = train_df["emotion_final"].tolist()
    w = compute_class_weight('balanced', classes=np.array(classes), y=np.array(y_train))
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)
    ce_fn = nn.CrossEntropyLoss(weight=class_weights)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    tr_loader = DataLoader(WavLMGraduationDataset(train_df, label2id, audio_map, processor), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(WavLMGraduationDataset(val_df, label2id, audio_map, processor), batch_size=BATCH_SIZE)
    te_loader = DataLoader(WavLMGraduationDataset(test_df, label2id, audio_map, processor), batch_size=BATCH_SIZE)

    # Initialize Model
    model = WavLMForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    total_steps = len(tr_loader) * NUM_EPOCHS // ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    best_va_f1 = 0
    ckpt_path = config.CHECKPOINT_DIR / "titan_graduation_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); t_loss = 0; optimizer.zero_grad()
        for b_idx, batch in enumerate(tqdm(tr_loader, desc=f"Ep{epoch}", leave=False)):
            inputs, lbs = batch["input_values"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = ce_fn(outputs.logits, lbs)
            (loss / ACCUM_STEPS).backward()
            
            if (b_idx+1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            t_loss += loss.item()
        
        # Eval
        model.eval(); preds, truth = [], []
        with torch.no_grad():
            for b in va_loader:
                os = model(b["input_values"].to(device))
                preds.extend(torch.argmax(os.logits, 1).cpu().numpy()); truth.extend(b["label"].numpy())
        
        va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
        
        # Test Eval
        te_preds, te_truth = [], []
        with torch.no_grad():
            for b_t in te_loader:
                os_t = model(b_t["input_values"].to(device))
                te_preds.extend(torch.argmax(os_t.logits, 1).cpu().numpy()); te_truth.extend(b_t["label"].numpy())
        te_acc, te_f1 = accuracy_score(te_truth, te_preds), f1_score(te_truth, te_preds, average="macro")

        print(f"   Ep {epoch:2d}/{NUM_EPOCHS} | Loss {t_loss/len(tr_loader):.3f} | Val {va_acc:.3f}/{va_f1:.3f} | Test {te_acc:.3f}/{te_f1:.3f}")
        
        if va_f1 > best_va_f1: 
            best_va_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"   🌟 New Peak: {va_acc*100:.1f} % Acc (Higher F1)")

    print(f"\n🏆 GRADUATION COMPLETE. PEAK F1: {best_va_f1:.3f}")

if __name__ == "__main__":
    main()
