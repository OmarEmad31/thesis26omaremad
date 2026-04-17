"""
Method 18: The WavLM Titan (Enrichment Engine).
Backbone: microsoft/wavlm-base-plus (Speech-native Transformer)
Strategy: 
1. Phase 1: Fine-tune WavLM on 915 Gold Samples. 
2. Phase 2: Pseudo-label the 5,000 Silver recordings.
3. Phase 3: Train final Student.
Goal: 60%+ Individual Audio Accuracy.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, AutoConfig
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
BATCH_SIZE     = 4     # WavLM is heavy, use small batch + accumulation
ACCUM_STEPS    = 16    # Virtual Batch 64
NUM_EPOCHS_T   = 10    # Teacher training
NUM_EPOCHS_S   = 15    # Student training
LR             = 2e-5
SR             = 16000
MAX_DURATION   = 6
CONF_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class WavLMDataset(Dataset):
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
        label = row["emotion_final"]
        label_id = self.label2id[label] if label in self.label2id else -1
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# TRAINING LOGIC
# ---------------------------------------------------------------------------
def train_loop(model, tr_loader, va_loader, optimizer, device, epochs, ckpt_path):
    best_f1 = 0
    for epoch in range(1, epochs + 1):
        model.train(); t_loss = 0; optimizer.zero_grad()
        for b_idx, batch in enumerate(tqdm(tr_loader, desc=f"Ep{epoch}", leave=False)):
            inputs, lbs = batch["input_values"].to(device), batch["label"].to(device)
            outputs = model(inputs, labels=lbs)
            loss = outputs.loss
            (loss / ACCUM_STEPS).backward()
            if (b_idx+1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            t_loss += loss.item()
        
        model.eval(); preds, truth = [], []
        with torch.no_grad():
            for b in va_loader:
                os = model(b["input_values"].to(device))
                preds.extend(torch.argmax(os.logits, 1).cpu().numpy()); truth.extend(b["label"].numpy())
        
        va_acc, va_f1 = accuracy_score(truth, preds), f1_score(truth, preds, average="macro")
        print(f"   Ep {epoch}/{epochs} | Loss {t_loss/len(tr_loader):.3f} | Val Acc {va_acc:.3f} | F1 {va_f1:.3f}")
        if va_f1 > best_f1: 
            best_f1 = va_f1; torch.save(model.state_dict(), ckpt_path)
            print(f"   🌟 New Best F1: {best_f1:.3f}")

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 PHASE 1: WAVLM TEACHER CALIBRATION (GOLD ONLY)")
    
    # 1. Load Data
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

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    
    tr_loader = DataLoader(WavLMDataset(train_df, label2id, audio_map, processor), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(WavLMDataset(val_df, label2id, audio_map, processor), batch_size=BATCH_SIZE)

    # Calculate Balanced Weights
    y_train = train_df["emotion_final"].tolist()
    w = compute_class_weight('balanced', classes=np.array(classes), y=np.array(y_train))
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    # Initialize Model
    model = WavLMForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    teacher_ckpt = config.CHECKPOINT_DIR / "wavlm_teacher.pt"
    train_loop(model, tr_loader, va_loader, optimizer, device, NUM_EPOCHS_T, teacher_ckpt)

    print(f"\n🚀 PHASE 2: PSEUDO-LABELING (SILVER ENRICHMENT)")
    model.load_state_dict(torch.load(teacher_ckpt)); model.eval()
    
    manifest_path = Path("/content/dataset/data/processed/manifest.csv")
    if not manifest_path.exists():
        manifest_path = Path(config.SPLIT_CSV_DIR).parent.parent / "manifest.csv"
    
    full_df = pd.read_csv(manifest_path)
    gold_files = set(pd.concat([train_df, val_df, test_df])["audio_relpath"].apply(lambda x: Path(x).name))
    silver_df = full_df[~full_df["audio_relpath"].apply(lambda x: Path(x).name).isin(gold_files)].copy()
    
    silver_ds = WavLMDataset(silver_df, label2id, audio_map, processor)
    silver_loader = DataLoader(silver_ds, batch_size=BATCH_SIZE)
    
    pseudo_labels = []; confs = []
    with torch.no_grad():
        for b in tqdm(silver_loader, desc="Labeling Silver Data"):
            os = model(b["input_values"].to(device))
            probs = F.softmax(os.logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pseudo_labels.extend(pred.cpu().numpy()); confs.extend(conf.cpu().numpy())
    
    silver_df["emotion_final"] = [id2label[p] for p in pseudo_labels]
    silver_df["confidence"] = confs
    enriched_df = silver_df[silver_df["confidence"] >= CONF_THRESHOLD].copy()
    
    print(f"📊 Enriched with {len(enriched_df)} high-confidence samples.")
    final_train_df = pd.concat([train_df, enriched_df], ignore_index=True)

    print(f"\n🚀 PHASE 3: FINAL PUSH (WAVLM STUDENT)")
    final_tr_loader = DataLoader(WavLMDataset(final_train_df, label2id, audio_map, processor), batch_size=BATCH_SIZE, shuffle=True)
    
    student_model = WavLMForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)
    
    student_ckpt = config.CHECKPOINT_DIR / "final_60_audio_wavlm.pt"
    train_loop(student_model, final_tr_loader, va_loader, optimizer, device, NUM_EPOCHS_S, student_ckpt)

    print(f"\n🏆 TITAN COMPLETE. MODEL: {student_ckpt}")

if __name__ == "__main__":
    main()
