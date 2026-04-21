"""
HArnESS-Deep-Alignment: Egyptian Arabic SER (The 50% Milestone Push).
Joint Text+Audio training (Alignment) resulting in a SOTA Individual Audio Student.

Teacher: UBC-NLP/MARBERT (Jointly Trained).
Student: microsoft/wavlm-base-plus.
Innovation: Cross-Modal KL-Divergence (Learning to hear what the text sees).
Inference: Strict Individual Audio Mode (Text model discarded).
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
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
# DATASET: JOINT (AUDIO + TEXT)
# ---------------------------------------------------------------------------
class JointDataset(Dataset):
    def __init__(self, df, audio_map, tokenizer):
        self.df = df; self.audio_map = audio_map; self.tokenizer = tokenizer
        self.max_wav = 160000; self.max_text = 64

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # Audio Processing
        if len(audio) > self.max_wav: audio = audio[:self.max_wav]
        else: audio = np.pad(audio, (0, self.max_wav - len(audio)))
            
        # Text Processing
        text = str(row["transcript"])
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_text)
        
        return {
            "wav": torch.tensor(audio, dtype=torch.float32),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# ARCHITECTURE: DUAL-MODAL ALIGNMENT
# ---------------------------------------------------------------------------
class DeepAlignmentSOTA(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # 1. THE STUDENT (Audio)
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.audio_classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(768, num_labels))
        
        # 2. THE TEACHER (Text)
        self.marbert = AutoModel.from_pretrained("UBC-NLP/MARBERT")
        self.text_classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, num_labels))

    def forward(self, wav, input_ids, attention_mask, mode="audio"):
        # Audio Path (The Student)
        if mode == "audio" or mode == "joint":
            wav_norm = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
            a_out = self.wavlm(wav_norm).last_hidden_state.mean(dim=1)
            a_logits = self.audio_classifier(a_out)
            if mode == "audio": return a_logits

        # Text Path (The Teacher)
        if mode == "text" or mode == "joint":
            t_out = self.marbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            t_logits = self.text_classifier(t_out)
            if mode == "text": return t_logits

        return a_logits, t_logits # Joint mode

# ---------------------------------------------------------------------------
# TRAINING & LOSS
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("🏗️ Initializing Deep-Alignment SOTA (Dual-Modality Joint Training)...")
    
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    
    classes = sorted(train_df["emotion_final"].unique()); lid = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    p = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for f in p.rglob(ext): audio_map[f.name] = f

    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(train_df["label_id"]), y=train_df["label_id"]), dtype=torch.float32).to(device)
    model = DeepAlignmentSOTA(len(classes)).to(device)
    
    tr_loader = DataLoader(JointDataset(train_df, audio_map, tokenizer), batch_size=16, shuffle=True)
    va_loader = DataLoader(JointDataset(val_df, audio_map, tokenizer), batch_size=16)
    te_loader = DataLoader(JointDataset(test_df, audio_map, tokenizer), batch_size=16)

    # 50x Head Accelerator
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-5},
        {"params": model.marbert.parameters(), "lr": 1e-5},
        {"params": model.audio_classifier.parameters(), "lr": 5e-4},
        {"params": model.text_classifier.parameters(), "lr": 5e-4},
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader)*2, num_training_steps=len(tr_loader)*30)
    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    ce_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    print("\n🚀 LAUNCHING DEEP-ALIGNMENT (T-Factor Sprint)")
    best_va = 0
    for epoch in range(1, 31):
        model.train(); pbar = tqdm(tr_loader, desc=f"Align Ep{epoch}")
        for b in pbar:
            w, ids, m, l = b["wav"].to(device), b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
            a_log, t_log = model(w, ids, m, mode="joint")
            
            # THE JOINT LOSS: Audio Learn, Text Learn, and Alignment (Student copy Teacher)
            l_audio = ce_criterion(a_log, l)
            l_text = ce_criterion(t_log, l)
            l_align = kl_criterion(F.log_softmax(a_log, dim=1), F.softmax(t_log.detach(), dim=1))
            
            loss = 1.0 * l_audio + 1.0 * l_text + 0.5 * l_align
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        
        # MONITOR INDIVIDUAL AUDIO SOTA ONLY
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for b in va_loader:
                al = model(b["wav"].to(device), None, None, mode="audio")
                ps.extend(torch.argmax(al, 1).cpu().numpy()); ts.extend(b["label"].numpy())
        
        va = accuracy_score(ts, ps); print(f"💎 Audio-Only Result: VAL Acc: {va:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "deep_alignment_best.pt")

if __name__ == "__main__": main()
