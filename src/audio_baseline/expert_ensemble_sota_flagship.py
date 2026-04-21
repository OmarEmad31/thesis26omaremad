"""
HArnESS-Expert-Ensemble: Egyptian Arabic SER (The 50% Milestone Push).
Dual-Backbone Fusion: WavLM (Prosody) + Whisper (Acoustic Context).

Backbone A: microsoft/wavlm-base-plus (768 dim).
Backbone B: openai/whisper-base (512 dim, Encoder-only).
Fusion: Concatenation (768 + 512 = 1280 dim).
Loss: Hybrid SupCon (Geometric) + Label Smoothed CrossEntropy.
"""

import os, sys, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, WhisperModel, get_cosine_schedule_with_warmup
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
# LOSS: SUPCON (Geometric Clustering)
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# ARCHITECTURE: DUAL-BACKBONE ENSEMBLE
# ---------------------------------------------------------------------------
class ExpertEnsembleSER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Backbone A: WavLM (Prosody/Speaker)
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        # Backbone B: Whisper (Acoustic Context)
        self.whisper = WhisperModel.from_pretrained("openai/whisper-base")
        
        # Total dim: 768 (WavLM) + 512 (Whisper Encoder) = 1280
        total_dim = 768 + 512
        
        self.projector = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask=None, mode="train"):
        # 1. Normalize
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        
        # 2. WavLM Features (last hidden state)
        # WavLM takes raw waveform
        wavlm_out = self.wavlm(wav).last_hidden_state
        if mask is not None:
            w_mask = mask[:, ::320][:, :wavlm_out.shape[1]]
            w_mask_exp = w_mask.unsqueeze(-1).expand(wavlm_out.size()).float()
            wavlm_pooled = torch.sum(wavlm_out * w_mask_exp, 1) / torch.clamp(w_mask_exp.sum(1), min=1e-9)
        else:
            wavlm_pooled = wavlm_out.mean(dim=1)
            
        # 3. Whisper Features (encoder-only)
        # Note: Whisper usually needs Mel-Spectrogram, but we can pass raw wav 
        # to its feature extractor (AutoFeatureExtractor) if we have it. 
        # For simplicity in this script, we pass raw if the model allows, otherwise we need spectrogram.
        # Actually, WhisperModel in Transformers.from_pretrained EXPECTS [BS, 80, 3000] Mel spectrum.
        # For this script to be "Expert", we will use the Whisper Encoder last hidden state.
        # WE NEED SPECTROGRAM FOR WHISPER. We will handle this in Dataset.
        pass

# ---------------------------------------------------------------------------
# RE-REFINING ARCHITECTURE FOR SPECTROGRAM COMPATIBILITY
# ---------------------------------------------------------------------------
class ExpertEnsembleSERFinal(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-base")
        
        # 768 (WavLM) + 512 (Whisper Base Encoder)
        self.total_dim = 768 + 512
        
        self.projector = nn.Sequential(
            nn.Linear(self.total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, wav, mel, mask=None, mode="train"):
        # 1. WavLM (Raw Audio)
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        w_out = self.wavlm(wav).last_hidden_state
        if mask is not None:
            m = mask[:, ::320][:, :w_out.shape[1]].unsqueeze(-1).expand(w_out.size()).float()
            w_pooled = torch.sum(w_out * m, 1) / torch.clamp(m.sum(1), min=1e-9)
        else:
            w_pooled = w_out.mean(dim=1)
            
        # 2. Whisper (Mel Spectrogram)
        # Whisper encoder output is [BS, 1500, 512]
        wh_out = self.whisper.encoder(mel).last_hidden_state
        wh_pooled = wh_out.mean(dim=1) 
        
        # 3. Fusion
        fused = torch.cat([w_pooled, wh_pooled], dim=-1)
        
        if mode == "contrast": return self.projector(fused)
        elif mode == "classify": return self.classifier(fused)
        else: return self.projector(fused), self.classifier(fused)

# ---------------------------------------------------------------------------
# DATASET: DUAL-INPUT (RAW + MEL)
# ---------------------------------------------------------------------------
from transformers import AutoFeatureExtractor
class ExpertDataset(Dataset):
    def __init__(self, df, audio_map):
        self.df = df; self.audio_map = audio_map
        self.wav_len = 160000 # 10s
        self.whisper_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn); audio, _ = librosa.load(path, sr=16000, mono=True)
        
        # Raw / Mask for WavLM
        raw_mask = np.zeros(self.wav_len, dtype=np.float32)
        if len(audio) > self.wav_len:
            raw_audio = audio[:self.wav_len]; raw_mask[:] = 1.0
        else:
            raw_mask[:len(audio)] = 1.0; raw_audio = np.pad(audio, (0, self.wav_len - len(audio)))
        
        # Mel for Whisper
        # Whisper expects exactly 30s mel (3000 frames) according to its config
        mel_inputs = self.whisper_extractor(audio, sampling_rate=16000, return_tensors="pt")
        
        return {
            "wav": torch.tensor(raw_audio, dtype=torch.float32),
            "mel": mel_inputs.input_features.squeeze(0),
            "mask": torch.tensor(raw_mask, dtype=torch.float32),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l = model(b["wav"].to(device), b["mel"].to(device), b["mask"].to(device), mode="classify")
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("🏗️ Initializing Expert Ensemble SOTA (WavLM + Whisper)...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    test_df  = pd.read_csv(config.SPLIT_CSV_DIR / "test.csv")
    classes = sorted(train_df["emotion_final"].unique()); lid = {l: i for i, l in enumerate(classes)}
    for df in [train_df, val_df, test_df]: df["label_id"] = df["emotion_final"].map(lid)
    
    audio_map = {}
    data_search_path = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for ext in ["*.wav", "*.WAV", "*.Wav"]:
        for p in data_search_path.rglob(ext): audio_map[p.name] = p

    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)

    model = ExpertEnsembleSERFinal(len(classes)).to(device)
    l_sup = SupConLoss(temperature=0.07); l_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    tr_loader = DataLoader(ExpertDataset(train_df, audio_map), batch_size=16, shuffle=True) # Smaller batch for dual backbone
    va_loader = DataLoader(ExpertDataset(val_df, audio_map), batch_size=16)
    te_loader = DataLoader(ExpertDataset(test_df, audio_map), batch_size=16)

    print("\n🔥 STAGE 1: DUAL-LENS WARMUP (Heads only)")
    for param in model.wavlm.parameters(): param.requires_grad = False
    for param in model.whisper.parameters(): param.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(1, 6):
        model.train(); pbar = tqdm(tr_loader, desc=f"Warmup Ep{epoch}")
        for b in pbar:
            w, mel, l, m = b["wav"].to(device), b["mel"].to(device), b["label"].to(device), b["mask"].to(device)
            p, c = model(w, mel, m, mode="both")
            loss = 1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)
            opt.zero_grad(); loss.backward(); opt.step()
        va, vf = evaluate(model, va_loader, device); print(f"📈 Warmup Epoch {epoch} | Val Acc: {va:.3f} | Val F1: {vf:.3f}")

    print("\n🚀 STAGE 2: EXPERT ENSEMBLE FINE-TUNING (50% Push)")
    for param in model.wavlm.parameters(): param.requires_grad = True
    for param in model.whisper.parameters(): param.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 1e-5}], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=len(tr_loader), num_training_steps=len(tr_loader)*25)
    
    best_va = 0
    for epoch in range(1, 26):
        model.train(); pbar = tqdm(tr_loader, desc=f"Expert Ep{epoch}")
        for b in pbar:
            w, mel, l, m = b["wav"].to(device), b["mel"].to(device), b["label"].to(device), b["mask"].to(device)
            p, c = model(w, mel, m, mode="both")
            loss = 1.0 * l_sup(p, l) + 1.0 * l_ce(c, l)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        va, vf = evaluate(model, va_loader, device); ta, tf = evaluate(model, te_loader, device)
        print(f"🏆 Epoch {epoch} Results: VAL Acc: {va:.3f} F1: {vf:.3f} | TEST Acc: {ta:.3f} F1: {tf:.3f}")
        if va > best_va:
            best_va = va; torch.save(model.state_dict(), config.CHECKPOINT_DIR / "expert_ensemble_best.pt")

if __name__ == "__main__": main()
