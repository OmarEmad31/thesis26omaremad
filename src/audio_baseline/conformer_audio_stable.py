import os, sys, subprocess, zipfile

# Auto-Install for Colab
def install_deps():
    try:
        import audiomentations, peft, transformers
    except ImportError:
        print("[INIT] Installing SOTA dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "audiomentations", "peft", "transformers", "-q"])
        import audiomentations, peft, transformers

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd, numpy as np, librosa, random
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

class ConformerStableTitan(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-large"):
        super().__init__()
        base_model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        peft_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none")
        self.wavlm = get_peft_model(base_model, peft_config)
        self.wavlm.gradient_checkpointing_enable() 
        
        # The Conformer Brain
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, dropout=0.2, batch_first=True)
        self.conformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 4, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(512, num_labels)
        )

    def forward(self, wav, mask, prosody):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav, attention_mask=mask)
        hidden = outputs.last_hidden_state 
        ctx = self.conformer(hidden) 
        pooled = ctx.mean(dim=1)
        fused = torch.cat([pooled, prosody], dim=-1)
        return self.classifier(fused)

class OlympicDataset(Dataset):
    def __init__(self, df, path_map, augment=False):
        self.df = df; self.path_map = path_map; self.max_len = 48000; self.aug = augment
        self.aug_pipe = Compose([AddGaussianNoise(p=0.4), PitchShift(p=0.4)])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            row = self.df.iloc[idx]; fname = Path(row["audio_relpath"]).name
            if fname in self.path_map:
                try:
                    p = self.path_map[fname]
                    wav, _ = librosa.load(p, sr=16000); yt, _ = librosa.effects.trim(wav, top_db=20)
                    if len(yt) > self.max_len:
                        start = random.randint(0, len(yt)-self.max_len) if self.aug else 0
                        yt = yt[start:start+self.max_len]
                    else: yt = np.pad(yt, (0, self.max_len - len(yt)))
                    f0 = librosa.yin(yt, fmin=65, fmax=2093); rms = np.mean(librosa.feature.rms(y=yt))
                    p_vec = np.array([rms, np.mean(librosa.feature.zero_crossing_rate(y=yt)), np.nanmean(f0)/500, np.nanstd(f0)/100], dtype=np.float32)
                    if self.aug: yt = self.aug_pipe(samples=yt, sample_rate=16000)
                    return {"wav": torch.tensor(yt, dtype=torch.float32), "prosody": torch.tensor(p_vec), "label": torch.tensor(row["lid"])}
                except: pass
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError("Audio sync failed")

def train():
    device = "cuda"; colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/text_hc"
    
    print("[INIT] Scanning audio...")
    path_map = {f.name: f for f in colab_root.rglob("*.wav")}
    if not path_map:
        zname = "Thesis_Audio_Full.zip"
        zpath = colab_root / zname
        if not zpath.exists():
            for f in colab_root.iterdir():
                if f.suffix == ".zip": zpath = f; break
        print(f"[INIT] Extracting {zpath.name} to /content/dataset...")
        with zipfile.ZipFile(zpath, 'r') as z: z.extractall("/content/dataset")
        path_map = {f.name: f for f in Path("/content/dataset").rglob("*.wav")}
    
    tr_df = pd.read_csv(csv_p/"train.csv"); va_df = pd.read_csv(csv_p/"val.csv"); te_df = pd.read_csv(csv_p/"test.csv")
    lid = {l: i for i, l in enumerate(sorted(tr_df["emotion_final"].unique()))}
    for df in [tr_df, va_df, te_df]: df["lid"] = df["emotion_final"].map(lid)
    
    tr_loader = DataLoader(OlympicDataset(tr_df, path_map, True), batch_size=4, shuffle=True, drop_last=True)
    va_loader = DataLoader(OlympicDataset(va_df, path_map), batch_size=4)
    te_loader = DataLoader(OlympicDataset(te_df, path_map), batch_size=4)
    
    model = ConformerStableTitan(7).to(device)
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 1e-6},
        {"params": model.conformer.parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)
    
    sch = get_cosine_schedule_with_warmup(opt, len(tr_loader)*2, len(tr_loader)*30)
    scaler = GradScaler(); crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"[START] CONFORMER STABLE TITAN. Sync verified.")
    for ep in range(1, 41):
        model.train(); tr_loss = 0
        for b in tqdm(tr_loader, desc=f"Ep {ep}", leave=False):
            w, p, l = b["wav"].to(device), b["prosody"].to(device), b["label"].to(device)
            with autocast("cuda"):
                logits = model(w, torch.ones_like(w).to(device), p); loss = crit(logits, l)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tr_loss += loss.item()
        
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for b in va_loader:
                l = model(b["wav"].to(device), torch.ones_like(b["wav"]).to(device), b["prosody"].to(device))
                ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
        acc = accuracy_score(ts, ps); f1 = f1_score(ts, ps, average="macro")
        
        tps, tts = [], []
        with torch.no_grad():
            for b in te_loader:
                l = model(b["wav"].to(device), torch.ones_like(b["wav"]).to(device), b["prosody"].to(device))
                tps.extend(torch.argmax(l, 1).cpu().numpy()); tts.extend(b["label"].numpy())
        tacc = accuracy_score(tts, tps)
        
        print(f"Ep {ep} | Loss: {tr_loss/len(tr_loader):.3f} | Val Acc: {acc:.3f} | Test Acc: {tacc:.3f} | F1: {f1:.3f}")
        if acc > 0.55: torch.save(model.state_dict(), f"stable_titan_ep{ep}.pt")

if __name__ == "__main__": train()
