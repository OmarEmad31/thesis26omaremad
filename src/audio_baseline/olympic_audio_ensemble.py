import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd, numpy as np, librosa, random
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import WavLMModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from audiomentations import Compose, AddGaussianNoise, PitchShift

class OlympicTitan(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/wavlm-large"):
        super().__init__()
        base_model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base_model, peft_config)
        self.wavlm.gradient_checkpointing_enable() 
        self.layer_weights = nn.Parameter(torch.ones(25))
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 4, 768), nn.BatchNorm1d(768), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(768, num_labels)
        )
        self.projector = nn.Sequential(nn.Linear(1024 + 4, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, wav, mask, prosody):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        outputs = self.wavlm(wav, attention_mask=mask)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_hidden = (hidden_states * w).sum(dim=0)
        mask_exp = mask[:, ::320][:, :weighted_hidden.shape[1]].unsqueeze(-1).expand(weighted_hidden.size()).float()
        pooled = torch.sum(weighted_hidden * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        fused = torch.cat([pooled, prosody], dim=-1)
        return self.classifier(fused), F.normalize(self.projector(fused), p=2, dim=1)

class SupConLoss(nn.Module):
    def __init__(self, t=0.15):
        super().__init__(); self.t = t
    def forward(self, z, l):
        mask = torch.eq(l.view(-1,1), l.view(-1,1).T).float().to(z.device)
        logits = torch.div(torch.matmul(z, z.T), self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp = torch.exp(logits) * (1 - torch.eye(z.shape[0]).to(z.device))
        log_prob = logits - torch.log(exp.sum(1, keepdim=True) + 1e-6)
        return -(mask * log_prob).sum(1).mean()

class OlympicDataset(Dataset):
    def __init__(self, df, audio_dir, augment=False):
        self.df = df; self.audio_dir = Path(audio_dir); self.max_len = 48000; self.aug = augment
        self.path_map = {f.name: f for f in self.audio_dir.rglob("*.wav")}
        self.aug_pipe = Compose([AddGaussianNoise(p=0.3), PitchShift(min_semitones=-1, max_semitones=1, p=0.3)])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        for _ in range(len(self.df)):
            row = self.df.iloc[idx]; fname = Path(row["audio_relpath"]).name
            if fname in self.path_map:
                try:
                    p = self.path_map[fname]
                    wav, _ = librosa.load(p, sr=16000)
                    yt, _ = librosa.effects.trim(wav, top_db=20)
                    yt = (yt - np.mean(yt)) / (np.std(yt) + 1e-6)
                    if len(yt) > self.max_len:
                        start = random.randint(0, len(yt)-self.max_len) if self.aug else 0
                        yt = yt[start:start+self.max_len]
                    else: yt = np.pad(yt, (0, self.max_len - len(yt)))
                    f0 = librosa.yin(yt, fmin=65, fmax=2093); rms = np.mean(librosa.feature.rms(y=yt))
                    prosody = np.array([rms, np.mean(librosa.feature.zero_crossing_rate(y=yt)), np.nanmean(f0)/500, np.nanstd(f0)/100], dtype=np.float32)
                    if self.aug: yt = self.aug_pipe(samples=yt, sample_rate=16000)
                    return {"wav": torch.tensor(yt), "prosody": torch.tensor(prosody), "label": torch.tensor(row["lid"])}
                except: pass
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError("Audio sync failed")

def evaluate(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l, _ = model(b["wav"].to(device), torch.ones_like(b["wav"]).to(device), b["prosody"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def train_fold(fold, tr_df, va_df, audio_dir, device):
    print(f"\n[FOLD {fold}] Starting Marathon..."); 
    tr_loader = DataLoader(OlympicDataset(tr_df, audio_dir, True), batch_size=4, shuffle=True)
    va_loader = DataLoader(OlympicDataset(va_df, audio_dir), batch_size=4)
    model = OlympicTitan(7).to(device)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    opt = torch.optim.AdamW([{"params": model.wavlm.parameters(), "lr": 1e-5}, {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 5e-4}], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, len(tr_loader), len(tr_loader)*15)
    l_ce = nn.CrossEntropyLoss(label_smoothing=0.1); l_sc = SupConLoss(); scaler = GradScaler()
    
    best_acc, best_f1 = 0, 0
    for ep in range(1, 16):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Fold {fold} Ep {ep}")
        for b in pbar:
            w, p, l = b["wav"].to(device), b["prosody"].to(device), b["label"].to(device)
            with torch.cuda.amp.autocast():
                logits, z = model(w, torch.ones_like(w).to(device), p)
                loss = (0.7*l_ce(logits, l) + 0.3*l_sc(z, l)) 
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        
        if ep >= 10: swa_model.update_parameters(model)
        acc, f1 = evaluate(model, va_loader, device)
        print(f"Fold {fold} Ep {ep} | Acc: {acc:.3f} | F1: {f1:.3f}")
        if acc > best_acc: 
            best_acc, best_f1 = acc, f1
            torch.save(model.state_dict(), f"fold_{fold}_best.pt")
    
    return {"acc": best_acc, "f1": best_f1}

def main():
    device = "cuda"; 
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    if colab_root.exists():
        csv_p = colab_root / "data/processed/splits/text_hc"
        audio_dir = "/content/dataset" if os.path.exists("/content/dataset") else "/content"
        print(f"[ENV] Colab Active. Data Root: {colab_root}")
    else:
        csv_p = Path("D:/Thesis Project/data/processed/splits/text_hc")
        audio_dir = Path("D:/Thesis Project/dataset")
        print(f"[ENV] Local Active. Data Root: {csv_p.parent}")

    if not csv_p.exists(): raise FileNotFoundError(f"Missing CSV splits at {csv_p}")
    
    dfs = [pd.read_csv(csv_p/f) for f in ["train.csv", "val.csv", "test.csv"]]
    full_df = pd.concat(dfs); lid = {l: i for i, l in enumerate(sorted(full_df["emotion_final"].unique()))}
    full_df["lid"] = full_df["emotion_final"].map(lid); skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    print(f"[START] OLYMPIC ENSEMBLE PRODUCTION. 5-Fold Marathon Active.")
    for fold, (t_idx, v_idx) in enumerate(skf.split(full_df, full_df["lid"])):
        out = train_fold(fold+1, full_df.iloc[t_idx], full_df.iloc[v_idx], audio_dir, device)
        results.append(out)
    
    df_res = pd.DataFrame(results)
    print("\n" + "="*40)
    print("      GRAND OLYMPIC SUMMARY      ")
    print("="*40)
    for i, r in enumerate(results): print(f"Fold {i+1}: Accuracy {r['acc']:.3f} | F1 {r['f1']:.3f}")
    print("-" * 40)
    print(f"MEAN:  Accuracy {df_res['acc'].mean():.4f} | F1 {df_res['f1'].mean():.4f}")
    print(f"STD:   Accuracy {df_res['acc'].std():.4f} | F1 {df_res['f1'].std():.4f}")
    print("="*40)

if __name__ == "__main__": main()
