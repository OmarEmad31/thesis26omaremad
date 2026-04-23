import os, sys, subprocess

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
    def __init__(self, df, path_map, augment=False):
        self.df = df; self.path_map = path_map; self.max_len = 48000; self.aug = augment
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
                    return {"wav": torch.tensor(yt, dtype=torch.float32), "prosody": torch.tensor(prosody), "label": torch.tensor(row["lid"])}
                except: pass
            idx = (idx + 1) % len(self.df)
        raise FileNotFoundError(f"Audio sync failed even after sledgehammer scan.")

def custom_update_bn(loader, model, device):
    model.train()
    with torch.no_grad():
        for b in tqdm(loader, desc="UpdateBN", leave=False):
            w, p = b["wav"].to(device), b["prosody"].to(device)
            m = torch.ones_like(w).to(device)
            with torch.cuda.amp.autocast():
                model(w, m, p)

def evaluate_fast(model, loader, device):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            l, _ = model(b["wav"].to(device), torch.ones_like(b["wav"]).to(device), b["prosody"].to(device))
            ps.extend(torch.argmax(l, 1).cpu().numpy()); ts.extend(b["label"].numpy())
    return accuracy_score(ts, ps), f1_score(ts, ps, average="macro")

def sliding_window_eval(model, df, path_map, device, chunk_size=48000, stride=24000):
    model.eval(); all_ps, all_ts = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SWE", leave=False):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            wav, _ = librosa.load(path_map[fname], sr=16000); yt, _ = librosa.effects.trim(wav, top_db=20)
            yt = (yt - np.mean(yt)) / (np.std(yt) + 1e-6); logits_list = []
            for start in range(0, max(1, len(yt)-chunk_size+1), stride):
                seg = yt[start:start+chunk_size]
                if len(seg) < chunk_size: seg = np.pad(seg, (0, chunk_size-len(seg)))
                f0 = librosa.yin(seg, fmin=65, fmax=2093); rms = np.mean(librosa.feature.rms(y=seg))
                p_vec = torch.tensor([rms, np.mean(librosa.feature.zero_crossing_rate(y=seg)), np.nanmean(f0)/500, np.nanstd(f0)/100], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    l, _ = model(torch.tensor(seg).unsqueeze(0).to(device), torch.ones(1, chunk_size).to(device), p_vec)
                    logits_list.append(F.softmax(l, dim=1))
            if logits_list:
                final_logits = torch.mean(torch.stack(logits_list), dim=0)
                all_ps.append(final_logits.argmax(1).item()); all_ts.append(row["lid"])
        except: pass
    return accuracy_score(all_ts, all_ps), f1_score(all_ts, all_ps, average="macro")

def train_fold(fold, tr_df, va_df, path_map, device):
    print(f"\n[FOLD {fold}] Training..."); 
    tr_loader = DataLoader(OlympicDataset(tr_df, path_map, True), batch_size=4, shuffle=True, drop_last=True)
    va_loader = DataLoader(OlympicDataset(va_df, path_map), batch_size=4)
    model = OlympicTitan(7).to(device)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    opt = torch.optim.AdamW([{"params": model.wavlm.parameters(), "lr": 1e-5}, {"params": [p for n, p in model.named_parameters() if "wavlm" not in n], "lr": 5e-4}], weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, len(tr_loader), len(tr_loader)*25)
    l_ce = nn.CrossEntropyLoss(label_smoothing=0.1); l_sc = SupConLoss(); scaler = GradScaler()
    
    best_acc, best_f1 = 0, 0
    for ep in range(1, 26):
        model.train()
        for b in tqdm(tr_loader, desc=f"Fold {fold} Ep {ep}", leave=False):
            w, p, l = b["wav"].to(device), b["prosody"].to(device), b["label"].to(device)
            with torch.cuda.amp.autocast():
                logits, z = model(w, torch.ones_like(w).to(device), p); loss = (0.9*l_ce(logits, l) + 0.1*l_sc(z, l))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        
        if ep >= 12: swa_model.update_parameters(model)
        acc, f1 = evaluate_fast(model, va_loader, device)
        print(f"Fold {fold} Ep {ep} | Acc: {acc:.3f} | F1: {f1:.3f}")
        if acc > best_acc: best_acc, best_f1 = acc, f1
    
    print(f"[FOLD {fold}] Finalizing SWA & SWE...")
    custom_update_bn(tr_loader, swa_model, device)
    swe_acc, swe_f1 = sliding_window_eval(swa_model, va_df, path_map, device)
    print(f"[RESULT] Fold {fold} Final SWE (SWA): Acc {swe_acc:.3f} | F1 {swe_f1:.3f}")
    return {"acc": swe_acc, "f1": swe_f1}

def sledgehammer_scan(root_dir):
    pm = {}
    print(f"[SCAN] Sledgehammer scan starting at {root_dir}...")
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".wav"): pm[f] = os.path.join(root, f)
    return pm

def main():
    device = "cuda"; colab_root = Path("/content/drive/MyDrive/Thesis Project")
    if colab_root.exists():
        csv_p = colab_root / "data/processed/splits/text_hc"
        path_map = sledgehammer_scan(str(colab_root))
        if not path_map:
            print("[WARN] Still nothing. Scanning /content/drive/MyDrive...")
            path_map = sledgehammer_scan("/content/drive/MyDrive")
    else:
        csv_p = Path("D:/Thesis Project/data/processed/splits/text_hc")
        path_map = sledgehammer_scan("D:/Thesis Project/dataset")

    if not path_map: raise FileNotFoundError("Fatal: Sledgehammer could not find any audio.")
    print(f"[SUCCESS] Sledgehammer found {len(path_map)} audio files.")

    dfs = [pd.read_csv(csv_p/f) for f in ["train.csv", "val.csv", "test.csv"]]
    full_df = pd.concat(dfs); lid = {l: i for i, l in enumerate(sorted(full_df["emotion_final"].unique()))}
    full_df["lid"] = full_df["emotion_final"].map(lid); skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    print(f"[START] OLYMPIC ENSEMBLE PRODUCTION. Deep Scan Verified.")
    for fold, (t_idx, v_idx) in enumerate(skf.split(full_df, full_df["lid"])):
        out = train_fold(fold+1, full_df.iloc[t_idx], full_df.iloc[v_idx], path_map, device); results.append(out)
    
    df_res = pd.DataFrame(results)
    print("\n" + "="*40 + "\n      GRAND OLYMPIC SUMMARY      \n" + "="*40)
    for i, r in enumerate(results): print(f"Fold {i+1}: Acc {r['acc']:.3f} | F1 {r['f1']:.3f}")
    print(f"-"*40 + f"\nMEAN: Acc {df_res['acc'].mean():.4f} | F1 {df_res['f1'].mean():.4f}\n" + "="*40)

if __name__ == "__main__": main()
