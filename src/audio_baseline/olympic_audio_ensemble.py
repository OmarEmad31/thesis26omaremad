"""
Olympic Soft-Voting Ensemble
======================================================
This script mathematically fuses the 3 peak architectural models built today:
1. The Semantic Vault (Titan) — Looks for high-level pure emotion.
2. The 12-Layer Engine (Elite Backup) — Looks for clustered acoustic patterns.
3. The Temporal Sequence (BiLSTM) — Looks for how emotion changes over 5 seconds.

It extracts their raw probability arrays, averages them manually to cancel out 
their individual blind spots, and generates the ultimate consolidated accuracy.
"""

import os, sys, subprocess, zipfile
from collections import defaultdict

def install_deps():
    pkgs = []
    for mod, pkg in [("peft", "peft"), ("transformers", "transformers")]:
        try: __import__(mod)
        except ImportError: pkgs.append(pkg)
    if pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

if "google.colab" in sys.modules or os.path.exists("/content"):
    install_deps()

import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
import pandas as pd, numpy as np, librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model

SR = 16000
MAX_LEN = 80000
MODEL_NAME = "microsoft/wavlm-base-plus"

# ─────────────────────────────────────────────────────────
# ARCHITECTURE 1: THE TEMPORAL BILSTM 
# ─────────────────────────────────────────────────────────
class AttentionPool_BiLSTM(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1) 
        return (x * w).sum(dim=1)

class WavLM_BiLSTM_Acoustic(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME)
        cfg  = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPool_BiLSTM(d=512)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(512 + 15, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(256, num_labels)
        )
    def forward(self, wav, acoustic_features):
        outputs = self.wavlm(wav).last_hidden_state
        lstm_out, _ = self.bilstm(outputs)
        deep_features = self.attn_pool(lstm_out)
        fused_features = torch.cat([deep_features, acoustic_features], dim=-1)
        return self.classifier(fused_features)

# ─────────────────────────────────────────────────────────
# ARCHITECTURE 2: THE 12-LAYER ELITE BACKUP & 3: THE TITAN
# (They coincidentally share the exact same PyTorch topology, 
#  they just have different frozen layers during training!)
# ─────────────────────────────────────────────────────────
class AttentionPool_Standard(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)  
        return (x * w).sum(dim=1)           

class ColabTitanSER(nn.Module):
    def __init__(self, num_labels, lora_alpha=16):
        super().__init__()
        base = WavLMModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        # 12-Layer elite used alpha=32, Titan used alpha=16
        cfg  = LoraConfig(r=8, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.wavlm = get_peft_model(base, cfg)
        self.layer_weights = nn.Parameter(torch.ones(6))
        self.attn_pool = AttentionPool_Standard(768)
        self.proj_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.classifier = nn.Sequential(
            nn.Linear(768 + 4, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
    def forward(self, wav, prosody):
        wav  = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-6)
        mask = torch.ones(wav.shape[:2], device=wav.device)
        out  = self.wavlm(wav, attention_mask=mask, output_hidden_states=True)
        hidden_states = out.hidden_states[-6:]               
        w = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w[i] * hidden_states[i] for i in range(6))  
        pooled = self.attn_pool(weighted)
        return self.classifier(torch.cat([pooled, prosody], dim=-1))

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION SUITE
# ─────────────────────────────────────────────────────────
def compute_acoustic_features(y: np.ndarray) -> np.ndarray:
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    f0 = np.mean(np.nan_to_num(librosa.yin(y, fmin=65, fmax=2093))) / 500.0
    vec = np.concatenate([mfcc, [rms, f0]])
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

def extract_prosody(y: np.ndarray) -> np.ndarray:
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    f0  = librosa.yin(y, fmin=65, fmax=2093)
    f0m = float(np.nanmean(f0)) / 500.0
    f0s = float(np.nanstd(f0))  / 100.0
    vec = np.array([rms, zcr, f0m, f0s], dtype=np.float32)
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)

def get_path_map(colab_root):
    pm = {f.name: str(f) for f in colab_root.rglob("*.wav")}
    if pm: return pm
    zpath = None
    for root, _, files in os.walk("/content/drive/MyDrive"):
        if "Thesis_Audio_Full.zip" in files: zpath = os.path.join(root, "Thesis_Audio_Full.zip"); break
    with zipfile.ZipFile(zpath) as z: z.extractall("/content/dataset")
    return {f.name: str(f) for f in Path("/content/dataset").rglob("*.wav")}

# ─────────────────────────────────────────────────────────
# THE OLYMPIC ENSEMBLE ENGINE
# ─────────────────────────────────────────────────────────
def print_confusion_matrix(cm, classes):
    print("\n[CONFUSION MATRIX] rows = True, cols = Pred")
    col_w = max(len(c) for c in classes) + 2 
    print("".rjust(col_w) + "".join([c[:3].rjust(5) for c in classes]))
    for i, row in enumerate(cm):
        print(classes[i].rjust(col_w) + "".join([str(val).rjust(5) for val in row]))

def run_ensemble():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    colab_root = Path("/content/drive/MyDrive/Thesis Project")
    csv_p = colab_root / "data/processed/splits/audio_hc"
    
    # 1. Verification of assets (BiLSTM explicitly ignored due to 4% crash weights)
    paths = {
        "Backup": colab_root / "wavlm_elite_best.pt",
        "Titan":  colab_root / "colab_titan_best.pt"
    }
    
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        print(f"⚠️ Missing Models: {missing}. The ensemble will still run using the available models to boost accuracy!")
    else:
        print("✅ Both peak models detected successfully.")

    available_models = [name for name, p in paths.items() if p.exists()]
    if not available_models:
        print("❌ Cannot run ensemble. No .pt files found in Drive.")
        return

    # 2. Data Initialization
    path_map = get_path_map(colab_root)
    va_df = pd.read_csv(csv_p / "val.csv")
    classes_list = sorted(va_df["emotion_final"].unique())
    lid = {l: i for i, l in enumerate(classes_list)}
    va_df["lid"] = va_df["emotion_final"].map(lid)
    NUM_LBLS = len(lid)

    # 3. Model Loading into Memory
    models = {}
    print("\n[LOAD] Awakening Neural Architectures...")
    if "BiLSTM" in available_models:
        print("  -> Loading WavLM BiLSTM Temporal Core...")
        m = WavLM_BiLSTM_Acoustic(NUM_LBLS).to(device).eval()
        m.load_state_dict(torch.load(paths["BiLSTM"], map_location=device))
        models["BiLSTM"] = m
        
    if "Backup" in available_models:
        print("  -> Loading WavLM Elite 12-Layer Core...")
        m = ColabTitanSER(NUM_LBLS, lora_alpha=32).to(device).eval() 
        m.load_state_dict(torch.load(paths["Backup"], map_location=device))
        models["Backup"] = m
        
    if "Titan" in available_models:
        print("  -> Loading WavLM Titan Semantic Core...")
        m = ColabTitanSER(NUM_LBLS, lora_alpha=16).to(device).eval()
        m.load_state_dict(torch.load(paths["Titan"], map_location=device))
        models["Titan"] = m

    # 4. The Soft-Voting Arena
    print("\n[ARENA] Passing Validation Set through mathematical cross-examination...")
    preds, truths = [], []

    for _, row in tqdm(va_df.iterrows(), total=len(va_df), desc="Ensembling"):
        fname = Path(row["audio_relpath"]).name
        if fname not in path_map: continue
        try:
            # Universal Preprocessing
            wav, _ = librosa.load(path_map[fname], sr=SR)
            yt, _  = librosa.effects.trim(wav, top_db=30)
            if len(yt) < SR // 2: continue
            
            # Simple Center Crop Evaluation
            if len(yt) >= MAX_LEN: s = (len(yt) - MAX_LEN)//2; yt = yt[s:s+MAX_LEN]
            else: yt = np.pad(yt, (0, MAX_LEN - len(yt)))
            
            wav_t = torch.tensor(yt, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Feature Extraction
            acoustic = torch.tensor(compute_acoustic_features(yt), dtype=torch.float32).unsqueeze(0).to(device)
            prosody  = torch.tensor(extract_prosody(yt), dtype=torch.float32).unsqueeze(0).to(device)
            
            # Confidence Harvesting
            probs = []
            with torch.no_grad(), autocast("cuda"):
                if "BiLSTM" in models:
                    logits = models["BiLSTM"](wav_t, acoustic)
                    probs.append(F.softmax(logits, dim=-1))
                if "Backup" in models:
                    logits = models["Backup"](wav_t, prosody)
                    probs.append(F.softmax(logits, dim=-1))
                if "Titan" in models:
                    logits = models["Titan"](wav_t, prosody)
                    probs.append(F.softmax(logits, dim=-1))
            
            # Mathematical Fusion
            final_prob = torch.stack(probs).mean(dim=0)
            preds.append(final_prob.argmax(-1).item())
            truths.append(int(row["lid"]))
            
        except Exception as e:
            continue

    # 5. Grading
    acc = accuracy_score(truths, preds)
    f1  = f1_score(truths, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(truths, preds)
    
    print(f"\n=======================================================")
    print(f"🎉 FINAL ENSEMBLE MULTIPLIER ACHIEVED")
    print(f"  Active Models Checked : {', '.join(available_models)}")
    print(f"  Final Consolidated Acc: {acc:.4f}")
    print(f"  Final Consolidated F1 : {f1:.4f}")
    print_confusion_matrix(cm, classes_list)
    print(f"=======================================================")

if __name__ == "__main__":
    run_ensemble()
