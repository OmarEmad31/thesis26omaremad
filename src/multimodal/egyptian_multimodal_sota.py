"""
HArnESS-Research-SOTA: Egyptian Arabic Multimodal Emotion Recognition.
Senior ML Research Engineer Specialized Implementation.

Backbones: XLSR-53-Arabic (24-layer Mix) + MARBERT.
Fusion: Gated Cross-Attention (Text-Query driven).
Loss: Supervised Contrastive (L2-norm, Tau 0.1) + Weighted Cross-Entropy.
Imbalance Fix: WeightedRandomSampler + Macro-F1 Monitoring.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, Wav2Vec2Model, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from pathlib import Path

# Try importing audiomentations for SOTA augmentation, fallback to librosa
try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False

# Add project root
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# RESEARCH UNIT: WEIGHTED LAYER POOLING
# ---------------------------------------------------------------------------
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_layers, layer_start=0, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.ones(num_layers) if layer_weights is None else torch.tensor(layer_weights))

    def forward(self, all_hidden_states):
        # all_hidden_states: list of Tensors [BS, SeqLen, Dim]
        # We only take the layers we care about
        stacked_layers = torch.stack(all_hidden_states[self.layer_start:], dim=0) # [NumLayers, BS, SeqLen, Dim]
        weights = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        weighted_sum = (stacked_layers * weights).sum(dim=0)
        return weighted_sum

# ---------------------------------------------------------------------------
# SOTA LOSS: SUPERVISED CONTRASTIVE LOSS
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [BS, Dim] (Must be L2-normalized)
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

# ---------------------------------------------------------------------------
# SOTA ARCHITECTURE: RESEARCH MODALINK
# ---------------------------------------------------------------------------
class EgyptianMultimodalModel(nn.Module):
    def __init__(self, num_labels, audio_name, text_name):
        super().__init__()
        # 1. Audio Backbone (XLSR-53)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_name)
        # Enable hidden states for weighted pooling
        self.audio_encoder.config.output_hidden_states = True
        self.audio_pooling = WeightedLayerPooling(num_layers=25, layer_start=0) # 24 layers + embedding layer
        
        # 2. Text Backbone (MARBERT)
        self.text_encoder = AutoModel.from_pretrained(text_name)
        
        a_dim = self.audio_encoder.config.hidden_size # 1024
        t_dim = self.text_encoder.config.hidden_size   # 768
        
        # 3. Projection to common dimension
        self.audio_proj = nn.Linear(a_dim, 768)
        self.text_proj = nn.Identity() # Already 768
        
        # 4. Cross-Attention Fusion (Text Query -> Audio K/V)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        self.norm_fusion = nn.LayerNorm(768)
        
        # 5. SupCon Head (3-layer MLP)
        self.supcon_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # 6. Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, audio_mask, input_ids, text_mask):
        # Audio: Weighted Layer Pooling
        a_outputs = self.audio_encoder(input_values, attention_mask=audio_mask)
        a_feat = self.audio_pooling(a_outputs.hidden_states) # [BS, SeqA, 1024]
        a_feat = self.audio_proj(a_feat) # [BS, SeqA, 768]
        
        # Text
        t_feat = self.text_encoder(input_ids, attention_mask=text_mask).last_hidden_state # [BS, SeqT, 768]
        
        # Fusion: Cross-Attention
        attn_out, _ = self.fusion_attn(query=t_feat, key=a_feat, value=a_feat)
        fused = self.norm_fusion(attn_out + t_feat)
        
        # Pooling (Mean over sequence)
        pooled = fused.mean(dim=1)
        
        # Embeddings for SupCon
        supcon_embeds = F.normalize(self.supcon_head(pooled), p=2, dim=1)
        
        # Logits
        logits = self.classifier(pooled)
        return logits, supcon_embeds

# ---------------------------------------------------------------------------
# DATASET & RESEARCH AUGMENTATION
# ---------------------------------------------------------------------------
class ResearchDataset(Dataset):
    def __init__(self, df, audio_proc, text_tok, audio_map, augment=False):
        self.df = df
        self.proc = audio_proc
        self.tok = text_tok
        self.audio_map = audio_map
        self.augment = augment
        if augment and HAS_AUDIOMENTATIONS:
            self.aug = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if self.augment:
            if HAS_AUDIOMENTATIONS:
                audio = self.aug(samples=audio, sample_rate=16000)
            else:
                if random.random() > 0.5:
                    audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-1.5, 1.5))

        a_inputs = self.proc(audio, sampling_rate=16000, padding="max_length", max_length=160000, truncation=True, return_tensors="pt")
        t_inputs = self.tok(str(row["transcript"]), padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        
        return {
            "v": a_inputs.input_values.squeeze(0),
            "a_m": a_inputs.attention_mask.squeeze(0) if "attention_mask" in a_inputs else torch.ones(500), # Wav2Vec2 scaling
            "i": t_inputs.input_ids.squeeze(0),
            "t_m": t_inputs.attention_mask.squeeze(0),
            "l": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# RESEARCH TRAINING LOOP
# ---------------------------------------------------------------------------
def train_step(model, loader, device, optimizer, scheduler, alpha, weights):
    ce_fn = nn.CrossEntropyLoss(weight=weights)
    sc_fn = SupConLoss(temperature=0.1)
    model.train()
    t_loss, preds, truth = 0, [], []
    
    pbar = tqdm(loader, desc="Fine-tuning", leave=False)
    for b in pbar:
        b = {k: v.to(device) for k, v in b.items()}
        logits, embeds = model(b["v"], b["a_m"], b["i"], b["t_m"])
        
        loss = (1-alpha)*ce_fn(logits, b["l"]) + alpha*sc_fn(embeds, b["l"])
        
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        t_loss += loss.item()
        
        # Metrics tracking
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        truth.extend(b["l"].cpu().numpy())
        
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "f1": f"{f1_score(truth, preds, average='macro'):.3f}"})
    
    return accuracy_score(truth, preds), f1_score(truth, preds, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    A_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    T_NAME = "UBC-NLP/MARBERT"
    
    print("📂 Loading Laboratory Splits...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    
    audio_map = {}
    src = Path("/content") if Path("/content").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    train_df["label_id"] = train_df["emotion_final"].map(label2id)
    val_df["label_id"] = val_df["emotion_final"].map(label2id)
    
    # CALCULATE WEIGHTS & SAMPLER
    y_tr = train_df["label_id"].values
    weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr), dtype=torch.float32).to(device)
    samples_weight = torch.from_numpy(np.array([1.0/np.bincount(y_tr)[t] for t in y_tr]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    proc = AutoProcessor.from_pretrained(A_NAME)
    tok = AutoTokenizer.from_pretrained(T_NAME)
    
    model = EgyptianMultimodalModel(len(classes), A_NAME, T_NAME).to(device)
    
    # ---------------------------------------------------------------------------
    # RESEARCH PROTOCOL: UNFREEZE TOP LAYERS IMMEDIATELY
    # ---------------------------------------------------------------------------
    print("🚀 UNFREEZING TOP LAYERS (Research-Grade Fine-tuning)")
    for name, param in model.audio_encoder.named_parameters():
        if "layers" in name and int(name.split(".")[2]) >= 18: param.requires_grad = True # Top 6 layers
        else: param.requires_grad = False
    
    # Text backbone is fully tune-able for alignment
    for param in model.text_encoder.parameters(): param.requires_grad = True

    train_loader = DataLoader(ResearchDataset(train_df, proc, tok, audio_map, augment=True), batch_size=2, sampler=sampler)
    val_loader   = DataLoader(ResearchDataset(val_df, proc, tok, audio_map), batch_size=2)
    
    optimizer = torch.optim.AdamW([
        {"params": model.audio_encoder.parameters(), "lr": 1e-6},
        {"params": model.text_encoder.parameters(), "lr": 1e-6},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ])
    
    total_steps = len(train_loader) * 15
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    for epoch in range(1, 16):
        acc, f1 = train_step(model, train_loader, device, optimizer, scheduler, alpha=0.7, weights=weights)
        
        # Validation
        model.eval(); v_preds, v_truth = [], []
        with torch.no_grad():
            for b in tqdm(val_loader, desc="Validating", leave=False):
                l, _ = model(b["v"].to(device), b["a_m"].to(device), b["i"].to(device), b["t_m"].to(device))
                v_preds.extend(torch.argmax(l, 1).cpu().numpy()); v_truth.extend(b["l"].numpy())
        
        v_acc, v_f1 = accuracy_score(v_truth, v_preds), f1_score(v_truth, v_preds, average="macro")
        print(f"🏆 Epoch {epoch} | Val Acc: {v_acc:.3f} | Macro-F1: {v_f1:.3f}")

if __name__ == "__main__":
    main()
