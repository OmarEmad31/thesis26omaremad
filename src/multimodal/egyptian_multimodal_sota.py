"""
HArnESS-SOTA: Egyptian Arabic Multimodal Emotion Recognition.
Senior ML Engineer Specialized Implementation.

Backbones: Wav2Vec2-XLSR-Arabic + MARBERT.
Fusion: Cross-Attention (Text Query -> Audio Key/Value).
Learning: Supervised Contrastive Learning (SupCon) + Joint Optimization.
"""

import os, sys, random, torch, librosa, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Wav2Vec2Model, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from pathlib import Path

# Add project root for local imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# SOTA LOSS: SUPERVISED CONTRASTIVE LOSS
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        if len(features.shape) < 3:
            features = features.unsqueeze(1) # [BS, 1, Dim]
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Calculate similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss

# ---------------------------------------------------------------------------
# SOTA ARCHITECTURE: CROSS-ATTENTION FUSION
# ---------------------------------------------------------------------------
class EgyptianMultimodalModel(nn.Module):
    def __init__(self, num_labels, audio_name, text_name):
        super().__init__()
        # 1. Feature Extractors
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_name)
        self.text_encoder = AutoModel.from_pretrained(text_name)
        
        audio_dim = self.audio_encoder.config.hidden_size # 1024 for large
        text_dim = self.text_encoder.config.hidden_size   # 768 for base
        
        # 2. Projection to Common Space
        self.audio_projector = nn.Linear(audio_dim, 512)
        self.text_projector = nn.Linear(text_dim, 512)
        
        # 3. Cross-Attention Fusion
        # Text queries attend to Audio keys/values
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm_fusion = nn.LayerNorm(512)
        
        # 4. SupCon Projector
        self.supcon_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256) # Normalized latent space
        )
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask_audio, input_ids, attention_mask_text):
        # Audio extraction
        audio_feats = self.audio_encoder(input_values, attention_mask=attention_mask_audio).last_hidden_state
        audio_proj = self.audio_projector(audio_feats) # [BS, Seq_A, 512]
        
        # Text extraction
        text_feats = self.text_encoder(input_ids, attention_mask=attention_mask_text).last_hidden_state
        text_proj = self.text_projector(text_feats)   # [BS, Seq_T, 512]
        
        # Fusion: Cross-Attention
        # Query: Text, Key/Value: Audio
        attn_out, _ = self.cross_modal_attn(query=text_proj, key=audio_proj, value=audio_proj)
        fusion_out = self.norm_fusion(attn_out + text_proj)
        
        # Global Average Pooling across the text sequence
        pooled = fusion_out.mean(dim=1)
        
        # Heads
        supcon_embeds = F.normalize(self.supcon_projector(pooled), p=2, dim=1)
        logits = self.classifier(pooled)
        
        return logits, supcon_embeds

# ---------------------------------------------------------------------------
# DATASET & AUGMENTATION
# ---------------------------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, df, audio_processor, text_tokenizer, audio_map):
        self.df = df
        self.audio_proc = audio_processor
        self.tokenizer = text_tokenizer
        self.audio_map = audio_map
        self.max_audio_len = 160000 # 10 seconds

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Audio Preprocessing
        bn = Path(str(row["audio_relpath"]).replace("\\", "/")).name
        path = self.audio_map.get(bn)
        audio, _ = librosa.load(path, sr=16000, mono=True)
        # Random Pitch Shift (Egyptian specific nuance)
        if random.random() > 0.8:
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-1, 1))
            
        audio_inputs = self.audio_proc(audio, sampling_rate=16000, padding="max_length", max_length=self.max_audio_len, truncation=True, return_tensors="pt")
        
        # Text Preprocessing
        text_inputs = self.tokenizer(str(row["text"]), padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        
        return {
            "input_values": audio_inputs.input_values.squeeze(0),
            "attention_mask_audio": audio_inputs.attention_mask.squeeze(0) if "attention_mask" in audio_inputs else torch.ones(self.max_audio_len // 320), # Rough mask if needed
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask_text": text_inputs.attention_mask.squeeze(0),
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# TRAINING LOGIC
# ---------------------------------------------------------------------------
def train_stage(model, loader, val_loader, device, optimizer, scheduler, alpha=0.7):
    ce_fn = nn.CrossEntropyLoss()
    supcon_fn = SupConLoss(temperature=0.07)
    
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="Training")
    
    for batch in progress:
        b = {k: v.to(device) for k, v in batch.items()}
        logits, embeds = model(b["input_values"], b["attention_mask_audio"], b["input_ids"], b["attention_mask_text"])
        
        l_ce = ce_fn(logits, b["label"])
        l_sc = supcon_fn(embeds, b["label"])
        loss = alpha * l_sc + (1 - alpha) * l_ce
        
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    # Validation
    model.eval(); preds, truth = [], []
    with torch.no_grad():
        for batch in val_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            l, _ = model(b["input_values"], b["attention_mask_audio"], b["input_ids"], b["attention_mask_text"])
            preds.extend(torch.argmax(l, 1).cpu().numpy()); truth.extend(b["label"].cpu().numpy())
    
    return accuracy_score(truth, preds), f1_score(truth, preds, average="macro")

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    AUDIO_NAME = "facebook/wav2vec2-large-xlsr-53-arabic"
    TEXT_NAME  = "UBC-NLP/MARBERT"
    
    # 1. Load All Data Splits
    print("📂 Loading Dataset Splits...")
    train_df = pd.read_csv(config.SPLIT_CSV_DIR / "train.csv")
    val_df   = pd.read_csv(config.SPLIT_CSV_DIR / "val.csv")
    
    # Resolve Audio Paths
    audio_map = {}
    src = Path("/content/dataset") if Path("/content/dataset").exists() else Path(config.DATA_ROOT).parent
    for p in src.rglob("*.wav"): audio_map[p.name] = p
    
    # Label Mapping
    classes = sorted(train_df["emotion_final"].unique())
    label2id = {l: i for i, l in enumerate(classes)}
    train_df["label_id"] = train_df["emotion_final"].map(label2id)
    val_df["label_id"] = val_df["emotion_final"].map(label2id)
    
    # 2. Initialize Processors & Model
    print("💎 Initializing SOTA Transformers...")
    audio_proc = AutoProcessor.from_pretrained(AUDIO_NAME)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_NAME)
    
    model = EgyptianMultimodalModel(num_labels=len(classes), audio_name=AUDIO_NAME, text_name=TEXT_NAME).to(DEVICE)
    
    train_ds = MultimodalDataset(train_df, audio_proc, text_tokenizer, audio_map)
    val_ds   = MultimodalDataset(val_df, audio_proc, text_tokenizer, audio_map)
    
    # Use small batch for Wav2Vec2-Large memory constraints
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=2)
    
    # ---------------------------------------------------------------------------
    # STAGE 1: WARMUP (FROZEN BACKBONES)
    # ---------------------------------------------------------------------------
    print("\n🔥 STAGE 1: TRAINING FUSION HEADS (FROZEN BACKBONES)")
    for param in model.audio_encoder.parameters(): param.requires_grad = False
    for param in model.text_encoder.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    num_steps = len(train_loader) * 3
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)
    
    for epoch in range(1, 4):
        acc, f1 = train_stage(model, train_loader, val_loader, DEVICE, optimizer, scheduler, alpha=0.7)
        print(f"📊 Stage 1 | Ep {epoch} | Val Acc: {acc:.3f} | F1: {f1:.3f}")
    
    # ---------------------------------------------------------------------------
    # STAGE 2: UNFREEZE AND FINE-TUNE (DIFFERENTIAL LR)
    # ---------------------------------------------------------------------------
    print("\n🚀 STAGE 2: END-TO-END FINE-TUNING (UNFROZEN)")
    for param in model.audio_encoder.parameters(): param.requires_grad = True
    for param in model.text_encoder.parameters(): param.requires_grad = True
    
    # Use differential learning rates to protect pre-trained weights
    optimizer = torch.optim.AdamW([
        {"params": model.audio_encoder.parameters(), "lr": 2e-6}, # Very low for Audio
        {"params": model.text_encoder.parameters(), "lr": 1e-6},  # Very low for Text
        {"params": model.classifier.parameters(), "lr": 5e-4}     # Normal for Head
    ])
    
    num_steps = len(train_loader) * 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=num_steps)
    
    best_acc = 0
    checkpoint_path = config.CHECKPOINT_DIR / "sota_multimodal_egyptian.pt"
    
    for epoch in range(1, 11):
        acc, f1 = train_stage(model, train_loader, val_loader, DEVICE, optimizer, scheduler, alpha=0.5)
        print(f"🏆 Stage 2 | Ep {epoch} | Val Acc: {acc:.3f} | F1: {f1:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   🌟 New Graduation Peak: {best_acc*100:.2f}%")

    print(f"\n✅ TRAINING COMPLETE. FINAL PEAK ACCURACY: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
