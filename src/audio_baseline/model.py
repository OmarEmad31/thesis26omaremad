import os
import sys
import contextlib
import torch
import torch.nn as nn
from funasr import AutoModel

class Emotion2VecBaseline(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(Emotion2VecBaseline, self).__init__()
        
        print(f"--- 🧠 Initializing emotion2vec+ via AutoModel ---")
        # 1. Load the official emotion2vec+ backbone
        # We use AutoModel directly for better parameter control
        # we use hub='ms' to ensure ModelScope registry is used
        self.backbone = AutoModel(model=model_name, hub='ms', trust_remote_code=True)
        
        # 2. Classifier Head (BiLSTM + Dense)
        hidden_size = 768
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def set_backbone_trainable(self, trainable: bool):
        """Enable/Disable training for the underlying emotion2vec+ backbone."""
        # AutoModel exposes .model which is the actual nn.Module
        if hasattr(self.backbone, "model"):
            for param in self.backbone.model.parameters():
                param.requires_grad = trainable
        else:
            # Fallback if structure varies
            for param in self.backbone.parameters():
                param.requires_grad = trainable
        print(f"🔓 Backbone Trainable: {trainable}")

    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        Input: torch.Tensor of shape [batch, 160000]
        Returns: logits, embeddings
        """
        device = input_values.device
        batch_feats = []
        
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else contextlib.nullcontext():
            # AutoModel can process the whole batch at once! Much faster than the old loop.
            # We use extract_embedding=True to get the utterance-level features
            res = self.backbone.generate(input=input_values, granularity="utterance", extract_embedding=True)
            
            # The result is a list of feature dictionaries
            for item in res:
                feat = torch.tensor(item['feats'], dtype=torch.float32).to(device)
                batch_feats.append(feat)
            
        # 1. Stack into [batch, 768]
        x_stacked = torch.stack(batch_feats)
        
        # 2. Reshape into 3D for LSTM: [batch, sequence_len=1, hidden_size=768]
        x = x_stacked.unsqueeze(1) 
        
        # 3. Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # 4. Take the last time step output: [batch, 512]
        pooled = lstm_out.squeeze(1)
        
        # 5. Final Classification
        logits = self.classifier(pooled)
        
        # Return BOTH logits (for CrossEntropy) and pooled (for SCL)
        return logits, pooled
