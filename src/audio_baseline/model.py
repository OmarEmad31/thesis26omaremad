import os
import sys
import contextlib
import torch
import torch.nn as nn
from funasr import AutoModel

class Emotion2VecBaseline(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(Emotion2VecBaseline, self).__init__()
        
        print(f"--- 🧠 Initializing emotion2vec+ via HuggingFace ---")
        # 1. Load the official emotion2vec+ backbone from HF hub
        # Better stability for Colab network
        self.backbone = AutoModel(model=model_name, hub="hf", trust_remote_code=True)
        
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
        target = self.backbone.model
        for param in target.parameters():
            param.requires_grad = trainable
            
        print(f"🔓 Backbone Trainable: {trainable}")

    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        Input: torch.Tensor of shape [batch, 160000]
        Returns: logits, embeddings
        """
        # 1. Native Batch Pass
        # We call .model directly to stay in the original torch graph.
        # 🛡️ BYPASS: We pass mask=False and features_only=True to specifically 
        # avoid the broken 'compute_mask_indices' logic in this version of FunASR.
        try:
            outputs = self.backbone.model(input_values, mask=False, features_only=True)
        except TypeError:
            # Fallback for versions that don't accept these specific kwargs
            outputs = self.backbone.model(input_values)
        
        # 2. Extract Hidden States
        # emotion2vec+ (based on Wav2Vec2) usually returns a tuple/dict
        # where the first element is the hidden states [batch, seq_len, 768]
        if isinstance(outputs, (list, tuple)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            # Fallback for direct tensor output
            last_hidden_state = outputs
            
        # 3. Global Average Pooling (over the sequence/time dimension)
        # last_hidden_state is [batch, seq_len, 768] -> we want [batch, 768]
        # We mean over dim=1 (time) to get a single vector per audio clip
        embeddings = torch.mean(last_hidden_state, dim=1)
        
        # 4. BiLSTM + Classification Head
        # Reshape for LSTM: [batch, sequence_len=1, hidden_size=768]
        x = embeddings.unsqueeze(1) 
        lstm_out, _ = self.lstm(x)
        
        # Pooled output from LSTM [batch, 512]
        pooled = lstm_out.squeeze(1)
        
        # 5. Final Classification
        logits = self.classifier(pooled)
        
        return logits, pooled
