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
        # emotion2vec+ can return a list, a tuple, a dict, or a specialized object.
        if isinstance(outputs, (list, tuple)):
            last_hidden_state = outputs[0]
        elif isinstance(outputs, dict):
            # Fairseq/FunASR 'features_only' often returns a dict with 'x'
            last_hidden_state = outputs.get("x", outputs.get("last_hidden_state", outputs.get("features", next(iter(outputs.values())))))
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs
            
        # 3. Temporal Processing (BiLSTM)
        # last_hidden_state is [batch, seq_len, 768]
        # We pass the FULL sequence into the BiLSTM so it can hear the timeline
        lstm_out, _ = self.lstm(last_hidden_state)
        
        # 4. Global Max Pooling (Better for catching emotional peaks)
        # We take the maximum value over the time dimension (dim=1)
        pooled = torch.max(lstm_out, dim=1)[0]
        
        # 5. Final Classification
        logits = self.classifier(pooled)
        
        return logits, pooled
