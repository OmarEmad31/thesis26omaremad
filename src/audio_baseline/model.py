"""Audio Emotion Classification Model (emotion2vec+ MSD)."""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Emotion2VecBaseline(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        
        # 1. Load the emotion2vec+ backbone
        # We use output_hidden_states=True for Weighted Layer Pooling
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.backbone = Wav2Vec2Model.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        
        # 2. Weighted Layer Pooling
        # We have 13 layers (Embeddings + 12 Transformer Layers)
        num_layers = self.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # 3. Multi-Sample Dropout (MSD)
        # We use 5 dropout paths just like the text baseline
        self.dropout_ops = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        
        # 4. Final Classifier
        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Args:
            input_values: (batch, seq_len)
            attention_mask: (batch, seq_len)
        """
        # Backbone forward
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        
        # Get all hidden states (list of 13 tensors: (batch, seq_len, hidden_size))
        all_hidden_states = outputs.hidden_states
        
        # Compute Weighted Layer Pooling
        # (batch, seq_len, hidden_size, 13)
        stacked_hidden_states = torch.stack(all_hidden_states, dim=-1)
        
        # Normalize weights using softmax
        weights = torch.softmax(self.layer_weights, dim=0)
        
        # Weighted sum: (batch, seq_len, hidden_size)
        fused_hidden_states = (stacked_hidden_states * weights).sum(dim=-1)
        
        # Global Average Pooling (Mean pooling over time)
        # (batch, hidden_size)
        pooled_output = fused_hidden_states.mean(dim=1)
        
        # Multi-Sample Dropout
        # We average the results of 5 different dropout paths
        logits = 0
        for dropout in self.dropout_ops:
            logits += self.classifier(dropout(pooled_output))
        
        logits = logits / len(self.dropout_ops)
        
        return logits
