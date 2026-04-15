"""Emotion MLP Classifier.

The emotion2vec backbone is used ONLY as a frozen feature extractor.
Embeddings are pre-computed once and cached. This module defines the
trainable MLP head that takes 768-dim embeddings and outputs class logits.

Architecture:
  [batch, 768]
    → LayerNorm
    → Linear(768 → 512) + GELU + Dropout(0.4)
    → Linear(512 → 256) + GELU + Dropout(0.3)   ← this is the 'embedding' for SCL
    → Linear(256 → num_labels)
"""

import torch
import torch.nn as nn


class EmotionMLP(nn.Module):
    def __init__(self, input_dim: int = 768, num_labels: int = 7):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, num_labels)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, 768]  (pre-computed emotion2vec embeddings)
        Returns: logits [batch, num_labels], embeddings [batch, 256]
        """
        embeddings = self.encoder(x)        # [batch, 256]
        logits = self.classifier(embeddings) # [batch, num_labels]
        return logits, embeddings
