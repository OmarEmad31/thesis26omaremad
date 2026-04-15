"""Emotion2Vec+ classification model.

Architecture:
  emotion2vec_plus_base backbone (frozen initially, unfrozen at UNFREEZE_EPOCH)
  → BiLSTM (2 layers, 256 hidden, bidirectional)
  → Mean Pooling over time
  → Linear classifier

Input: raw normalized waveform [batch, time] (float32)
Output: (logits [batch, num_labels], pooled [batch, 512])
"""

import torch
import torch.nn as nn
from funasr import AutoModel


class Emotion2VecBaseline(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        print(f"🧠 Loading backbone: {model_name}")
        # Load emotion2vec+ backbone via FunASR
        self.backbone = AutoModel(model=model_name, hub="hf", trust_remote_code=True)

        # Classifier head
        hidden_size = 768  # emotion2vec+ output dim

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    # ------------------------------------------------------------------
    def set_backbone_trainable(self, trainable: bool):
        for param in self.backbone.model.parameters():
            param.requires_grad = trainable
        status = "UNFROZEN 🔓" if trainable else "frozen 🔒"
        print(f"Backbone {status}")

    # ------------------------------------------------------------------
    def _extract_hidden_states(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Pass raw waveform through backbone and extract [batch, seq_len, 768].

        emotion2vec+ expects float32 waveforms at 16 kHz.
        We use features_only=True to skip the self-supervised masking head
        (which has a known bug with 'add_masks' in the installed version).
        """
        input_values = input_values.float()

        try:
            outputs = self.backbone.model(input_values, mask=False, features_only=True)
        except TypeError:
            # Some FunASR versions don't accept those kwargs — plain call
            outputs = self.backbone.model(input_values)

        # emotion2vec can return a dict, tuple, list or a dataclass
        if isinstance(outputs, dict):
            # features_only=True returns {"x": tensor, ...}
            hidden = outputs.get("x",
                      outputs.get("last_hidden_state",
                      outputs.get("features",
                      next(iter(outputs.values())))))
        elif isinstance(outputs, (list, tuple)):
            hidden = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs

        # Safety: if still not a tensor try first subscript
        if not isinstance(hidden, torch.Tensor):
            hidden = hidden[0]

        return hidden  # [batch, seq_len, 768]

    # ------------------------------------------------------------------
    def forward(self, input_values, attention_mask=None, **kwargs):
        # 1. Extract features [batch, seq_len, 768]
        hidden = self._extract_hidden_states(input_values)

        # 2. BiLSTM over full sequence
        lstm_out, _ = self.lstm(hidden)  # [batch, seq_len, 512]

        # 3. Mean pooling over time
        pooled = lstm_out.mean(dim=1)   # [batch, 512]

        # 4. Classify
        logits = self.classifier(pooled)  # [batch, num_labels]

        return logits, pooled
