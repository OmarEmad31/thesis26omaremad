import os
import sys
import contextlib
import torch
import torch.nn as nn
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class Emotion2VecBaseline(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(Emotion2VecBaseline, self).__init__()
        
        print(f"--- 🧠 Initializing emotion2vec+ via ModelScope ---")
        # 1. Load the official emotion2vec+ pipeline
        self.backbone = pipeline(
            task=Tasks.emotion_recognition, 
            model=model_name
        )
        
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
        # Find the actual torch parameters deep inside the ModelScope wrapper
        target = self.backbone.model
        
        # If it's a GenericFunASR wrapper, the real module is in .model
        if hasattr(target, "model") and isinstance(target.model, torch.nn.Module):
            target = target.model
            
        for param in target.parameters():
            param.requires_grad = trainable
            
        print(f"🔓 Backbone Trainable: {trainable}")

    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        Input: torch.Tensor of shape [batch, 160000]
        Returns: logits, embeddings
        """
        device = input_values.device
        batch_feats = []
        
        # DISCOVERY: Find the real module to check requires_grad
        target_mod = self.backbone.model
        if hasattr(target_mod, "model"):
            target_mod = target_mod.model
        
        # Switch to training mode if backbone is being fine-tuned
        is_finetuning = any(p.requires_grad for p in target_mod.parameters())
        
        with torch.set_grad_enabled(is_finetuning):
            # Process each item (pipeline usually expects single items or lists)
            for i in range(input_values.shape[0]):
                audio_data = input_values[i].cpu().numpy()
                
                # Silence internal pipeline logging for speed
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    result = self.backbone(audio_data, granularity="utterance", extract_embedding=True)
                
                if isinstance(result, list): result = result[0]
                feat = torch.tensor(result['feats'], dtype=torch.float32).to(device)
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
