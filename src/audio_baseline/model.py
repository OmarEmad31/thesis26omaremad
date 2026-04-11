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
        # This model is the best for emotional prosody in AI today
        self.feature_extractor = pipeline(
            task=Tasks.emotion_recognition, 
            model="iic/emotion2vec_plus_base"
        )
        
        # 2. Classifier Head (BiLSTM + Dense)
        # emotion2vec-base produces 768-dimensional embeddings
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
        # The ModelScope pipeline holds the underlying model in .model
        if hasattr(self.feature_extractor, "model"):
            for param in self.feature_extractor.model.parameters():
                param.requires_grad = trainable
        print(f"🔓 Backbone Trainable: {trainable}")

    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        Input: torch.Tensor of shape [batch, 160000]
        Returns: logits, embeddings
        """
        device = input_values.device
        embeddings = []
        
        # We extract features for each item in the batch
        for i in range(input_values.shape[0]):
            audio_data = input_values[i].cpu().numpy()
            
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                result = self.feature_extractor(audio_data, granularity="utterance", extract_embedding=True)
            
            # The pipeline returns a list of results - grab the first one
            if isinstance(result, list):
                feat_data = result[0]['feats']
            else:
                feat_data = result['feats']
                
            feat = torch.tensor(feat_data, dtype=torch.float32).to(device)
            embeddings.append(feat)
            
        # 1. Stack into [batch, 768]
        x_stacked = torch.stack(embeddings)
        
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
