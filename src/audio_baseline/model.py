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

    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        Input: torch.Tensor of shape [batch, 160000]
        """
        device = input_values.device
        embeddings = []
        
        # We extract features for each item in the batch
        # Note: emotion2vec+ works best on CPU for feature extraction 
        # or we can pass tensors if the pipeline supports it.
        for i in range(input_values.shape[0]):
            audio_data = input_values[i].cpu().numpy()
            
            # extract_embedding=True gets us the 768-d vector
            # granularity="utterance" gives us one vector for the whole clip
            result = self.feature_extractor(audio_data, granularity="utterance", extract_embedding=True)
            
            # result['feats'] is the numeric representation of the emotion
            feat = torch.tensor(result['feats']).to(device) # Shape: [1, 768]
            embeddings.append(feat)
            
        # Stack into [batch, 1, 768]
        x = torch.stack(embeddings)
        
        # Pass through LSTM
        # lstm_out shape: [batch, 1, 512]
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        pooled = lstm_out[:, -1, :]
        
        # Final Classification
        logits = self.classifier(pooled)
        return logits
