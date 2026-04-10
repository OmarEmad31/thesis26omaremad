import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class MARBERTWithMultiSampleDropout(nn.Module):
    """Custom MARBERT with Multi-Sample Dropout head for better generalization."""

    def __init__(self, model_name, num_labels, id2label, label2id, num_dropouts=5, dropout_rate=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        
        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_dropouts)])
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        
        # Use [CLS] token from last hidden state
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Multi-Sample Dropout
        logits = 0
        for dropout in self.dropouts:
            logits += self.classifier(dropout(pooled_output))
        logits = logits / len(self.dropouts)
        
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def state_dict(self, *args, **kwargs):
        """Ensure all tensors are contiguous to avoid Safetensors/Checkpointing errors."""
        sd = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() for k, v in sd.items()}
