import torch
import torch.nn as nn
import sys
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    AutoModel = None
    AutoConfig = None
    print("ERROR: HuggingFace Transformers library not installed or import failed.")
    print("       Please install it: pip install transformers")
    sys.exit(1)
import config
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library failed to import correctly.")
        try:
            print(f"Loading Transformer config: {model_name} for {n_classes} classes")
            self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
            print(f"Loading Transformer model: {model_name}")
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        except OSError as e:
             print(f"\nError loading transformer model/config '{model_name}'.")
             print(f"Ensure the model name is correct and you have an internet connection if it needs downloading.")
             print(f"Or, if it's a local path, ensure the path is correct.")
             print(f"Original error: {e}")
             sys.exit(1)
        except Exception as e:
             print(f"An unexpected error occurred while loading the transformer model '{model_name}': {e}")
             import traceback
             traceback.print_exc()
             sys.exit(1)
        clf_dropout = getattr(self.config, 'classifier_dropout', None)
        hidden_dropout = getattr(self.config, 'hidden_dropout_prob', None)
        if clf_dropout is not None:
            dropout_prob = clf_dropout
            print(f"  Using classifier_dropout: {dropout_prob:.2f}")
        elif hidden_dropout is not None:
            dropout_prob = hidden_dropout
            print(f"  Using hidden_dropout_prob: {dropout_prob:.2f}")
        else:
            dropout_prob = 0.1
            print(f"  Using default dropout: {dropout_prob:.2f}")
        if not isinstance(dropout_prob, (float, int)):
            print(f"Warning: Invalid dropout value ({dropout_prob}). Resetting to default 0.1.")
            dropout_prob = 0.1
        self.dropout = nn.Dropout(float(dropout_prob))
        self.classifier = nn.Linear(self.config.hidden_size, n_classes)
        print(f"  TransformerClassifier using '{model_name}' initialized.")
        print(f"  Using hidden size: {self.config.hidden_size}")
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits