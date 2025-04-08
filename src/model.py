import torch.nn as nn
from transformers import AutoModel, AutoConfig
import config

class EmotionClassifier(nn.Module):
    """
    Transformer-based model for emotion classification.
    Uses a pre-trained transformer model and adds a classification head.
    """
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        # Load the pre-trained model configuration
        model_config = AutoConfig.from_pretrained(config.MODEL_NAME, num_labels=n_classes)

        # Load the pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(config.MODEL_NAME, config=model_config)

        # Optional: Add dropout for regularization
        # Use model_config.dropout for DistilBert or other models that use 'dropout' attribute
        # Fall back to a default value if neither attribute exists
        dropout_prob = getattr(model_config, 'hidden_dropout_prob', 
                      getattr(model_config, 'dropout', 0.1))
        self.dropout = nn.Dropout(dropout_prob)

        # Define the classification layer
        # The input dimension should match the hidden size of the transformer model
        self.classifier = nn.Linear(model_config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
        Returns:
            torch.Tensor: Logits for each class.
        """
        # Pass input through the transformer model
        # We are interested in the output corresponding to the [CLS] token for classification
        # The `pooler_output` is typically used for sequence classification tasks.
        # Some models might return last_hidden_state instead, in which case you'd take the [CLS] token's embedding ([:, 0, :])
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get the pooler output (representation of the [CLS] token)
        # For models like BERT/DistilBERT, pooler_output is suitable.
        # For others like RoBERTa, using the hidden state of the first token might be preferred,
        # but AutoModel usually provides a consistent interface or pooler_output.
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]


        # Apply dropout
        output = self.dropout(pooled_output)

        # Pass the output through the classifier layer
        logits = self.classifier(output)

        return logits

# # --- Alternative using AutoModelForSequenceClassification (Simpler) ---
# # You could replace the entire class above with just this function
# # if you don't need custom modifications to the architecture.
#
# from transformers import AutoModelForSequenceClassification
#
# def create_preset_model(n_classes):
#     """Creates a model directly using AutoModelForSequenceClassification."""
#     model = AutoModelForSequenceClassification.from_pretrained(
#         config.MODEL_NAME,
#         num_labels=n_classes
#     )
#     return model
#
# # If using the alternative above, you would change the model initialization in train.py:
# # model = model_module.create_preset_model(config.NUM_CLASSES)
# # And in predict.py:
# # model = create_preset_model(config.NUM_CLASSES) # Assuming appropriate import
# # Note: If using AutoModelForSequenceClassification, the forward pass in engine.py
# # might need slight adjustment as it directly outputs logits in a SequenceClassifierOutput object.
# # outputs = model(...) -> loss = outputs.loss, logits = outputs.logits
# # However, often just using outputs works if you only need the logits for inference/loss calc.
# # The provided engine.py code should work fine with the logits returned directly.