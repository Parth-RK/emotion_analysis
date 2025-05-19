import torch
import torch.nn as nn
import sys

try:
    # AutoModel loads the base transformer model (e.g., BERT, RoBERTa)
    # AutoConfig loads the model's configuration
    from transformers import AutoModel, AutoConfig
except ImportError:
    AutoModel = None
    AutoConfig = None
    print("ERROR: HuggingFace Transformers library not installed or import failed.")
    print("       Please install it: pip install transformers")
    sys.exit(1)

import config # Contains model name and other settings

class TransformerClassifier(nn.Module):
    """
    Transformer model for classification.
    Supports both single-label (with CrossEntropyLoss) and multi-label (with BCEWithLogitsLoss)
    depending on the loss function used in the training loop.
    """
    def __init__(self, model_name, n_classes):
        """
        Initializes the Transformer classifier.

        Args:
            model_name (str): Name or path of the pre-trained transformer model.
            n_classes (int): The number of output classes. This will be the size
                             of the final classification layer's output.
                             For multi-label, this is the total number of distinct labels (e.g., 28).
        """
        super().__init__()

        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library failed to import correctly.")

        try:
            print(f"Loading Transformer config: {model_name} for {n_classes} output classes")
            # AutoConfig can often infer settings like hidden_size from the model name.
            # We don't *need* to pass num_labels here for our custom head, but it's good info.
            self.config = AutoConfig.from_pretrained(model_name)
            # Optionally, set num_labels in the config if needed by the base model itself
            # (e.g., if using a built-in classification head, which we are not).
            # Let's print it for clarity:
            print(f"  Base Transformer Config loaded. Hidden size: {self.config.hidden_size}")
            # Some configs have a default num_labels, which is informational here:
            print(f"  Base Config default num_labels: {getattr(self.config, 'num_labels', 'N/A')}")


            print(f"Loading Transformer model: {model_name}")
            # Load the base transformer model without the classification head
            # Some models might load weights for a classification head if available,
            # but loading the base model via AutoModel should ideally skip this,
            # or our subsequent load_state_dict will overwrite it.
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
            print("  Transformer model loaded.")

        except OSError as e:
             print(f"\nError loading transformer model/config '{model_name}'.")
             print(f"Ensure the model name is correct and you have an internet connection if it needs downloading.")
             print(f"Or, if it's a local path, ensure the path is correct.")
             print(f"Original error: {e}")
             sys.exit(1) # Exit if the base model can't be loaded
        except Exception as e:
             print(f"An unexpected error occurred while loading the transformer model '{model_name}': {e}")
             import traceback
             traceback.print_exc()
             sys.exit(1) # Exit on other loading errors


        # Define dropout probability
        # Prioritize classifier_dropout if available, then hidden_dropout_prob, then default
        clf_dropout = getattr(self.config, 'classifier_dropout', None)
        hidden_dropout = getattr(self.config, 'hidden_dropout_prob', None)

        dropout_prob = 0.1 # Common default value
        if clf_dropout is not None:
            dropout_prob = clf_dropout
            print(f"  Using classifier_dropout from config: {dropout_prob:.2f}")
        elif hidden_dropout is not None:
            dropout_prob = hidden_dropout
            print(f"  Using hidden_dropout_prob from config: {dropout_prob:.2f}")
        else:
             print(f"  Using default dropout probability: {dropout_prob:.2f}")


        # Ensure dropout_prob is a valid number
        if not isinstance(dropout_prob, (float, int)):
            print(f"Warning: Invalid dropout value read from config ({dropout_prob}). Resetting to default 0.1.")
            dropout_prob = 0.1
        elif not (0 <= dropout_prob <= 1):
             print(f"Warning: Dropout value from config ({dropout_prob}) is outside [0, 1]. Resetting to default 0.1.")
             dropout_prob = 0.1


        self.dropout = nn.Dropout(float(dropout_prob))

        # Define the classification head
        # The output size matches the number of classes (28 for GoEmotions)
        self.classifier = nn.Linear(self.config.hidden_size, n_classes)
        print(f"  Classification head initialized with input size {self.config.hidden_size} and output size {n_classes}.")

        print(f"  TransformerClassifier based on '{model_name}' initialized.")


    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Raw logits for each class [batch_size, n_classes].
                          Sigmoid is applied *outside* the model during evaluation/inference
                          when using BCEWithLogitsLoss.
        """
        # Pass inputs through the base transformer model
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
            # Add token_type_ids=token_type_ids if your tokenizer provides them
        )

        # Get the representation for classification.
        # Common methods:
        # 1. Pooler output (if the model has a pooling layer, e.g., BERT's [CLS] token output)
        # 2. The representation of the first token ([CLS] token) from the last hidden state
        # outputs.pooler_output is generally preferred if available and meaningful for the model.
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
            # print("  Using pooler_output") # Debugging print (remove for training)
        else:
            # If pooler_output is None or not available, use the [CLS] token representation
            # (assuming the first token is the CLS token, standard for BERT-like models)
            pooled_output = outputs.last_hidden_state[:, 0]
            # print("  Using last_hidden_state[:, 0] (CLS token)") # Debugging print (remove for training)

        # Apply dropout to the pooled output
        dropped_output = self.dropout(pooled_output)

        # Pass through the classification head
        logits = self.classifier(dropped_output)

        return logits
