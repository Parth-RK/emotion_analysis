# Standard library imports
import os
import argparse

# Third-party imports
import torch
from transformers import AutoTokenizer

# Local application imports - import directly from src package
from src import config
from src.model import EmotionClassifier
from src.preprocess import clean_text


def predict_emotion(text, model, tokenizer, device, max_len):
    """
    Predicts the emotion for a single piece of text.
    """
    model.eval() # Set model to evaluation mode

    # Clean and tokenize the input text
    cleaned_text = clean_text(text)
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Move tensors to the configured device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get model prediction without calculating gradients
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs are logits from our custom model
        # If using AutoModelForSequenceClassification, might be outputs.logits

    # Get the predicted class index (highest logit)
    pred_index = torch.argmax(outputs, dim=1).item()

    # Convert index back to emotion string
    predicted_emotion = config.ID_TO_EMOTION.get(pred_index, "Unknown")

    # Optional: Calculate probabilities using Softmax
    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    predicted_probability = probabilities[pred_index]

    return predicted_emotion, predicted_probability

def main():
    parser = argparse.ArgumentParser(description="Predict emotion from text using a trained model.")
    parser.add_argument("--text", type=str, required=True, help="Text input for emotion prediction.")
    parser.add_argument("--model_path", type=str, default=config.BEST_MODEL_PATH, help="Path to the trained model file (.bin).")
    # Optional: Add arguments to override config settings like model name if needed
    # parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="Name of the transformer model used for tokenization.")
    args = parser.parse_args()

    # --- Load Tokenizer and Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first using 'python -m src.train' (if running from root) or 'python train.py' (if running from src).")
        return

    print(f"Using device: {config.DEVICE}")
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading model structure: {config.MODEL_NAME} with {config.NUM_CLASSES} classes")
    # --- Instantiate the correct model class ---
    model = EmotionClassifier(n_classes=config.NUM_CLASSES)
    # Or if using the alternative factory function:
    # model = create_preset_model(config.NUM_CLASSES) # Needs appropriate import
    # ------------------------------------------
    model.to(config.DEVICE) # Move model structure to device before loading state_dict

    try:
        print(f"Loading model weights from: {args.model_path}")
        # Load the state dict, ensuring map_location handles CPU/GPU differences
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(config.DEVICE)))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in model.py matches the saved weights.")
        return

    # --- Predict ---
    try:
        predicted_emotion, probability = predict_emotion(
            args.text,
            model,
            tokenizer,
            config.DEVICE,
            config.MAX_LEN
        )

        print("\n--- Prediction ---")
        print(f"Input Text: \"{args.text}\"")
        print(f"Predicted Emotion: {predicted_emotion}")
        print(f"Confidence: {probability:.4f}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()