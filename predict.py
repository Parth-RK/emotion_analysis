import torch
from transformers import AutoTokenizer
import argparse
import os

# Import local modules - adjust path if running predict.py directly from root
try:
    from src import config
    from src.model import EmotionClassifier
    from src.preprocess import clean_text
except ImportError:
    # Handle case where script is run from the project root directory
    import config
    from model import EmotionClassifier
    from preprocess import clean_text


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
        # outputs are logits

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
    args = parser.parse_args()

    # --- Load Tokenizer and Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first using 'python src/train.py'")
        return

    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading model from: {args.model_path}")
    model = EmotionClassifier(n_classes=config.NUM_CLASSES)
    try:
        # Load the state dict, ensuring map_location handles CPU/GPU differences
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(config.DEVICE)))
        model.to(config.DEVICE)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # --- Predict ---
    predicted_emotion, probability = predict_emotion(
        args.text,
        model,
        tokenizer,
        config.DEVICE,
        config.MAX_LEN
    )

    print("\n--- Prediction ---")
    print(f"Input Text: {args.text}")
    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Confidence: {probability:.4f}")


if __name__ == "__main__":
    main()