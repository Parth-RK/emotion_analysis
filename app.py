import torch
import config
import models
import data_handler
from data_handler import load_label_map, Vocabulary

def load_vocab(vocab_path):
    try:
        vocab, n_class = Vocabulary.load(vocab_path)
        if n_class is None:
             raise ValueError("n_class not found in vocabulary file.")
        return vocab, n_class
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}")
        raise
    except ValueError as e:
        print(f"Error loading vocabulary: {e}")
        raise

class Predictor:
    def __init__(self, model, vocab, label_map, device):
        self.model = model
        self.vocab = vocab
        self.label_map = label_map
        self.device = device
        self.text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)
        self.model.eval()

    def preprocess(self, text):
        if not isinstance(text, str):
            text = str(text)
        tokens = self.text_preprocessor.clean_and_tokenize(text)
        max_content_len = config.MAX_LENGTH - 2
        if len(tokens) > max_content_len:
             tokens = tokens[:max_content_len]
        seq = [config.SOS_IDX] + self.vocab.numericalize(tokens) + [config.EOS_IDX]
        return torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict(self, text):
        processed_input = self.preprocess(text)
        with torch.no_grad():
            logits = self.model(processed_input)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            if self.label_map is not None:
                pred_label = self.label_map.get(pred_idx, f"Unknown Index: {pred_idx}")
            else:
                pred_label = str(pred_idx)
            return pred_label, probs.squeeze().cpu().numpy()

def load_artifacts():
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Loading vocabulary from: {config.VOCAB_SAVE_PATH}")
    vocab, n_class = load_vocab(config.VOCAB_SAVE_PATH)
    print(f"Loading label map from: {config.LABEL_MAP_PATH}")
    label_map = load_label_map(config.LABEL_MAP_PATH)
    print(f"Loading model state dict from: {config.MODEL_SAVE_PATH}")
    model = models.LSTMNetwork(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_class=n_class,
        n_layers=config.N_LAYERS,
        pad_idx=config.PAD_IDX,
        dropout_prob=getattr(config, 'DROPOUT_PROB', 0.0)
    )
    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        if 'model_state_dict' not in checkpoint:
             raise KeyError("Checkpoint file missing 'model_state_dict' key.")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dict loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        raise
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        raise
    model.to(device)
    model.eval()
    print("Creating Predictor...")
    predictor = Predictor(model, vocab, label_map, device)
    print("Artifacts loaded successfully.")
    return predictor

def main():
    print("Loading model and artifacts...")
    try:
        predictor = load_artifacts()
        print("Ready to predict emotions!\n")
    except Exception as e:
        print(f"Fatal Error: Could not load artifacts. Exiting. Error: {e}")
        return
    examples = [
        "I am so happy to see you!",
        "I feel terrible and alone.",
        "This is absolutely amazing!",
        "I am not happy about this situation.",
        "feeling quite sad today"
    ]
    print("--- Hardcoded Example Predictions ---")
    for text in examples:
        try:
            label, probs = predictor.predict(text)
            if predictor.label_map:
                prob_dict = {predictor.label_map.get(i, str(i)): p for i, p in enumerate(probs)}
                sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
                top_probs_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted_probs[:3]])
            else:
                top_probs_str = ", ".join([f"{i}: {p:.3f}" for i, p in enumerate(probs)])
            print(f"Text: '{text}'")
            print(f"Predicted Emotion: {label}")
            print("-" * 10)
        except Exception as e:
            print(f"Error predicting for text '{text}': {e}")
    print("\n--- Interactive Prediction (type 'q' or 'quit' to exit) ---")
    while True:
        try:
            user_input = input("Enter text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {'q', 'quit'}:
                print("Exiting.")
                break
            label, probs = predictor.predict(user_input)
            print(f"Predicted Emotion: {label}\n")
        except EOFError:
             print("\nExiting.")
             break
        except KeyboardInterrupt:
             print("\nExiting.")
             break
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()