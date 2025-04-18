import torch
import os
import config
import models
import data_handler
from data_handler import load_label_map

def load_vocab(vocab_path):
    vocab, _ = data_handler.Vocabulary.load(vocab_path)
    return vocab

class Predictor:
    def __init__(self, model, vocab, label_map, device):
        self.model = model
        self.vocab = vocab
        self.label_map = label_map
        self.device = device
        self.text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)

    def preprocess(self, text):
        tokens = self.text_preprocessor.clean_and_tokenize(text)
        seq = [config.SOS_IDX] + self.vocab.numericalize(tokens)[:config.MAX_LENGTH] + [config.EOS_IDX]
        return torch.tensor(seq, dtype=torch.long).unsqueeze(0)

    def predict(self, text):
        self.model.eval()
        seq = self.preprocess(text).to(self.device)
        with torch.no_grad():
            logits = self.model(seq)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            if self.label_map is not None:
                pred_label = self.label_map.get(pred_idx, str(pred_idx))
            else:
                pred_label = pred_idx
            return pred_label, probs.squeeze().cpu().numpy()

def load_artifacts():
    device = config.DEVICE
    label_map = load_label_map(config.LABEL_MAP_SAVE_PATH)
    vocab = load_vocab(config.VOCAB_SAVE_PATH)
    n_class = len(label_map) if label_map is not None else None
    if n_class is None:
        _, n_class = data_handler.Vocabulary.load(config.VOCAB_SAVE_PATH)
    model = models.LSTMNetwork(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_class=n_class,
        n_layers=config.N_LAYERS,
        pad_idx=config.PAD_IDX
    )
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device)['model_state_dict'])
    model.to(device)
    return Predictor(model, vocab, label_map, device)

def main():
    print("Loading model and artifacts...")
    predictor = load_artifacts()
    print("Ready to predict emotions!\n")

    examples = [
        "I am so happy to see you!",
        "I feel terrible and alone.",
        "This is absolutely amazing!"
    ]
    print("--- Hardcoded Example Predictions ---")
    for text in examples:
        label, probs = predictor.predict(text)
        print(f"Text: {text}\nPredicted Emotion: {label}\n")

    print("--- Interactive Prediction (type 'q' or 'quit' to exit) ---")
    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() in {'q', 'quit'}:
            print("Exiting.")
            break
        label, probs = predictor.predict(user_input)
        print(f"Predicted Emotion: {label}\n")

if __name__ == "__main__":
    main()