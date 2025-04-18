import torch
import os
import json
import argparse
import sys
from operator import itemgetter
try:
    import config
    import data_handler
    import engine
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure config.py, data_handler.py, and engine.py are accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)
def load_run_config(model_type_dir):
    config_path = os.path.join(model_type_dir, config.RUN_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file not found at {config_path}. Using global config.py defaults.")
        class RunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
        return RunConfig()
    try:
        with open(config_path, 'r') as f: loaded_config = json.load(f)
        class RunConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                if not hasattr(self, 'TRANSFORMER_MODEL_NAME'):
                     print("Warning: Loaded config missing TRANSFORMER_MODEL_NAME. Using global default.")
                     self.TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
                if not hasattr(self, 'PREPROCESSOR_TYPE'):
                     self.PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
        print(f"Loaded run configuration from {config_path}")
        loaded_type = loaded_config.get('MODEL_TYPE')
        if loaded_type != os.path.basename(model_type_dir):
             print(f"Warning: Loaded config MODEL_TYPE ('{loaded_type}') mismatches directory ('{os.path.basename(model_type_dir)}').")
        if loaded_type != 'Transformer':
             print(f"Warning: Loaded config specifies MODEL_TYPE '{loaded_type}', but this app expects 'Transformer'. Proceeding with caution.")
        return RunConfig(**loaded_config)
    except Exception as e:
        print(f"Error loading run config from {config_path}: {e}. Using global defaults.")
        class RunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
        return RunConfig()
def load_prediction_artifacts(model_type_dir):
    print(f"\nLoading artifacts from directory: {model_type_dir}")
    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory not found at {model_type_dir}")
        return None, None, None, None, None
    run_cfg = load_run_config(model_type_dir)
    label_to_int, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_PATH)
    if not int_to_label:
        print("Warning: Label map not found or empty. Predictions will show integer labels.")
        int_to_label = {}
    n_classes = len(int_to_label) if int_to_label else 0
    if n_classes == 0:
        print("Warning: Cannot determine number of classes from label map. Trying to infer...")
        try:
             from transformers import AutoConfig as HfAutoConfig
             model_hf_config = HfAutoConfig.from_pretrained(run_cfg.TRANSFORMER_MODEL_NAME)
             n_classes = model_hf_config.num_labels
             print(f"Inferred n_classes={n_classes} from Transformer config.")
             if n_classes <= 1: raise ValueError("Inferred <= 1 class.")
        except Exception as infer_e:
              print(f"Error: Failed to determine n_classes from label map or model config ({infer_e}). Cannot load model.")
              return None, None, None, None, None
    try:
        print(f"Loading tokenizer: {run_cfg.TRANSFORMER_MODEL_NAME}")
        tokenizer = data_handler.AutoTokenizer.from_pretrained(run_cfg.TRANSFORMER_MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer '{run_cfg.TRANSFORMER_MODEL_NAME}': {e}")
        return None, None, None, None, None
    model_path = os.path.join(model_type_dir, "model", config.BEST_MODEL_FILENAME)
    try:
        model = engine.load_trained_model(model_path, 'Transformer', n_classes)
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {model_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading trained model: {e}")
        return None, None, None, None, None
    print(f"Initializing preprocessor: {run_cfg.PREPROCESSOR_TYPE}")
    if run_cfg.PREPROCESSOR_TYPE != 'basic':
         print(f"Warning: Run config specified preprocessor '{run_cfg.PREPROCESSOR_TYPE}', but using 'basic'.")
    preprocessor = data_handler.BasicTextCleaner()
    return model, tokenizer, preprocessor, int_to_label, run_cfg
class EmotionPredictor:
    def __init__(self, model, tokenizer, preprocessor, int_to_label, run_config):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.int_to_label = int_to_label if int_to_label else {}
        self.run_config = run_config
        self.device = config.DEVICE
        self.model.to(self.device)
        self.model.eval()
        print("\nEmotionPredictor initialized.")
    def _preprocess_input(self, text):
        cleaned_text = self.preprocessor.clean(text)
        return cleaned_text
    def predict(self, text):
        processed_input_text = self._preprocess_input(text)
        try:
            with torch.no_grad():
                encoding = self.tokenizer.encode_plus(
                    processed_input_text, add_special_tokens=True,
                    max_length=self.run_config.MAX_LEN, padding='max_length',
                    truncation=True, return_attention_mask=True, return_tensors='pt',
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=1).squeeze()
            probabilities_np = probabilities.cpu().numpy()
            results = []
            num_outputs = logits.shape[1]
            for i in range(num_outputs):
                prob = probabilities_np[i] if i < len(probabilities_np) else 0.0
                label_name = self.int_to_label.get(i, f"Label_{i}")
                results.append({'label': label_name, 'score': float(prob)})
            results.sort(key=itemgetter('score'), reverse=True)
            return results
        except Exception as e:
            print(f"\nError during prediction: {e}")
            import traceback; traceback.print_exc()
            return None
def run_demo_evaluation(predictor):
    print("\n===================================")
    print("=== Running Built-in Examples ===")
    print("===================================")
    demo_examples = [
        "I am feeling incredibly happy and excited about the party tonight!",
        "This movie is making me feel really sad and thoughtful.",
        "I'm absolutely furious that my flight was cancelled again!",
        "Wow, I did not expect that plot twist at all!",
        "Walking alone late at night makes me feel quite anxious.",
        "I just love the way the sun sets over the ocean.",
        "He seemed quite indifferent to the news.",
        "This complex puzzle is incredibly frustrating!",
        "I feel so calm and peaceful listening to this music.",
        "The project deadline is approaching very quickly."
    ]
    for i, text in enumerate(demo_examples):
        print(f"\nExample {i+1}/{len(demo_examples)}: '{text}'")
        results = predictor.predict(text)
        if results:
            top_result = results[0]
            print(f"  --> Predicted: {top_result['label']} (Score: {top_result['score']:.4f})")
        else:
            print("  --> Prediction failed for this example.")
    print("\n===================================")
    print("=== Built-in Examples Finished ===")
    print("===================================")
def run_interactive_app(predictor):
    print("\n--- Interactive Emotion Prediction ---")
    print(f"Using model: {predictor.run_config.TRANSFORMER_MODEL_NAME}")
    print("Enter text to classify, or type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if not user_input: continue
            if user_input.lower() in ['quit', 'exit']: print("Exiting."); break
            prediction_results = predictor.predict(user_input)
            if prediction_results:
                print("\nPrediction Results:")
                max_score = prediction_results[0]['score'] if prediction_results else 0
                for result in prediction_results:
                    indicator = " *" if result['score'] == max_score and max_score > 0 else ""
                    print(f"  - {result['label']}: {result['score']:.4f}{indicator}")
            else: print("  Prediction failed.")
        except (EOFError, KeyboardInterrupt): print("\nExiting."); break
        except Exception as e: print(f"An unexpected error occurred in the loop: {e}")
def main():
    parser = argparse.ArgumentParser(description="Interactive Emotion Prediction App (Transformer)")
    args = parser.parse_args()
    model_type_dir = config.MODEL_TYPE_ARTIFACTS_DIR
    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory for Transformer not found at {model_type_dir}")
        print("Please ensure the model has been trained first using 'python main.py'.")
        sys.exit(1)
    print(f"Loading Transformer artifacts from: {model_type_dir}")
    model, tokenizer, preprocessor, int_to_label, run_cfg = load_prediction_artifacts(model_type_dir)
    if model is None or tokenizer is None:
        print("Failed to load necessary artifacts (model/tokenizer). Exiting.")
        sys.exit(1)
    predictor = EmotionPredictor(model, tokenizer, preprocessor, int_to_label, run_cfg)
    run_demo_evaluation(predictor)
    run_interactive_app(predictor)
if __name__ == "__main__":
    main()