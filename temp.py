import torch
import os
import json
import sys
import traceback
import gradio as gr
try:
    import config
    import data_handler
    import engine
    from transformers import AutoTokenizer as HfAutoTokenizer
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure config.py, data_handler.py, engine.py, and Hugging Face transformers are accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)

def load_run_config(model_type_dir):
    config_path = os.path.join(model_type_dir, config.RUN_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file not found at {config_path}. Using global config.py defaults.")
        class DefaultRunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = getattr(config, 'PREPROCESSOR_TYPE', 'basic')
             TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
        return DefaultRunConfig()

    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        class LoadedRunConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                if not hasattr(self, 'TRANSFORMER_MODEL_NAME') or not self.TRANSFORMER_MODEL_NAME:
                     print("Warning: Loaded config missing TRANSFORMER_MODEL_NAME or value is empty. Using global default.")
                     self.TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
                if not hasattr(self, 'PREPROCESSOR_TYPE') or not self.PREPROCESSOR_TYPE:
                     print("Warning: Loaded config missing PREPROCESSOR_TYPE or value is empty. Using global default 'basic'.")
                     self.PREPROCESSOR_TYPE = 'basic'

        print(f"Loaded run configuration from {config_path}")
        loaded_type = loaded_config.get('MODEL_TYPE')
        if loaded_type and loaded_type != os.path.basename(model_type_dir):
             print(f"Warning: Loaded config MODEL_TYPE ('{loaded_type}') mismatches directory ('{os.path.basename(model_type_dir)}').")
        if loaded_type and loaded_type != 'Transformer':
             print(f"Warning: Loaded config specifies MODEL_TYPE '{loaded_type}', but this app expects 'Transformer'. Proceeding with caution.")

        return LoadedRunConfig(**loaded_config)

    except Exception as e:
        print(f"Error loading run config from {config_path}: {e}. Using global defaults.")
        class DefaultRunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = getattr(config, 'PREPROCESSOR_TYPE', 'basic')
             TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
        return DefaultRunConfig()

def load_prediction_artifacts(model_type_dir):
    print(f"\nLoading artifacts from directory: {model_type_dir}")
    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory not found at {model_type_dir}")
        return None, None, None, None, None
    run_cfg = load_run_config(model_type_dir)
    label_to_int, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_PATH)
    if not int_to_label:
        print(f"Warning: Label map not found or empty at {config.LABEL_MAP_PATH}. Predictions will show integer labels or may fail.")
        int_to_label = {}
    n_classes = len(int_to_label) if int_to_label else 0
    if n_classes == 0:
        print("Warning: Cannot determine number of classes from label map. Trying to infer from Transformer config...")
        try:
             from transformers import AutoConfig as HfAutoConfig
             model_name_for_config = getattr(run_cfg, 'TRANSFORMER_MODEL_NAME', config.TRANSFORMER_MODEL_NAME)
             if not model_name_for_config:
                  raise ValueError("TRANSFORMER_MODEL_NAME is not specified in config or loaded config.")

             model_hf_config = HfAutoConfig.from_pretrained(model_name_for_config)
             n_classes = getattr(model_hf_config, 'num_labels', 0)
             if n_classes <= 1:
                 raise ValueError("Inferred <= 1 class from Transformer config num_labels.")

             print(f"Inferred n_classes={n_classes} from Transformer config.")
        except Exception as infer_e:
              print(f"Error: Failed to determine n_classes from label map or model config ({infer_e}). Cannot load model.")
              return None, None, None, None, None

    try:
        print(f"Loading tokenizer: {run_cfg.TRANSFORMER_MODEL_NAME}")
        tokenizer = HfAutoTokenizer.from_pretrained(run_cfg.TRANSFORMER_MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer '{run_cfg.TRANSFORMER_MODEL_NAME}': {e}")
        return None, None, None, None, None

    model_path = os.path.join(model_type_dir, "model", config.BEST_MODEL_FILENAME)
    try:
        print(f"Loading model from {model_path}")
        model = engine.load_trained_model(model_path, run_cfg.MODEL_TYPE, n_classes)
        if model is None:
             raise ValueError("engine.load_trained_model returned None.")
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {model_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading trained model: {e}")
        traceback.print_exc()
        return None, None, None, None, None

    print(f"Initializing preprocessor...")
    preprocessor = data_handler.BasicTextCleaner()
    print(f"Using preprocessor: BasicTextCleaner")


    print("Artifacts loaded successfully.")
    return model, tokenizer, preprocessor, int_to_label, run_cfg

class EmotionPredictor:
    def __init__(self, model, tokenizer, preprocessor, int_to_label, run_config):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.int_to_label = int_to_label if int_to_label else {}
        self.run_config = run_config
        self.device = getattr(config, 'DEVICE', 'cpu')
        try:
            self.model.to(self.device)
            print(f"Model moved to device: {self.device}")
        except Exception as e:
            print(f"Warning: Could not move model to {self.device}: {e}. Using CPU.")
            self.device = 'cpu'
            self.model.to(self.device)

        self.model.eval()
        print("\nEmotionPredictor initialized.")
        print(f"Model MAX_LEN: {self.run_config.MAX_LEN}")
        print(f"Number of classes inferred/loaded: {len(self.int_to_label) if self.int_to_label else 'N/A (using inferred n_classes)'}")


    def _preprocess_input(self, text):
        cleaned_text = self.preprocessor.clean(text)
        return cleaned_text

    def predict(self, text):
        if not text or not isinstance(text, str):
            return {}

        processed_input_text = self._preprocess_input(text)

        try:
            with torch.no_grad():
                encoding = self.tokenizer.encode_plus(
                    processed_input_text,
                    add_special_tokens=True,
                    max_length=self.run_config.MAX_LEN,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                model_input = {'input_ids': input_ids, 'attention_mask': attention_mask}

                logits = self.model(**model_input)
                if isinstance(logits, tuple):
                    logits = logits[0]

            probabilities = torch.softmax(logits, dim=1).squeeze()
            probabilities_np = probabilities.cpu().numpy()

            results_dict = {}
            num_outputs = logits.shape[1]
            
            if not self.int_to_label:
                 print("Warning: Using integer labels as label map is empty.")
                 for i in range(num_outputs):
                     prob = probabilities_np[i] if i < len(probabilities_np) else 0.0
                     results_dict[f"Label_{i}"] = float(prob)
            else:
                 if num_outputs != len(self.int_to_label):
                      print(f"Warning: Model output size ({num_outputs}) mismatches label map size ({len(self.int_to_label)}). Mapping might be incorrect.")
                      use_int_labels_fallback = True
                      if num_outputs <= len(self.int_to_label):
                           print("Mapping available labels up to model output size.")
                           for i in range(num_outputs):
                               prob = probabilities_np[i] if i < len(probabilities_np) else 0.0
                               label_name = self.int_to_label.get(i, f"Label_{i} (unmapped)")
                               results_dict[label_name] = float(prob)
                           use_int_labels_fallback = False

                      if use_int_labels_fallback:
                           print("Falling back to integer labels due to mapping mismatch.")
                           for i in range(num_outputs):
                               prob = probabilities_np[i] if i < len(probabilities_np) else 0.0
                               results_dict[f"Label_{i}"] = float(prob)

                 else:
                    for i in range(num_outputs):
                         prob = probabilities_np[i] if i < len(probabilities_np) else 0.0
                         label_name = self.int_to_label.get(i, f"Label_{i} (missing_map)")
                         results_dict[label_name] = float(prob)

            return results_dict

        except Exception as e:
            print(f"\nError during prediction: {e}")
            traceback.print_exc()
            return {"Prediction Error": 0.0}


predictor = None
app_load_error = None

model_type_dir = config.MODEL_TYPE_ARTIFACTS_DIR

if not os.path.isdir(model_type_dir):
    app_load_error = f"Error: Artifact directory for Transformer not found at {model_type_dir}. Please ensure the model has been trained first using 'python main.py'."
    print(app_load_error)
else:
    print(f"Attempting to load Transformer artifacts from: {model_type_dir}")
    model, tokenizer, preprocessor, int_to_label, run_cfg = load_prediction_artifacts(model_type_dir)

    if model is None or tokenizer is None or preprocessor is None or run_cfg is None:
        app_load_error = "Failed to load necessary artifacts (model, tokenizer, preprocessor, or config). Cannot start application."
        print(app_load_error)
    else:
        try:
            predictor = EmotionPredictor(model, tokenizer, preprocessor, int_to_label, run_cfg)
            print("Predictor successfully initialized.")
        except Exception as e:
            app_load_error = f"Error initializing EmotionPredictor: {e}"
            print(app_load_error)
            traceback.print_exc()


emotion_emojis = {
    "anger": "\U0001F620",
    "fear": "\U0001F628",
    "sadness": "\U0001F622",
    "surprise": "\U0001F632",
    "love": "\U0001F60D",
    "joy": "\U0001F60A",
    "indifference": "\U0001F610"
}
default_emoji = "\u2753"

def get_emotion_emoji(label):
    if label is None or not isinstance(label, str):
        return default_emoji
    cleaned_label = label.strip().lower()
    if cleaned_label.startswith("label_") or "(unmapped)" in cleaned_label or "(missing_map)" in cleaned_label:
        return default_emoji
    return emotion_emojis.get(cleaned_label, default_emoji)


def format_prediction_html(prediction_results_dict):
    if not prediction_results_dict:
        return "<p>No prediction results.</p>"
    sorted_results = dict(sorted(prediction_results_dict.items(), key=lambda item: item[1], reverse=True))
    top_emotion_label = next(iter(sorted_results), None)
    top_emotion_emoji = get_emotion_emoji(top_emotion_label)
    html_output = f'<div style="text-align: center; font-size: 5em; margin-bottom: 5px;">{top_emotion_emoji}</div>'
    if top_emotion_label and not top_emotion_label.startswith("Label_"):
        html_output += f'<div style="text-align: center; font-size: 1.2em; font-weight: bold; margin-bottom: 15px;">{top_emotion_label.capitalize()}</div>'
    else:
        html_output += '<div style="margin-bottom: 15px;"></div>'
    html_output += '<div style="text-align: left; max-height: 200px; overflow-y: auto;">'
    for label, score in sorted_results.items():
        percentage = int(score * 100)
        html_output += f'<p style="margin: 5px 0; font-size: 1em;"><b>{label.capitalize()}</b>: {percentage}%</p>'
    html_output += '</div>'
    return html_output


def gradio_predict_emotion(text):
    if app_load_error:
        return f'<div style="color: red; text-align: center; font-size: 1.2em;">\U0001F6AB Application Load Error: {app_load_error}</div>'
    if predictor is None:
        return '<div style="color: red; text-align: center; font-size: 1.2em;">\U0001F6AB Application Not Initialized.</div>'
    prediction_results_dict = predictor.predict(text)
    if not prediction_results_dict or list(prediction_results_dict.keys())[0] in [
        "No input text", "Preprocessing resulted in empty text", "Prediction resulted in zeros or empty", "Prediction Error"]:
        error_key = list(prediction_results_dict.keys())[0] if prediction_results_dict else "Unknown Error"
        details = prediction_results_dict.get("Details", "") if isinstance(prediction_results_dict, dict) else ""
        error_message = f"\u2757 Prediction Issue: {error_key}"
        if details:
            error_message += f" - {details}"
        return f'<div style="color: red; text-align: center; font-size: 1.2em;">{error_message}</div>'
    return format_prediction_html(prediction_results_dict)

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

if app_load_error:
    print("Gradio app starting in error state due to loading failure.")
    interface = gr.Interface(
        fn=lambda text: f'<div style="color: red; text-align: center; font-size: 1.2em;">\U0001F6AB Application Load Error: {app_load_error}</div>',
        inputs=gr.Textbox(label="Enter Text (App Failed to Load)"),
        outputs=gr.HTML(label="Error Details"),
        title="Emotion Classifier (Loading Error)",
        description=app_load_error,
        allow_flagging="never",
        theme=gr.themes.Soft()
    )
else:
    print("Gradio app starting with loaded artifacts.")
    model_name_desc = getattr(predictor.run_config, 'TRANSFORMER_MODEL_NAME', 'Transformer Model')
    description_text = f"Enter text below to get predicted emotion scores using the model: {model_name_desc}"
    output_html = gr.HTML(
        label="Predicted Emotion Results",
        value=""
    )
    interface = gr.Interface(
        fn=gradio_predict_emotion,
        inputs=gr.Textbox(
            lines=3,
            label="Enter Text",
            placeholder="Type your sentence here...",
            show_label=True
        ),
        outputs=output_html,
        title="Text Emotion Analyzer",
        description=description_text,
        examples=demo_examples,
        cache_examples=False,
        allow_flagging="never",
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    launch_port = getattr(config, 'APP_PORT', 7860)
    try:
        print(f"Launching Gradio interface on port {launch_port}...")
        interface.launch(server_port=launch_port, share=True)
    except OSError as e:
        print(f"Error: Port {launch_port} already in use or blocked.")
        print(f"Please try a different port, e.g., interface.launch(server_port=XXXX).")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during Gradio launch: {e}")
        traceback.print_exc()
        sys.exit(1)