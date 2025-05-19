import torch
import os
import json
import sys
import traceback
import gradio as gr

# Add project root to sys.path if necessary
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
    import data_handler # Need data_handler for loading label maps and cleaner
    import engine # Need engine for loading the trained model
    # Specific transformer imports for loading tokenizer/config
    from transformers import AutoTokenizer as HfAutoTokenizer
    from transformers import AutoConfig as HfAutoConfig # Need AutoConfig to potentially infer n_classes
    import numpy as np # For handling numpy arrays from predictions
    import torch.nn.functional as F # For softmax or sigmoid

except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure config.py, data_handler.py, engine.py, and Hugging Face transformers are accessible.")
    print("Have you run 'pip install -r requirements.txt'?")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_run_config(model_type_dir):
    """Loads the saved run configuration or provides defaults."""
    config_path = os.path.join(model_type_dir, config.RUN_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file not found at {config_path}. Using global config.py defaults.")
        # Return an object that behaves like the config object for essential keys
        class DefaultRunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = getattr(config, 'PREPROCESSOR_TYPE', 'basic')
             TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
             PREDICTION_THRESHOLD = getattr(config, 'PREDICTION_THRESHOLD', 0.5) # Added threshold for app
             # Add other relevant config items used by the app if needed, e.g., APP_PORT

             def __getattr__(self, name):
                 # Allows accessing attributes that weren't explicitly set,
                 # falling back to the original config module if they exist there.
                 # Be cautious with this - could mask missing attributes.
                 if hasattr(config, name):
                     return getattr(config, name)
                 raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


        return DefaultRunConfig()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)

        class LoadedRunConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                # Provide fallbacks for essential keys if missing in the file
                if not hasattr(self, 'TRANSFORMER_MODEL_NAME') or not self.TRANSFORMER_MODEL_NAME:
                     print("Warning: Loaded config missing TRANSFORMER_MODEL_NAME or value is empty. Using global default.")
                     self.TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
                if not hasattr(self, 'PREPROCESSOR_TYPE') or not self.PREPROCESSOR_TYPE:
                     print("Warning: Loaded config missing PREPROCESSOR_TYPE or value is empty. Using global default 'basic'.")
                     self.PREPROCESSOR_TYPE = 'basic'
                if not hasattr(self, 'MAX_LEN'):
                    print("Warning: Loaded config missing MAX_LEN. Using global default.")
                    self.MAX_LEN = config.MAX_LEN
                if not hasattr(self, 'MODEL_TYPE'):
                    print("Warning: Loaded config missing MODEL_TYPE. Using global default.")
                    self.MODEL_TYPE = config.MODEL_TYPE
                if not hasattr(self, 'PREDICTION_THRESHOLD'): # Load threshold used during training/eval if saved
                    print("Warning: Loaded config missing PREDICTION_THRESHOLD. Using global default 0.5.")
                    self.PREDICTION_THRESHOLD = 0.5
                # Ensure other potentially used configs from the original config are available
                if not hasattr(self, 'APP_PORT'): self.APP_PORT = config.APP_PORT
                # Add other config items from original config here if needed by the app


        print(f"Loaded run configuration from {config_path}")
        # Optional: Check if loaded model type matches the directory name
        loaded_type = loaded_config.get('MODEL_TYPE')
        if loaded_type and loaded_type != os.path.basename(model_type_dir):
             print(f"Warning: Loaded config MODEL_TYPE ('{loaded_type}') mismatches directory ('{os.path.basename(model_type_dir)}').")
        # Warn if the model type in config is not Transformer (this app is Transformer-specific)
        if loaded_type and loaded_type != 'Transformer':
             print(f"Warning: Loaded config specifies MODEL_TYPE '{loaded_type}', but this app expects 'Transformer'. Proceeding, but compatibility issues possible.")

        return LoadedRunConfig(**loaded_config)

    except Exception as e:
        print(f"Error loading run config from {config_path}: {e}. Using global defaults.")
        traceback.print_exc() # Print error for debugging config loading
        class DefaultRunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = getattr(config, 'PREPROCESSOR_TYPE', 'basic')
             TRANSFORMER_MODEL_NAME = config.TRANSFORMER_MODEL_NAME
             PREDICTION_THRESHOLD = getattr(config, 'PREDICTION_THRESHOLD', 0.5)
             APP_PORT = config.APP_PORT # Include app port

             def __getattr__(self, name):
                 if hasattr(config, name):
                     return getattr(config, name)
                 raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


        return DefaultRunConfig()


def load_prediction_artifacts(model_type_dir):
    """Loads the trained model, tokenizer, preprocessor, label map, and run config."""
    print(f"\nLoading artifacts from directory: {model_type_dir}")

    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory not found at {model_type_dir}")
        return None, None, None, None, None

    # 1. Load Run Configuration
    run_cfg = load_run_config(model_type_dir)
    model_name = run_cfg.TRANSFORMER_MODEL_NAME
    model_type = run_cfg.MODEL_TYPE # Should be 'Transformer'
    max_len = run_cfg.MAX_LEN
    # Prediction threshold is used in evaluate_step for metrics, but sigmoid outputs are shown in app

    # 2. Load Label Mappings
    # For GoEmotions, the label map should be a fixed 28-class map saved during training
    label_map_path = getattr(run_cfg, 'LABEL_MAP_PATH', config.LABEL_MAP_PATH) # Use path from loaded config if available, else global
    label_to_int, int_to_label = data_handler.load_label_mappings(label_map_path)

    if not int_to_label or not label_to_int:
        print(f"CRITICAL ERROR: Label map not found or empty at {label_map_path}. Cannot determine classes and label names.")
        # Attempt to infer n_classes from model config as a last resort, but map is needed for names
        n_classes = 0
        try:
            print("Attempting to infer n_classes from Transformer config...")
            # Use model_name from loaded config
            model_hf_config = HfAutoConfig.from_pretrained(model_name)
            # Check standard attributes for num_labels
            inferred_n_classes = getattr(model_hf_config, 'num_labels', None)

            if inferred_n_classes is None or inferred_n_classes <= 1:
                 # Some base configs might not have num_labels set if no head was used
                 # Try getting the size of the dummy head if we were training from this config
                 # This is a heuristic and might fail if the model name points to a non-standard config
                 print(f"  'num_labels' not found or <= 1 in base config ({inferred_n_classes}). Trying heuristic...")
                 try:
                     # Temporarily initialize a model just to get the classifier output size
                     # Need to use a placeholder n_classes first, then get the *actual* size if successful
                     temp_model_for_size = engine.initialize_model(model_type, 100) # Arbitrary number >= expected
                     inferred_n_classes = temp_model_for_size.classifier.out_features # Get output size of the classifier head
                     # Clean up the temporary model to free memory
                     del temp_model_for_size
                     torch.cuda.empty_cache() # Clear CUDA cache if using GPU

                 except Exception as dummy_err:
                     print(f"  Heuristic failed: {dummy_err}. Cannot infer n_classes.")
                     inferred_n_classes = 0 # Set to 0 if heuristic fails


            if inferred_n_classes is not None and inferred_n_classes > 1:
                 n_classes = inferred_n_classes
                 print(f"  Inferred n_classes={n_classes} from Transformer config or model structure.")
                 # If map is missing, create a dummy map for outputting integer labels
                 int_to_label = {i: f"Label_{i}" for i in range(n_classes)}
                 label_to_int = {v: k for k, v in int_to_label.items()}
                 print("  Using integer labels as text map is missing.")
            else:
                 print("  Failed to infer n_classes from config or model. Cannot load model.")
                 return None, None, None, None, None
        except Exception as infer_e:
              print(f"Error: Failed to determine n_classes from label map or model config ({infer_e}). Cannot load model.")
              traceback.print_exc()
              return None, None, None, None, None
    else:
         n_classes = len(int_to_label)
         print(f"Label mappings loaded successfully. Determined {n_classes} classes.")


    # 3. Load Tokenizer
    try:
        print(f"Loading tokenizer: {model_name}")
        tokenizer = HfAutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer '{model_name}': {e}")
        traceback.print_exc()
        return None, None, None, None, None

    # 4. Load Model Weights
    # Pass the inferred or loaded n_classes to initialize the model architecture correctly
    model_path = os.path.join(model_type_dir, "model", config.BEST_MODEL_FILENAME)
    try:
        print(f"Loading model from {model_path}")
        # engine.load_trained_model initializes the model architecture based on type and n_classes, then loads weights
        model = engine.load_trained_model(model_path, model_type, n_classes)
        if model is None:
             raise ValueError("engine.load_trained_model returned None unexpectedly.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Trained model file not found at {model_path}. Please ensure training completed successfully.")
        return None, None, None, None, None
    except Exception as e:
        print(f"CRITICAL ERROR: Error loading trained model from {model_path}: {e}")
        traceback.print_exc()
        return None, None, None, None, None

    # 5. Initialize Preprocessor
    print(f"Initializing preprocessor: {run_cfg.PREPROCESSOR_TYPE}")
    # We only support 'basic' which now handles emojis
    if run_cfg.PREPROCESSOR_TYPE == 'basic':
         preprocessor = data_handler.BasicTextCleaner()
         print(f"Using preprocessor: BasicTextCleaner (with emoji handling)")
    else:
         print(f"Warning: Unknown preprocessor type '{run_cfg.PREPROCESSOR_TYPE}' from config. Using BasicTextCleaner (with emoji handling).")
         preprocessor = data_handler.BasicTextCleaner()


    print("All artifacts loaded successfully.")
    # Return loaded components and the run_config
    return model, tokenizer, preprocessor, int_to_label, run_cfg


class EmotionPredictor:
    """Handles prediction using the loaded model and artifacts."""
    def __init__(self, model, tokenizer, preprocessor, int_to_label, run_config):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.int_to_label = int_to_label if int_to_label else {} # Ensure it's a dict
        # Create a sorted list of label names based on integer keys
        # This ensures the order matches the model's output layer which is 0..n-1
        self.label_names = [self.int_to_label.get(i, f"Label_{i}") for i in sorted(self.int_to_label.keys())]
        self.run_config = run_config
        # Use device from config or loaded config if available
        self.device = getattr(run_config, 'DEVICE', getattr(config, 'DEVICE', 'cpu'))
        self.max_len = getattr(run_config, 'MAX_LEN', config.MAX_LEN) # Get MAX_LEN from loaded config
        # Prediction threshold is NOT used here, we output probabilities

        # Move model to device and set to eval mode
        try:
            self.model.to(self.device)
            print(f"Model moved to device: {self.device}")
        except Exception as e:
            print(f"Warning: Could not move model to {self.device}: {e}. Falling back to CPU.")
            self.device = 'cpu'
            self.model.to(self.device) # Ensure it's on CPU if CUDA failed

        self.model.eval() # Set model to evaluation mode
        print("\nEmotionPredictor initialized.")
        print(f"Model MAX_LEN: {self.max_len}")
        print(f"Number of classes: {len(self.int_to_label) if self.int_to_label else 'N/A (map missing)'}")
        # No prediction threshold printed here as we output probabilities


    def _preprocess_input(self, text):
        """Applies the configured text cleaning."""
        # Use the preprocessor instance loaded during artifact loading
        return self.preprocessor.clean(text)


    def predict(self, text):
        """
        Predicts emotion probabilities for a given text.

        Args:
            text (str): Input text string.

        Returns:
            dict: A dictionary mapping emotion label names to predicted probabilities (float).
                  Returns empty dict or error message if prediction fails.
        """
        if not text or not isinstance(text, str):
            return {"Input Error": 1.0} # Return error for invalid input

        # 1. Preprocess text
        processed_input_text = self._preprocess_input(text)
        # print(f"Processed Text: '{processed_input_text}'") # Optional debug

        # 2. Tokenize and prepare input tensors
        try:
            with torch.no_grad(): # No gradients needed for inference
                encoding = self.tokenizer.encode_plus(
                    processed_input_text,
                    add_special_tokens=True, # Add CLS and SEP tokens
                    max_length=self.max_len,
                    padding='max_length', # Pad to max_len
                    truncation=True, # Truncate if longer than max_len
                    return_attention_mask=True,
                    return_tensors='pt', # Return PyTorch tensors
                )

                # Move tensors to the correct device and squeeze batch dimension
                # Unsqueeze to add batch dimension = 1
                input_ids = encoding['input_ids'].to(self.device).unsqueeze(0)
                attention_mask = encoding['attention_mask'].to(self.device).unsqueeze(0)

                model_input = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # 3. Get model output (logits)
                logits = self.model(**model_input)
                # Some models might return a tuple, the first element is typically the logits
                if isinstance(logits, tuple):
                    logits = logits[0]

            # 4. Apply Sigmoid to get probabilities for multi-label
            # Sigmoid is applied per-neuron, independently.
            probabilities = torch.sigmoid(logits).squeeze(0) # Remove batch dim after sigmoid


            # 5. Convert probabilities to a dictionary
            probabilities_np = probabilities.cpu().numpy() # Move to CPU and convert to numpy array

            results_dict = {}
            num_outputs = probabilities_np.shape[-1] # Number of outputs from the model

            # Check if the number of model outputs matches the loaded label map size
            if self.int_to_label and num_outputs != len(self.int_to_label):
                 print(f"Warning: Model output size ({num_outputs}) mismatches loaded label map size ({len(self.int_to_label)}). Mapping might be incorrect.")
                 print("Falling back to integer labels for output dictionary keys.")
                 # Create dummy map and label names based on model output size
                 self.int_to_label = {i: f"Label_{i}" for i in range(num_outputs)}
                 self.label_names = [f"Label_{i}" for i in range(num_outputs)]


            # Populate results dictionary with label names and probabilities
            if self.int_to_label:
                 # Use sorted keys to match the order expected by the model output layer (0 to n-1)
                 sorted_label_indices = sorted(self.int_to_label.keys())
                 if len(sorted_label_indices) == num_outputs:
                      for i in range(num_outputs):
                          label_id = sorted_label_indices[i]
                          label_name = self.int_to_label.get(label_id, f"Label_{label_id} (missing_map)")
                          prob = probabilities_np[i] # Get probability at this index
                          results_dict[label_name] = float(prob)
                 else:
                      # Fallback if sorted indices don't match output size (shouldn't happen if n_classes was set correctly)
                      print("Warning: Label map size and sorted indices mismatch model output size. Using integer indices.")
                      for i in range(num_outputs):
                          results_dict[f"Label_{i}"] = float(probabilities_np[i])
            else:
                 # Fallback if no int_to_label map was loaded at all
                 print("Warning: No int_to_label map available. Using integer indices for output.")
                 for i in range(num_outputs):
                    results_dict[f"Label_{i}"] = float(probabilities_np[i])

            # Gradio's gr.Label automatically sorts by probability descending when given a dictionary
            return results_dict

        except Exception as e:
            print(f"\nError during prediction: {e}")
            traceback.print_exc()
            # Return a dictionary indicating error
            return {"Prediction Error": 1.0}


# --- Gradio App Setup ---
predictor = None # Global variable to hold the initialized predictor
app_load_error = None # Global variable to store loading errors

# Determine the directory where artifacts for this model type should be
model_type_dir = config.MODEL_TYPE_ARTIFACTS_DIR

# Attempt to load artifacts when the app starts
try:
    print(f"Attempting to load Transformer artifacts from: {model_type_dir}")
    model, tokenizer, preprocessor, int_to_label, run_cfg = load_prediction_artifacts(model_type_dir)

    if model is None or tokenizer is None or preprocessor is None or run_cfg is None or not int_to_label:
        # If any critical artifact failed to load, set an error message
        error_details = []
        if model is None: error_details.append("Model")
        if tokenizer is None: error_details.append("Tokenizer")
        if preprocessor is None: error_details.append("Preprocessor")
        if run_cfg is None: error_details.append("Run Config")
        if not int_to_label: error_details.append("Label Map")
        app_load_error = f"Failed to load necessary artifacts: {', '.join(error_details)}. Cannot start application."
        print(f"CRITICAL ERROR: {app_load_error}")
    else:
        # If artifacts loaded successfully, initialize the predictor
        try:
            predictor = EmotionPredictor(model, tokenizer, preprocessor, int_to_label, run_cfg)
            print("EmotionPredictor successfully initialized.")
        except Exception as e:
            # Error during predictor initialization (e.g., moving model to device)
            app_load_error = f"Error initializing EmotionPredictor: {e}"
            print(f"CRITICAL ERROR: {app_load_error}")
            traceback.print_exc()


except Exception as e:
    # Catch any unexpected errors during the initial loading process
    app_load_error = f"An unexpected error occurred during application startup artifact loading: {e}"
    print(f"CRITICAL ERROR: {app_load_error}")
    traceback.print_exc()


def gradio_predict_emotion(text):
    """Gradio interface function for prediction."""
    # If there was an error during startup loading, return the error
    if app_load_error:
         return {"Application Loading Error": 1.0, "Details": app_load_error}

    # If predictor wasn't successfully initialized, return an error
    if predictor is None:
         return {"Application Not Initialized": 1.0}

    # Call the predictor's predict method
    prediction_results_dict = predictor.predict(text)

    # Handle case where predictor.predict returned None or error dict
    if prediction_results_dict is None:
        return {"Prediction Failed": 1.0}

    # Gradio's gr.Label automatically sorts by probability descending
    return prediction_results_dict


# --- Gradio Interface Definition ---
demo_examples = [
    "I feel happy, excited, and optimistic about this!", # Example with multiple emotions
    "That movie was really sad and disappointing.",
    "I'm so angry and annoyed by this traffic!",
    "Wow, that was surprising! I'm a bit confused.",
    "Walking alone at night makes me feel fearful and nervous.",
    "This food is absolutely disgusting 🤢", # Example with emoji
    "I feel gratitude and admiration for your help 🙏", # Example with multiple emotions and emoji
    "He seemed quite neutral.",
    "This problem is frustrating." # Annoyance/disappointment
]

# If there was an error loading artifacts, show an error message in the UI
if app_load_error:
    print("Gradio app starting in error state due to loading failure.")
    interface = gr.Interface(
        fn=lambda text: {"Error": 1.0, "Details": app_load_error}, # Simple function returning the error
        inputs=gr.Textbox(label="Enter Text (Application Failed to Load)"),
        outputs=gr.Label(label="Loading Error"),
        title="Emotion Classifier (Loading Error)",
        description=f"The application failed to load necessary models and artifacts. Details: {app_load_error}",
        allow_flagging="never",
        theme=gr.themes.Soft(),
        examples=None # Disable examples in error state
    )
else:
    # If artifacts loaded successfully, create the functional interface
    print("Gradio app starting with loaded artifacts.")
    model_name_desc = getattr(predictor.run_config, 'TRANSFORMER_MODEL_NAME', 'Transformer Model')
    num_classes_desc = len(predictor.int_to_label) if predictor.int_to_label else '?'
    description_text = f"Enter text below to get predicted multi-label emotion scores for {num_classes_desc} classes using the model: **{model_name_desc}**"

    interface = gr.Interface(
        fn=gradio_predict_emotion, # The prediction function
        inputs=gr.Textbox(
            lines=3,
            label="Enter Text",
            placeholder="Type your sentence here...",
            show_label=True
        ),
        outputs=gr.Label(
            num_top_classes=None, # Show all classes
            label="Predicted Emotion Scores"
        ),
        title="Text Emotion Classifier (Multi-Label)",
        description=description_text,
        examples=demo_examples, # Use multi-label/emoji examples
        cache_examples=False, # Don't cache predictions for examples
        allow_flagging="never", # Disable flagging
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    # Added argparse to allow specifying port from command line
    # This is useful if default port is in use
    arg_parser = argparse.ArgumentParser(description="Run the Gradio web application.")
    arg_parser.add_argument("--port", type=int, default=getattr(config, 'APP_PORT', 7860), help="Port to run the Gradio server on.")
    app_args = arg_parser.parse_args()
    launch_port = app_args.port


    try:
        print(f"Launching Gradio interface on port {launch_port}...")
        # Use share=True to get a public link (useful for demos)
        # Set debug=True for more detailed Gradio logs during development
        interface.launch(server_port=launch_port, share=True)
    except OSError as e:
        print(f"Error: Port {launch_port} already in use or blocked.")
        print(f"Please try a different port using the --port command line argument, e.g., python app.py --port 8000.")
        print("Or modify config.py APP_PORT.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during Gradio launch: {e}")
        traceback.print_exc()
        sys.exit(1)
