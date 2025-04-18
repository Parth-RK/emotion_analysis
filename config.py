# --- config.py ---
import torch
import os
import json
from pathlib import Path

# --- Core Settings ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Selection ---
# Hardcoding to Transformer as it's the only supported model now.
MODEL_TYPE = 'Transformer'

# --- Data Configuration ---
DATA_DIR = "data"
TRAIN_FILE = "training.csv"
TRAIN_FILE_PATH = os.path.join(DATA_DIR, TRAIN_FILE)
VALID_FILE = "validation.csv"
VALID_FILE_PATH = os.path.join(DATA_DIR, VALID_FILE) if VALID_FILE else None
TEST_FILE = "test.csv"
TEST_FILE_PATH = os.path.join(DATA_DIR, TEST_FILE) if TEST_FILE else None

INPUT_FILE_FORMAT = "csv" # Options: "csv", "tsv", "jsonl"
TEXT_COLUMN_INDEX = 0
LABEL_COLUMN_INDEX = 1
COLUMN_NAMES = ['text', 'label']
HAS_HEADER = True

# --- Data Splitting (Used if VALID_FILE_PATH or TEST_FILE_PATH is None) ---
VALIDATION_SPLIT_SIZE = 0.15
TEST_SPLIT_SIZE = 0.15
STRATIFY_SPLIT = True

# --- Artifacts & Output ---
ARTIFACTS_DIR = "artifacts"
MODEL_TYPE_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, MODEL_TYPE)
MODEL_SAVE_DIR = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, "model")
BEST_MODEL_FILENAME = "best_model.pt"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME)

# Global label map path (outside model-specific dirs)
LABEL_MAP_FILENAME = "label_map.json"
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, LABEL_MAP_FILENAME)

# Model-specific output files (inside model type dir)
TRAINING_PLOTS_FILENAME = "training_plots.png"
TRAINING_PLOTS_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, TRAINING_PLOTS_FILENAME)
TEST_REPORT_FILENAME = "test_report.txt"
TEST_REPORT_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, TEST_REPORT_FILENAME)
CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png"
CONFUSION_MATRIX_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, CONFUSION_MATRIX_FILENAME)
RUN_CONFIG_FILENAME = "run_config.json"
RUN_CONFIG_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, RUN_CONFIG_FILENAME)

# Ensure artifact directories exist
os.makedirs(MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True) # For global label map

# --- Preprocessing ---
# Basic cleaner is generally sufficient for Transformers
PREPROCESSOR_TYPE = 'basic'

# --- Transformer Model Specific ---
# TRANSFORMER_MODEL_NAME = "distilbert-base-uncased" # OR
TRANSFORMER_MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased" 


# --- Training Configuration ---
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
OPTIMIZER_TYPE = 'AdamW'
SCHEDULER_TYPE = 'linear_warmup'
WARMUP_PROPORTION = 0.1
GRADIENT_CLIP_VALUE = 1.0

# --- Evaluation & Plotting ---
PLOT_TRAINING_HISTORY = True
GENERATE_TEST_REPORT = True
GENERATE_CONFUSION_MATRIX = True
METRIC_FOR_BEST_MODEL = 'accuracy' # or 'f1_weighted' or 'loss'

# --- Logging & Config Saving ---
def save_run_config(filepath=RUN_CONFIG_PATH):
    """Saves the current configuration values to a JSON file for the run."""
    import json
    import inspect
    config_vars = {}
    current_module = __import__(__name__)
    for name, obj in inspect.getmembers(current_module):
        if name.isupper() and not name.startswith("__") and not inspect.ismodule(obj) and \
           not inspect.isfunction(obj) and not inspect.isclass(obj) and \
           isinstance(obj, (str, int, float, bool, list, tuple, dict, type(None))):
            if isinstance(obj, Path): config_vars[name] = str(obj)
            elif isinstance(obj, (list, tuple)):
                 config_vars[name] = [str(i) if isinstance(i, Path) else i for i in obj]
            else: config_vars[name] = obj
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f: json.dump(config_vars, f, indent=4, default=str)
    except Exception as e:
        print(f"Warning: Could not save run configuration to {filepath}. Error: {e}")

# --- Print Summary ---
print(f"--- Configuration Loaded ---")
print(f"Selected Model Type: {MODEL_TYPE}")
print(f"Artifacts Directory: {MODEL_TYPE_ARTIFACTS_DIR}")
print(f"Device: {DEVICE}")
print(f"Training Data: {TRAIN_FILE_PATH}")
print(f"Validation Data: {'Provided (' + str(VALID_FILE_PATH) + ')' if VALID_FILE_PATH else 'Splitting from Train'}")
print(f"Test Data: {'Provided (' + str(TEST_FILE_PATH) + ')' if TEST_FILE_PATH else 'Splitting from Train'}")
print(f"Preprocessor: {PREPROCESSOR_TYPE}")
print(f"Transformer Model: {TRANSFORMER_MODEL_NAME}")
print("---------------------------")
