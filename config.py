# --- config.py ---
import torch
import os
import json
from pathlib import Path
import inspect # For saving config
import sys
# --- Core Settings ---
SEED = 42
# Use CUDA if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Selection ---
# Hardcoding to Transformer as it's the only supported model now.
MODEL_TYPE = 'Transformer'

# --- Data Configuration ---
DATA_DIR = "data/goemotions" # Directory where data files are stored
# Specify your GoEmotions file paths here
TRAIN_FILE = "train.tsv" # Example TSV train file name
TRAIN_FILE_PATH = os.path.join(DATA_DIR, TRAIN_FILE)
VALID_FILE = "dev.tsv" # Example TSV validation file name
VALID_FILE_PATH = os.path.join(DATA_DIR, VALID_FILE) if VALID_FILE else None
TEST_FILE = "test.tsv" # Example TSV test file name
TEST_FILE_PATH = os.path.join(DATA_DIR, TEST_FILE) if TEST_FILE else None

# Configuration for loading the data file(s)
INPUT_FILE_FORMAT = "tsv" # Options: "csv", "tsv", "jsonl"
TEXT_COLUMN_INDEX = 0 # Index of the column containing text
# For TSV/CSV with comma-separated labels, this is the index of that column
LABEL_COLUMN_INDEX = 1 # Index of the column containing labels
# Column names if HAS_HEADER is False. Used by pandas read_csv/tsv.
# 'labels_str' is a temporary name used in data_handler before splitting the string.
COLUMN_NAMES = ['text', 'labels_str']
# Set to True if the first row of your TSV/CSV is a header, False otherwise.
HAS_HEADER = False # Based on common GoEmotions TSV format

# --- Data Splitting (Used if VALID_FILE_PATH or TEST_FILE_PATH is None/not found) ---
# Note: Multi-label stratification is NOT implemented in data_handler's train_test_split
# Simple random split will be used even if STRATIFY_SPLIT is True.
VALIDATION_SPLIT_SIZE = 0.15
TEST_SPLIT_SIZE = 0.15
STRATIFY_SPLIT = True # Note: Stratification is NOT supported for multi-label splits here.

# --- Artifacts & Output ---
ARTIFACTS_DIR = "artifacts" # Base directory for all model artifacts
MODEL_TYPE_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, MODEL_TYPE) # Model-specific artifacts dir
MODEL_SAVE_DIR = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, "model") # Directory to save model checkpoints
BEST_MODEL_FILENAME = "best_model.pt" # Filename for the best model checkpoint
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME) # Full path to the best model file

# Global label map path (should be consistent across runs for a specific dataset)
# This JSON file should contain the mapping for all 28 GoEmotions labels.
LABEL_MAP_FILENAME = "label_map.json" # Standard filename for label map
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, LABEL_MAP_FILENAME) # Path to the GoEmotions label map

# Model-specific output files (inside model type dir)
TRAINING_PLOTS_FILENAME = "training_plots.png"
TRAINING_PLOTS_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, TRAINING_PLOTS_FILENAME)
TEST_REPORT_FILENAME = "test_report.txt"
TEST_REPORT_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, TEST_REPORT_FILENAME)
CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png" # NOTE: CM plot is skipped for multi-label
CONFUSION_MATRIX_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, CONFUSION_MATRIX_FILENAME)
RUN_CONFIG_FILENAME = "run_config.json" # File to save the config used for a run
RUN_CONFIG_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, RUN_CONFIG_FILENAME)

# --- Preprocessing ---
# The BasicTextCleaner now includes enhanced emoji handling.
PREPROCESSOR_TYPE = 'basic'

# --- Transformer Model Specific ---
# Choose your desired transformer model from Hugging Face
# TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"

# --- Training Configuration ---
MAX_LEN = 128 # Maximum token length for input sequences
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
OPTIMIZER_TYPE = 'AdamW' # AdamW is standard for Transformers
SCHEDULER_TYPE = 'linear_warmup' # 'linear_warmup', 'reduce_on_plateau', or None
WARMUP_PROPORTION = 0.1 # Proportion of training steps for linear warmup
GRADIENT_CLIP_VALUE = 1.0 # Clip gradient norms to prevent exploding gradients (None to disable)

# --- Evaluation & Plotting ---
PLOT_TRAINING_HISTORY = True # Generate plot of training/validation metrics over epochs
GENERATE_TEST_REPORT = True # Generate classification report on the test set
GENERATE_CONFUSION_MATRIX = True # Generate confusion matrix plot (NOTE: Skipped for multi-label)
# Metric to monitor on the validation set for saving the best model.
# For multi-label, 'f1_weighted' is a common choice. 'loss' is also valid.
METRIC_FOR_BEST_MODEL = 'f1_weighted' # Options: 'loss', 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'
PREDICTION_THRESHOLD = 0.5 # Threshold for converting sigmoid probabilities to binary predictions during evaluation/inference

# --- Inference App Configuration ---
APP_PORT = 7860 # Port for the Gradio web application

# --- Logging & Config Saving ---
def save_run_config(filepath=RUN_CONFIG_PATH):
    """Saves the current configuration values to a JSON file for the run."""
    # Get all uppercase variables from the current module
    config_vars = {}
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if name.isupper() and not name.startswith("__") and not inspect.ismodule(obj) and \
           not inspect.isfunction(obj) and not inspect.isclass(obj):
            # Attempt to serialize common types, handle Paths specifically
            try:
                if isinstance(obj, Path):
                    config_vars[name] = str(obj)
                elif isinstance(obj, (list, tuple)):
                     # Try to convert elements within lists/tuples
                     config_vars[name] = [str(i) if isinstance(i, Path) else i for i in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                     config_vars[name] = obj
                elif isinstance(obj, dict):
                     # Recursively handle dictionaries (simple conversion)
                     config_vars[name] = {k: (str(v) if isinstance(v, Path) else v) for k, v in obj.items()}
                else:
                     # Fallback for other types, might fail but try converting to string
                     config_vars[name] = str(obj)
                     # print(f"Warning: Config variable '{name}' has type {type(obj)}. Saved as string.") # Optional debug

            except Exception as e:
                print(f"Warning: Could not serialize config variable '{name}' (type {type(obj)}): {e}. Skipping or saving as str().")
                try: config_vars[name] = str(obj)
                except: pass # Give up if str() fails too


    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_vars, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save run configuration to {filepath}. Error: {e}")

# --- Print Summary ---
# Print key configuration settings when the module is loaded
print(f"--- Configuration Loaded ---")
print(f"Selected Model Type: {MODEL_TYPE}")
print(f"Artifacts Directory: {MODEL_TYPE_ARTIFACTS_DIR}")
print(f"Device: {DEVICE}")
print(f"Training Data: {TRAIN_FILE_PATH}")
print(f"Validation Data: {'Provided (' + str(VALID_FILE_PATH) + ')' if VALID_FILE_PATH else 'Splitting from Train'}")
print(f"Test Data: {'Provided (' + str(TEST_FILE_PATH) + ')' if TEST_FILE_PATH else 'Splitting from Train'}")
print(f"Data File Format: {INPUT_FILE_FORMAT}")
print(f"Text Column Index: {TEXT_COLUMN_INDEX}, Label Column Index: {LABEL_COLUMN_INDEX}")
print(f"Has Header: {HAS_HEADER}")
print(f"Preprocessor: {PREPROCESSOR_TYPE} (with emoji handling)")
print(f"Transformer Model: {TRANSFORMER_MODEL_NAME}")
print(f"Max Sequence Length: {MAX_LEN}")
print(f"Train Batch Size: {TRAIN_BATCH_SIZE}, Valid Batch Size: {VALID_BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Optimizer: {OPTIMIZER_TYPE}, Scheduler: {SCHEDULER_TYPE}")
print(f"Metric for Best Model: {METRIC_FOR_BEST_MODEL}")
print(f"Prediction Threshold: {PREDICTION_THRESHOLD}")
print("---------------------------")

# Ensure artifact directories exist on module load
os.makedirs(MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True) # For global label map
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data dir exists as well
