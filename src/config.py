import torch
import os

# --- Data Configuration ---
DATA_DIR = "data"
# Use the full dataset path when available
# DATA_PATH = os.path.join(DATA_DIR, "emotion_dataset_full.csv")
DATA_PATH = os.path.join(DATA_DIR, "emotion_data.csv") # Using the sample for now
OUTPUT_DIR = "models"
MODEL_NAME = "distilbert-base-uncased" # Efficient & strong baseline
# MODEL_NAME = "roberta-base" # Potentially higher accuracy, more resource-intensive

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME.replace('/', '_')}_best_model.bin")


# --- Model & Training Configuration ---
MAX_LEN = 128 # Max sequence length for tokenizer
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 4 # Start with a few epochs, increase if needed
LEARNING_RATE = 3e-5 # AdamW default, often good for transformers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42 # For reproducibility

# --- Emotion Mapping ---
# Explicitly define the 13 emotions based on the prompt
EMOTIONS = [
    "anger", "boredom", "empty", "enthusiasm", "fun", "happiness",
    "hate", "love", "neutral", "relief", "sadness", "surprise", "worry"
]
NUM_CLASSES = len(EMOTIONS)

# Create mappings
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
ID_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

print(f"Using device: {DEVICE}")
print(f"Number of emotion classes: {NUM_CLASSES}")
print(f"Model path: {BEST_MODEL_PATH}")