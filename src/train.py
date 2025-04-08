import pandas as pd
import torch
import numpy as np
import os
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

# --- Use standard relative imports ---
import config
import dataset
import model as model_module
from preprocess import clean_text
from engine import train_fn, eval_fn
# -----------------------------------

def run():
    """
    Main function to run the training process.
    """
    # Set seed for reproducibility early
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    # Load data
    df = pd.read_csv(config.DATA_PATH)
    print(f"Original dataset shape: {df.shape}")
    # Handle potential NaNs more explicitly if needed (e.g., inspect rows)
    df = df.dropna(subset=['content', 'sentiment']).reset_index(drop=True)
    print(f"Shape after dropping NaNs in 'content' or 'sentiment': {df.shape}")

    # --- Apply text cleaning *before* splitting ---
    print("Applying text cleaning...")
    df['content'] = df['content'].apply(clean_text)
    print("Text cleaning finished.")
    # ---------------------------------------------

    # Map labels to IDs
    # Check if all sentiments are in our defined map
    unknown_sentiments = df[~df['sentiment'].isin(config.EMOTION_TO_ID.keys())]['sentiment'].unique()
    if len(unknown_sentiments) > 0:
        print(f"Warning: Found sentiments in data not in EMOTION_TO_ID mapping: {unknown_sentiments}")
        # Option: Filter out these rows or add them to the mapping
        df = df[df['sentiment'].isin(config.EMOTION_TO_ID.keys())]
        print(f"Shape after removing unknown sentiments: {df.shape}")

    df['label'] = df['sentiment'].map(config.EMOTION_TO_ID)

    print(f"Value counts per emotion label:\n{df['label'].value_counts(normalize=True)}") # Use normalize=True for proportions

    # Split data
    df_train, df_valid = train_test_split(
        df,
        test_size=0.1, # Consider making this a config parameter
        random_state=config.SEED,
        stratify=df['label'].values # Stratify based on the numeric label
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_valid)}")

    # Initialize tokenizer
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Create datasets
    # --- Pass the dataframe directly, not texts/labels ---
    train_dataset = dataset.EmotionDataset(
        dataframe=df_train,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        include_labels=True # Explicitly state we need labels processed
    )
    valid_dataset = dataset.EmotionDataset(
        dataframe=df_valid,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        include_labels=True # Explicitly state we need labels processed
    )
    # -----------------------------------------------------

    # Create dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=0, # Keep 0 for compatibility, increase for performance if possible
        shuffle=True,
        pin_memory=True if config.DEVICE == "cuda" else False # Slight speedup on GPU
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0, # Keep 0 for compatibility
        shuffle=False,
        pin_memory=True if config.DEVICE == "cuda" else False # Slight speedup on GPU
    )

    # Initialize model
    device = torch.device(config.DEVICE)
    print(f"Initializing model: {config.MODEL_NAME} with {config.NUM_CLASSES} classes")
    # --- Instantiate the actual model from model_module ---
    model = model_module.EmotionClassifier(n_classes=config.NUM_CLASSES)
    # Or if using the alternative factory function:
    # model = model_module.create_preset_model(config.NUM_CLASSES)
    # --------------------------------------------------
    model.to(device)

    # Optimizer and Scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.01}, # Standard weight decay for transformers
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    # --- Add warmup steps (e.g., 5-10% of total steps is common) ---
    num_warmup_steps = int(num_train_steps * 0.06) # Example: 6% warmup
    print(f"Total training steps: {num_train_steps}, Warmup steps: {num_warmup_steps}")
    # -----------------------------------------------------------

    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )

    # Training loop
    best_accuracy = 0
    print(f"--- Starting Training on {device} for {config.EPOCHS} Epochs ---")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")
        valid_loss, valid_accuracy, valid_report = eval_fn(valid_data_loader, model, device) # report is now a dict

        # Save best model based on validation accuracy
        if valid_accuracy > best_accuracy:
            print(f"Validation accuracy improved ({best_accuracy:.4f} --> {valid_accuracy:.4f}). Saving model...")
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            best_accuracy = valid_accuracy
            # Optionally save the classification report of the best epoch
            # with open(os.path.join(config.OUTPUT_DIR, "best_model_report.txt"), "w") as f:
            #     f.write(classification_report(..., output_dict=False)) # Re-run or save from eval_fn

    print(f"\n--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")

if __name__ == "__main__":
    # Consider adding basic try-except block for catching CUDA errors etc.
    try:
        run()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Optionally re-raise or exit
        raise e