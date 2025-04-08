import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

# Import project modules
from . import config
from . import dataset
from . import model as model_module
from .preprocess import clean_text
from .engine import train_fn, eval_fn

def run():
    """
    Main function to run the training process.
    """
    # Load data
    df = pd.read_csv(config.DATA_PATH).dropna().reset_index(drop=True)
    print(f"Dataset shape: {df.shape}")
    print("Applying text cleaning...")
    df['content'] = df['content'].apply(clean_text)
    df['label'] = df['sentiment'].map(config.EMOTION_TO_ID)
    print("Text cleaning finished.")
    print(f"Value counts per emotion:\n{df['label'].value_counts()}")

    # Split data
    df_train, df_valid = train_test_split(
        df,
        test_size=0.1,
        random_state=config.SEED,
        stratify=df['label'].values
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_valid)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Create datasets
    train_dataset = dataset.EmotionDataset(
        texts=df_train.content.values,
        labels=df_train.label.values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    valid_dataset = dataset.EmotionDataset(
        texts=df_valid.content.values,
        labels=df_valid.label.values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    # Create dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=0,
        shuffle=True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0,
        shuffle=False
    )

    # Initialize model
    device = torch.device(config.DEVICE)
    model = model_module.EmotionClassifier(n_classes=config.NUM_CLASSES)
    model.to(device)

    # Optimizer and Scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # Training loop
    best_accuracy = 0
    print(f"--- Starting Training for {config.EPOCHS} Epochs ---")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid_loss, valid_accuracy, valid_report = eval_fn(valid_data_loader, model, device)

        # Save best model based on validation accuracy
        if valid_accuracy > best_accuracy:
            print(f"Validation accuracy improved ({best_accuracy:.4f} --> {valid_accuracy:.4f}). Saving model...")
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            best_accuracy = valid_accuracy

    print(f"\n--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")

if __name__ == "__main__":
    run()
