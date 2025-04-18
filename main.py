import torch
import torch.nn as nn
import torch.optim as optim
import os
import nltk
from torch.utils.data import DataLoader
import sys
import json
import numpy as np

import config
import data_handler
from data_handler import load_label_map
import models
import engine

PAD_IDX = config.PAD_IDX

def check_nltk_resource(resource_id, resource_name):
    try:
        nltk.data.find(f'corpora/{resource_id}')
        print(f"NLTK resource '{resource_name}' already downloaded.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' not found. Downloading...")
        try:
            nltk.download(resource_id)
        except Exception as e:
            print(f"Failed to download NLTK resource '{resource_name}': {e}")
            if resource_id == 'wordnet':
                 print("Warning: WordNet download failed. Lemmatization might be affected.")
            else:
                 print(f"Continuing without '{resource_name}'...")

def run_training():
    print("--- Starting Emotion Classification Training ---")
    print(f"Using device: {config.DEVICE}")
    print(f"Selected model type: {config.MODEL_TYPE}")

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    check_nltk_resource('stopwords', 'stopwords')
    check_nltk_resource('wordnet', 'wordnet')

    int_to_label_map = None
    if os.path.exists(config.LABEL_MAP_SAVE_PATH):
        try:
            int_to_label_map = load_label_map(config.LABEL_MAP_SAVE_PATH)
            print("Loaded existing label map for analysis.")
        except Exception as e:
            print(f"Warning: Could not load label map at {config.LABEL_MAP_SAVE_PATH}: {e}")
            int_to_label_map = None

    try:
        print("Loading data and handling labels...")
        train_df, val_df, test_df, _, loaded_int_to_label, n_class = data_handler.load_and_prepare_data(
            config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH, config.LABEL_MAP_SAVE_PATH
        )
        int_to_label_map = loaded_int_to_label
        print(f"Number of classes determined: {n_class}")
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}. Please check paths in config.py")
        sys.exit(1)
    except TypeError as e:
         print(f"Error: Problem with label types: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Data Analysis ---")
    if int_to_label_map:
        train_df_display = train_df.copy()
        train_df_display['label_str'] = train_df_display['label'].map(int_to_label_map)
        data_handler.plot_class_distribution(
            train_df_display,
            label_column='label_str',
            title="Training Set Class Distribution",
            save_path=os.path.join(config.ARTIFACTS_DIR, "train_class_distribution.png")
        )
    else:
         data_handler.plot_class_distribution(
            train_df,
            label_column='label',
            title="Training Set Class Distribution (Integer Labels)",
            save_path=os.path.join(config.ARTIFACTS_DIR, "train_class_distribution.png")
        )

    print("\nInitializing TextPreprocessor...")
    text_preprocessor = data_handler.TextPreprocessor(use_stopwords=True)

    print("Preprocessing training data (for sequence length analysis)...")
    train_tokens_list = text_preprocessor.preprocess_dataframe(train_df)

    data_handler.plot_sequence_lengths(
        train_tokens_list,
        title="Training Sequence Length Distribution (Before Padding)",
        save_path=os.path.join(config.ARTIFACTS_DIR, "train_seq_length_distribution.png")
    )

    print("\nInitializing and building Vocabulary...")
    vocabulary = data_handler.Vocabulary(freq_threshold=config.MIN_FREQ)
    vocabulary.build_vocabulary(train_tokens_list)
    vocabulary.save(config.VOCAB_SAVE_PATH, n_class=n_class)
    vocab_size = len(vocabulary)
    print(f"Vocabulary size: {vocab_size}")

    print("Numericalizing datasets...")
    def numericalize_tokens(tokens_list, vocab, max_len):
        numericalized = []
        for tokens in tokens_list:
             truncated_tokens = tokens[:max_len-2]
             seq = [config.SOS_IDX] + vocab.numericalize(truncated_tokens) + [config.EOS_IDX]
             numericalized.append(seq)
        return numericalized

    train_sequences = numericalize_tokens(train_tokens_list, vocabulary, config.MAX_LENGTH)

    print("Preprocessing and numericalizing validation data...")
    val_tokens_list = text_preprocessor.preprocess_dataframe(val_df)
    val_sequences = numericalize_tokens(val_tokens_list, vocabulary, config.MAX_LENGTH)

    print("Preprocessing and numericalizing test data...")
    test_tokens_list = text_preprocessor.preprocess_dataframe(test_df)
    test_sequences = numericalize_tokens(test_tokens_list, vocabulary, config.MAX_LENGTH)

    train_labels = train_df['label'].to_numpy(dtype=np.int64)
    val_labels = val_df['label'].to_numpy(dtype=np.int64)
    test_labels = test_df['label'].to_numpy(dtype=np.int64)

    print("Creating PyTorch Datasets...")
    train_dataset = data_handler.EmotionDataset(train_sequences, train_labels)
    val_dataset = data_handler.EmotionDataset(val_sequences, val_labels)
    test_dataset = data_handler.EmotionDataset(test_sequences, test_labels)

    print("Creating DataLoaders...")
    pin_memory = config.DEVICE == "cuda"
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_DATA, collate_fn=data_handler.collate_batch,
        num_workers=config.NUM_WORKERS, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=data_handler.collate_batch,
        num_workers=config.NUM_WORKERS, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=data_handler.collate_batch,
        num_workers=config.NUM_WORKERS, pin_memory=pin_memory
    )

    print(f"Building model: {config.MODEL_TYPE}")
    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=n_class,
            n_layers=config.N_LAYERS,
            pad_idx=PAD_IDX,
            dropout_prob=config.DROPOUT_PROB
        )
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_LSTM, weight_decay=config.WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()

    print(f"Model:\n{model}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")

    print("\nStarting training...")
    trained_model, history_df = engine.trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.EPOCHS,
        device=config.DEVICE,
        val_loader=val_loader,
        model_save_path=config.MODEL_SAVE_PATH
    )

    print("\nPlotting training history...")
    engine.plot_history(history_df, config.RESULTS_PLOT_PATH)

    print("\nEvaluating final model on test set...")
    if int_to_label_map is None:
         print("Warning: Label map not found. Reports will use integer labels.")
         max_label = test_df['label'].max()
         int_to_label_map = {i: str(i) for i in range(max_label + 1)}

    engine.generate_test_report(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=config.DEVICE,
        int_to_label_map=int_to_label_map,
        report_save_path=os.path.join(config.ARTIFACTS_DIR, "test_classification_report.txt"),
        conf_matrix_save_path=os.path.join(config.ARTIFACTS_DIR, "test_confusion_matrix.png")
    )

    print("\n--- Training Pipeline Finished ---")

if __name__ == "__main__":
    if not hasattr(config, 'NUM_WORKERS'):
        config.NUM_WORKERS = 0
        print("Setting config.NUM_WORKERS to default 0")
    if not hasattr(config, 'DROPOUT_PROB'):
        config.DROPOUT_PROB = 0.5
        print("Setting config.DROPOUT_PROB to default 0.5")
    if not hasattr(config, 'WEIGHT_DECAY'):
        config.WEIGHT_DECAY = 0.0
        print("Setting config.WEIGHT_DECAY to default 0.0")

    run_training()