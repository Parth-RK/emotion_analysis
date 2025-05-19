import sys
import os
import argparse

# Add project root to sys.path if necessary for imports to work
# assuming main.py is at the root and modules are in subdirectories or root
# For this structure (all files at root), standard import should work.

try:
    import config
    import train # train.py orchestrates the pipeline
    # Other modules (data_handler, engine, models, plotter) are imported within train.py
except ImportError as e:
     print(f"Error importing core modules: {e}")
     print("Ensure config.py and train.py are accessible.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)


def main():
    """Main entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification Framework - Training Pipeline (Transformer Focus)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # Add command-line arguments to override config values
    parser.add_argument('--transformer_name', type=str, default=None, help='Override config.TRANSFORMER_MODEL_NAME')
    parser.add_argument('--epochs', type=int, default=None, help='Override config.EPOCHS')
    parser.add_argument('--lr', type=float, default=None, help='Override config.LEARNING_RATE')
    parser.add_argument('--batch_size', type=int, default=None, help='Override config.TRAIN_BATCH_SIZE')
    parser.add_argument('--max_len', type=int, default=None, help='Override config.MAX_LEN')
    # Add other potentially useful overrides
    parser.add_argument('--val_batch_size', type=int, default=None, help='Override config.VALID_BATCH_SIZE')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override config.WEIGHT_DECAY')
    parser.add_argument('--grad_clip', type=float, default=None, help='Override config.GRADIENT_CLIP_VALUE')
    parser.add_argument('--metric_best', type=str, default=None, help='Override config.METRIC_FOR_BEST_MODEL')


    args = parser.parse_args()

    # Apply overrides to config
    print("--- Applying Configuration Overrides ---")
    config_overridden = False # Flag to indicate if any override occurred

    if args.transformer_name is not None and args.transformer_name != config.TRANSFORMER_MODEL_NAME:
         print(f"Overriding config.TRANSFORMER_MODEL_NAME: '{config.TRANSFORMER_MODEL_NAME}' -> '{args.transformer_name}'")
         config.TRANSFORMER_MODEL_NAME = args.transformer_name
         config_overridden = True

    if args.epochs is not None and args.epochs != config.EPOCHS:
         print(f"Overriding config.EPOCHS: {config.EPOCHS} -> {args.epochs}")
         config.EPOCHS = args.epochs
         config_overridden = True

    if args.lr is not None and args.lr != config.LEARNING_RATE:
         print(f"Overriding config.LEARNING_RATE: {config.LEARNING_RATE} -> {args.lr}")
         config.LEARNING_RATE = args.lr
         config_overridden = True

    if args.batch_size is not None and args.batch_size != config.TRAIN_BATCH_SIZE:
         print(f"Overriding config.TRAIN_BATCH_SIZE: {config.TRAIN_BATCH_SIZE} -> {args.batch_size}")
         config.TRAIN_BATCH_SIZE = args.batch_size
         config_overridden = True

    if args.max_len is not None and args.max_len != config.MAX_LEN:
        print(f"Overriding config.MAX_LEN: {config.MAX_LEN} -> {args.max_len}")
        config.MAX_LEN = args.max_len
        config_overridden = True

    if args.val_batch_size is not None and args.val_batch_size != config.VALID_BATCH_SIZE:
         print(f"Overriding config.VALID_BATCH_SIZE: {config.VALID_BATCH_SIZE} -> {args.val_batch_size}")
         config.VALID_BATCH_SIZE = args.val_batch_size
         config_overridden = True

    if args.weight_decay is not None and args.weight_decay != config.WEIGHT_DECAY:
         print(f"Overriding config.WEIGHT_DECAY: {config.WEIGHT_DECAY} -> {args.weight_decay}")
         config.WEIGHT_DECAY = args.weight_decay
         config_overridden = True

    if args.grad_clip is not None: # Can override with 0 or None
         if args.grad_clip != getattr(config, 'GRADIENT_CLIP_VALUE', None):
             print(f"Overriding config.GRADIENT_CLIP_VALUE: {getattr(config, 'GRADIENT_CLIP_VALUE', None)} -> {args.grad_clip}")
             config.GRADIENT_CLIP_VALUE = args.grad_clip
             config_overridden = True

    if args.metric_best is not None and args.metric_best != config.METRIC_FOR_BEST_MODEL:
         print(f"Overriding config.METRIC_FOR_BEST_MODEL: '{config.METRIC_FOR_BEST_MODEL}' -> '{args.metric_best}'")
         valid_metrics = ['loss', 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
         if args.metric_best not in valid_metrics:
              print(f"Warning: Invalid metric '{args.metric_best}' for --metric_best. Must be one of {valid_metrics}. Using default: '{config.METRIC_FOR_BEST_MODEL}'.")
         else:
             config.METRIC_FOR_BEST_MODEL = args.metric_best
             config_overridden = True


    if not config_overridden:
        print("No config overrides from command line.")
    print("----------------------------------------")


    print("=============================================")
    print("=== Emotion Classification (Transformer) ===")
    print("=============================================")
    print(f"Using Transformer Model: {config.TRANSFORMER_MODEL_NAME}")
    print(f"Artifacts will be saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")

    # Ensure necessary directories exist before proceeding
    try:
        os.makedirs(config.MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        # The label map path might be outside the model type dir
        if os.path.dirname(config.LABEL_MAP_PATH):
             os.makedirs(os.path.dirname(config.LABEL_MAP_PATH), exist_ok=True)
        # Also ensure data directory exists (user should put data there)
        os.makedirs(config.DATA_DIR, exist_ok=True)

    except OSError as e:
        print(f"Error creating artifact or data directories: {e}")
        sys.exit(1)

    # Save the configuration used for this run for reproducibility
    try:
        config.save_run_config(filepath=config.RUN_CONFIG_PATH)
        print(f"\nCurrent run configuration saved to {config.RUN_CONFIG_PATH}")
    except Exception as e:
        print(f"Warning: Failed to save run configuration to {config.RUN_CONFIG_PATH}. {e}")

    # --- Run the main training pipeline ---
    try:
        train.run_training_pipeline()
        print("\n--- Training Pipeline Completed Successfully ---")
        sys.exit(0) # Exit successfully
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
        sys.exit(0) # Exit gracefully on interruption
    except FileNotFoundError as e:
        print(f"\n--- File Not Found Error ---")
        print(f"Error: {e}")
        print("Please check data file paths (e.g., config.TRAIN_FILE_PATH) and label map path (config.LABEL_MAP_PATH).")
        sys.exit(1) # Exit with error code
    except ValueError as e:
        print(f"\n--- Configuration or Data Error ---")
        print(f"Error: {e}")
        print("Please check your config settings (columns, format, splits) or data content.")
        sys.exit(1) # Exit with error code
    except ImportError as e:
        print(f"\n--- Import Error During Pipeline ---")
        print(f"Error: {e}")
        print("Ensure required libraries (transformers, torch, sklearn, pandas, emoji) are installed.")
        sys.exit(1) # Exit with error code
    except Exception as e:
        print(f"\n--- An Unhandled Error Occurred During Training ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("----------------------------------------------------")
        import traceback; traceback.print_exc() # Print full traceback for debugging
        print("----------------------------------------------------")
        print("Training failed.")
        sys.exit(1) # Exit with error code


if __name__ == "__main__":
    main()