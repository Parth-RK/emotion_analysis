import sys
import os
import argparse
try:
    import config
    import train
except ImportError as e:
     print(f"Error importing core modules: {e}")
     print("Ensure config.py and train.py are accessible.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)
def main():
    parser = argparse.ArgumentParser(
        description="Emotion Classification Framework - Training Pipeline (Transformer Focus)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--transformer_name', type=str, default=None, help='Override config.TRANSFORMER_MODEL_NAME')
    parser.add_argument('--epochs', type=int, default=None, help='Override config.EPOCHS')
    parser.add_argument('--lr', type=float, default=None, help='Override config.LEARNING_RATE')
    parser.add_argument('--batch_size', type=int, default=None, help='Override config.TRAIN_BATCH_SIZE')
    parser.add_argument('--max_len', type=int, default=None, help='Override config.MAX_LEN')
    args = parser.parse_args()
    config_overridden = False
    if args.transformer_name and args.transformer_name != config.TRANSFORMER_MODEL_NAME:
         print(f"Overriding config.TRANSFORMER_MODEL_NAME: '{config.TRANSFORMER_MODEL_NAME}' -> '{args.transformer_name}'")
         config.TRANSFORMER_MODEL_NAME = args.transformer_name
         config_overridden = True
    if args.epochs and args.epochs != config.EPOCHS:
         print(f"Overriding config.EPOCHS: {config.EPOCHS} -> {args.epochs}")
         config.EPOCHS = args.epochs
         config_overridden = True
    if args.lr and args.lr != config.LEARNING_RATE:
         print(f"Overriding config.LEARNING_RATE: {config.LEARNING_RATE} -> {args.lr}")
         config.LEARNING_RATE = args.lr
         config_overridden = True
    if args.batch_size and args.batch_size != config.TRAIN_BATCH_SIZE:
         print(f"Overriding config.TRAIN_BATCH_SIZE: {config.TRAIN_BATCH_SIZE} -> {args.batch_size}")
         config.TRAIN_BATCH_SIZE = args.batch_size
         config_overridden = True
    if args.max_len and args.max_len != config.MAX_LEN:
        print(f"Overriding config.MAX_LEN: {config.MAX_LEN} -> {args.max_len}")
        config.MAX_LEN = args.max_len
        config_overridden = True
    print("=============================================")
    print("=== Emotion Classification (Transformer) ===")
    print("=============================================")
    print(f"Using Transformer Model: {config.TRANSFORMER_MODEL_NAME}")
    print(f"Artifacts will be saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")
    try:
        os.makedirs(config.MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        if os.path.dirname(config.LABEL_MAP_PATH):
             os.makedirs(os.path.dirname(config.LABEL_MAP_PATH), exist_ok=True)
    except OSError as e:
        print(f"Error creating artifact directories: {e}")
        sys.exit(1)
    try:
        config.save_run_config(filepath=config.RUN_CONFIG_PATH)
        print(f"Current run configuration saved to {config.RUN_CONFIG_PATH}")
    except Exception as e:
        print(f"Warning: Failed to save run configuration. {e}")
    try:
        train.run_training_pipeline()
        print("\n--- Training Pipeline Completed Successfully ---")
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n--- File Not Found Error ---")
        print(f"Error: {e}")
        print("Please check data file paths in config.py (e.g., TRAIN_FILE_PATH).")
        sys.exit(1)
    except ImportError as e:
        print(f"\n--- Import Error During Pipeline ---")
        print(f"Error: {e}")
        print("Ensure required libraries (transformers, torch, sklearn, pandas) are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- An Unhandled Error Occurred During Training ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("----------------------------------------------------")
        import traceback; traceback.print_exc()
        print("----------------------------------------------------")
        print("Training failed.")
        sys.exit(1)
if __name__ == "__main__":
    main()