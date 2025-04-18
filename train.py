# --- train.py ---
import torch
import os
import sys
import numpy as np

# Import necessary modules from the project
try:
    import config
    import data_handler
    import engine
    import models # Keep import for clarity, though only TransformerClassifier used
    import plotter
except ImportError as e:
    print(f"Error importing necessary modules in train.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports in train.py: {e}")
    sys.exit(1)

def run_training_pipeline():
    """ Executes the full training and evaluation pipeline for Transformer. """
    print(f"\n--- Starting Training Pipeline for {config.MODEL_TYPE} ---") # MODEL_TYPE is 'Transformer'

    # --- 1. Data Loading and Preparation ---
    try:
        # Adjusted return values from the simplified get_data_pipeline
        train_loader, val_loader, test_loader, \
        label_to_int, int_to_label, n_classes, \
        tokenizer = data_handler.get_data_pipeline() # No vocab_size, tokenizer instead

        if train_loader is None:
            print("Error: Training DataLoader is None. Cannot proceed.")
            sys.exit(1)
        if n_classes <= 1:
             print(f"Error: Only {n_classes} class(es) detected. Need >= 2.")
             sys.exit(1)

        print(f"\nData pipeline finished. Number of classes: {n_classes}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}") # Use tokenizer info

    except FileNotFoundError as e: print(f"\nData Loading Error: {e}"); sys.exit(1)
    except ValueError as e: print(f"\nData Processing Error: {e}"); sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during data preparation: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    # --- 2. Model Initialization ---
    try:
        # Simplified call - no vocab_size needed
        model = engine.initialize_model(
            model_type=config.MODEL_TYPE, # Always 'Transformer'
            n_classes=n_classes
        )
    except ValueError as e: print(f"\nModel Initialization Error: {e}"); sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during model initialization: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    # --- 3. Optimizer and Scheduler Initialization ---
    try:
        num_train_steps = None
        if config.SCHEDULER_TYPE == 'linear_warmup':
            num_train_steps = len(train_loader) * config.EPOCHS
            if num_train_steps <= 0:
                 print("Warning: Calculated num_train_steps <= 0. Scheduler might misbehave.")

        optimizer, scheduler = engine.initialize_optimizer_scheduler(
            model=model,
            optimizer_type=config.OPTIMIZER_TYPE,
            scheduler_type=config.SCHEDULER_TYPE,
            num_train_steps=num_train_steps
        )
    except ValueError as e: print(f"\nOptimizer/Scheduler Error: {e}"); sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during optimizer/scheduler setup: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    # --- 4. Training ---
    try:
        history = engine.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            epochs=config.EPOCHS,
            model_save_path=config.BEST_MODEL_PATH, # Uses fixed path
            metric_for_best=config.METRIC_FOR_BEST_MODEL
        )
    except Exception as e:
        print(f"\nUnexpected error during model training: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)

    # --- 5. Plot Training History ---
    if config.PLOT_TRAINING_HISTORY and history:
        print("\n--- Plotting Training History ---")
        try:
            plotter.plot_training_history(history, save_path=config.TRAINING_PLOTS_PATH)
        except Exception as e: print(f"Warning: Failed to plot training history. Error: {e}")
    elif not history: print("\nSkipping training plot: History unavailable.")

    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    if test_loader is None:
        print("Test DataLoader is None. Skipping final test evaluation.")
    else:
        try:
            print(f"Loading best model from: {config.BEST_MODEL_PATH}")
            # Simplified call - no vocab_size
            best_model = engine.load_trained_model(
                model_path=config.BEST_MODEL_PATH,
                model_type=config.MODEL_TYPE, # Always 'Transformer'
                n_classes=n_classes
            )
            # No need to move to device again, load_trained_model handles it

            test_metrics = engine.evaluate_step(best_model, test_loader, config.DEVICE)

            print("\n--- Test Set Performance (Best Model) ---")
            print(f"  Test Loss:      {test_metrics.get('loss', float('nan')):.4f}")
            print(f"  Test Accuracy:  {test_metrics.get('accuracy', 0.0):.4f}")
            print(f"  Test Precision: {test_metrics.get('precision_weighted', 0.0):.4f} (W)")
            print(f"  Test Recall:    {test_metrics.get('recall_weighted', 0.0):.4f} (W)")
            print(f"  Test F1-Score:  {test_metrics.get('f1_weighted', 0.0):.4f} (W)")
            print("-------------------------------------------")

            # --- 7. Generate Test Report & Confusion Matrix ---
            if config.GENERATE_TEST_REPORT or config.GENERATE_CONFUSION_MATRIX:
                if 'predictions' in test_metrics and 'true_labels' in test_metrics:
                    print("\n--- Generating Test Analysis ---")
                    plotter.generate_classification_analysis(
                        true_labels=test_metrics['true_labels'],
                        predictions=test_metrics['predictions'],
                        int_to_label=int_to_label,
                        report_path=config.TEST_REPORT_PATH if config.GENERATE_TEST_REPORT else None,
                        cm_path=config.CONFUSION_MATRIX_PATH if config.GENERATE_CONFUSION_MATRIX else None,
                        prefix="Test Set"
                    )
                else: print("Warning: Cannot generate test report/CM. Data missing.")

        except FileNotFoundError:
            print(f"Error: Best model not found at {config.BEST_MODEL_PATH}. Cannot run final eval.")
        except Exception as e:
            print(f"\nError during final test evaluation/analysis: {e}")
            import traceback; traceback.print_exc()

    print(f"\n--- Training Pipeline for {config.MODEL_TYPE} Finished ---")

# Optional: Allow direct running if needed, though main.py is the entry point
# if __name__ == "__main__":
#     run_training_pipeline()