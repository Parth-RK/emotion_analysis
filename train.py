# --- train.py ---
import torch
import os
import sys
import numpy as np
import torch.nn as nn # Import nn for criterion

# Import necessary modules from the project
try:
    import config
    import data_handler # Data loading and preprocessing
    import engine # Model initialization, training, evaluation
    import models # Model definition (TransformerClassifier) - keep import for clarity
    import plotter # Plotting and reporting
except ImportError as e:
    print(f"Error importing necessary modules in train.py: {e}")
    print("Please ensure config.py, data_handler.py, engine.py, models.py, and plotter.py are in the same directory or accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports in train.py: {e}")
    import traceback; traceback.print_exc();
    sys.exit(1)


def run_training_pipeline():
    """ Executes the full training and evaluation pipeline for Transformer (Multi-label). """
    print(f"\n--- Starting Training Pipeline for {config.MODEL_TYPE} (Multi-Label) ---") # Indicate multi-label mode

    # --- 1. Data Loading and Preparation ---
    try:
        # get_data_pipeline now returns pos_weight_np
        train_loader, val_loader, test_loader, \
        label_to_int, int_to_label, n_classes, \
        tokenizer, pos_weight_np = data_handler.get_data_pipeline() # Added pos_weight_np

        if train_loader is None:
            print("Error: Training DataLoader is None after data pipeline. Cannot proceed.")
            sys.exit(1)

        if n_classes <= 1:
             print(f"Error: Only {n_classes} class(es) detected based on loaded label map. Need >= 2 for classification.")
             sys.exit(1)

        print(f"\nData pipeline finished successfully.")
        print(f"Number of classes from label map: {n_classes}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}") # Use tokenizer info


    except FileNotFoundError as e:
        print(f"\nCRITICAL Data Loading Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nCRITICAL Data Processing Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected critical error during data preparation: {e}")
        import traceback; traceback.print_exc();
        sys.exit(1)


    # --- 2. Model Initialization ---
    try:
        # Pass the determined number of classes (28 for GoEmotions)
        model = engine.initialize_model(
            model_type=config.MODEL_TYPE, # Always 'Transformer'
            n_classes=n_classes # Use the number of classes from the label map
        )
    except ValueError as e:
        print(f"\nCRITICAL Model Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected critical error during model initialization: {e}")
        import traceback; traceback.print_exc();
        sys.exit(1)


    # --- 3. Optimizer and Scheduler Initialization ---
    try:
        num_train_steps = None
        if config.SCHEDULER_TYPE == 'linear_warmup':
            # Calculate total training steps for the linear warmup scheduler
            if train_loader and len(train_loader) > 0:
                num_train_steps = len(train_loader) * config.EPOCHS
            else:
                 num_train_steps = 0 # No steps if no data
                 print("Warning: Training DataLoader is empty. Num train steps for scheduler is 0.")

            if num_train_steps <= 0:
                 print("Warning: Calculated num_train_steps <= 0. Linear Warmup scheduler might misbehave or be skipped.")


        optimizer, scheduler = engine.initialize_optimizer_scheduler(
            model=model,
            optimizer_type=config.OPTIMIZER_TYPE,
            scheduler_type=config.SCHEDULER_TYPE,
            num_train_steps=num_train_steps
        )

    except ValueError as e:
        print(f"\nCRITICAL Optimizer/Scheduler Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected critical error during optimizer/scheduler setup: {e}")
        import traceback; traceback.print_exc();
        sys.exit(1)

    # --- 4. Initialize Criterion (Loss Function) with pos_weight ---
    # BCEWithLogitsLoss for multi-label
    # Pass the calculated pos_weight to address class imbalance
    try:
        pos_weight_tensor = torch.tensor(pos_weight_np, device=config.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"\nInitialized Loss Function: {type(criterion).__name__} with pos_weight on {config.DEVICE}")
        # print(f"Pos Weight Tensor: {pos_weight_tensor}") # Optional: print tensor
    except Exception as e:
        print(f"\nCRITICAL Error initializing criterion with pos_weight: {e}")
        print("Ensure pos_weight_np is a valid numpy array of floats with size matching n_classes.")
        sys.exit(1)


    # --- 5. Training ---
    try:
        # Pass the initialized criterion to the training engine
        history = engine.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, # Pass validation loader (can be None)
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            epochs=config.EPOCHS,
            model_save_path=config.BEST_MODEL_PATH, # Uses fixed path from config
            criterion=criterion, # Pass the criterion
            metric_for_best=config.METRIC_FOR_BEST_MODEL # Uses metric from config
        )
    except Exception as e:
        print(f"\nUnexpected critical error during model training: {e}")
        import traceback; traceback.print_exc();
        sys.exit(1) # Exit on training failure


    # --- 6. Plot Training History ---
    if config.PLOT_TRAINING_HISTORY and history:
        print("\n--- Plotting Training History ---")
        try:
            plotter.plot_training_history(history, save_path=config.TRAINING_PLOTS_PATH)
        except Exception as e:
            print(f"Warning: Failed to plot training history. Error: {e}")
    elif not history:
        print("\nSkipping training plot: History unavailable.")


    # --- 7. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    if test_loader is None or len(test_loader) == 0:
        print("Test DataLoader is None or empty. Skipping final test evaluation.")
    else:
        try:
            print(f"Loading best model weights from: {config.BEST_MODEL_PATH}")
            # load_trained_model will initialize architecture and load weights
            best_model = engine.load_trained_model(
                model_path=config.BEST_MODEL_PATH,
                model_type=config.MODEL_TYPE, # Always 'Transformer'
                n_classes=n_classes # Use the correct number of classes
            )
            # Model is automatically set to .eval() in load_trained_model

            # Perform evaluation on the test set - pass the criterion
            # Note: pos_weight in criterion does NOT affect evaluation metrics,
            # but the evaluate_step function calculates loss using the criterion for logging.
            test_metrics = engine.evaluate_step(best_model, test_loader, config.DEVICE, criterion)

            print("\n--- Test Set Performance (Best Model) ---")
            # Print multi-label specific metrics
            print(f"  Test Loss:              {test_metrics.get('loss', float('nan')):.4f}")
            print(f"  Test Subset Accuracy:   {test_metrics.get('accuracy', 0.0):.4f}") # Subset accuracy
            print(f"  Test Precision (W):     {test_metrics.get('precision_weighted', 0.0):.4f}")
            print(f"  Test Recall (W):        {test_metrics.get('recall_weighted', 0.0):.4f}")
            print(f"  Test F1-Score (W):      {test_metrics.get('f1_weighted', 0.0):.4f}")
            print("-------------------------------------------")

            # --- 8. Generate Test Report & Confusion Matrix ---
            # generate_classification_analysis now handles multi-label report and skips CM plot
            if (config.GENERATE_TEST_REPORT or config.GENERATE_CONFUSION_MATRIX) and \
               'predictions' in test_metrics and 'true_labels' in test_metrics:

                print("\n--- Generating Test Analysis ---")
                plotter.generate_classification_analysis(
                    true_labels=test_metrics['true_labels'], # These are multi-hot numpy arrays
                    predictions=test_metrics['predictions'], # These are binary prediction numpy arrays
                    int_to_label=int_to_label, # Pass the loaded int_to_label map
                    report_path=config.TEST_REPORT_PATH if config.GENERATE_TEST_REPORT else None,
                    cm_path=config.CONFUSION_MATRIX_PATH if config.GENERATE_CONFUSION_MATRIX else None, # CM plot is skipped internally for multi-label
                    prefix="Test Set"
                )
            else:
                 print("Warning: Cannot generate test report/CM. Metric results missing 'predictions' or 'true_labels'.")


        except FileNotFoundError:
            print(f"Error: Best model not found at {config.BEST_MODEL_PATH}. Cannot run final evaluation.")
        except Exception as e:
            print(f"\nError during final test evaluation/analysis: {e}")
            import traceback; traceback.print_exc()


    print(f"\n--- Training Pipeline for {config.MODEL_TYPE} Finished ---")

# The main entry point is main.py, which calls run_training_pipeline.
# This check allows running train.py directly for testing if needed, but main.py is preferred.
# if __name__ == "__main__":
#     run_training_pipeline()