import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings

try:
    # Need these for multi-label classification_report
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    # MultiLabelBinarizer is not needed here as we handle multi-hot conversion in data_handler
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError:
    classification_report = None
    confusion_matrix = None
    accuracy_score = None
    MultiLabelBinarizer = None
    print("Warning: scikit-learn not installed. Classification report and confusion matrix generation will be unavailable.")
    print("         Install it using: pip install scikit-learn")

import config

# Set theme for plots
sns.set_theme(style="whitegrid")

def plot_training_history(history, save_path=None):
    """Plots training and validation loss and metrics."""
    if not isinstance(history, dict) or not history:
        print("Plotter Warning: History dictionary is empty or invalid. Cannot plot training history.")
        return

    # Default save path from config
    if save_path is None:
        save_path = getattr(config, 'TRAINING_PLOTS_PATH', None)
        if save_path is None:
             print("Plotter Error: Default save path (config.TRAINING_PLOTS_PATH) not found and no save_path provided.")
             return

    # Check for required data
    if 'train_loss' not in history or not history['train_loss']:
         print("Plotter Warning: 'train_loss' not found or empty in history. Cannot plot.")
         return

    # Convert history dict to DataFrame for easier plotting
    df = pd.DataFrame(history)
    if df.empty:
        print("Plotter Warning: History DataFrame is empty. Cannot plot.")
        return

    df['epoch'] = range(1, len(df) + 1)

    # Define plottable metrics (matching keys returned by engine.evaluate_step)
    # Added precision and recall for multi-label
    plottable_metrics = {
        'loss': {'train': 'train_loss', 'val': 'val_loss', 'title': 'Loss', 'ylim': (None, None)},
        'accuracy': {'train': None, 'val': 'val_accuracy', 'title': 'Subset Accuracy', 'ylim': (0, 1)}, # Renamed Accuracy
        'f1': {'train': None, 'val': 'val_f1_weighted', 'title': 'Weighted F1 Score', 'ylim': (0, 1)},
        'precision': {'train': None, 'val': 'val_precision_weighted', 'title': 'Weighted Precision', 'ylim': (0, 1)},
        'recall': {'train': None, 'val': 'val_recall_weighted', 'title': 'Weighted Recall', 'ylim': (0, 1)},
    }

    # Filter for metrics present in the DataFrame and having non-NaN values
    active_plots = {
        k: v for k, v in plottable_metrics.items()
        if (v['train'] in df.columns and df[v['train']].notna().any()) or (v['val'] in df.columns and df[v['val']].notna().any())
    }

    if not active_plots:
        print("Plotter Warning: No plottable metrics found in history dict.")
        return

    num_plots = len(active_plots)
    # Calculate figure size based on the number of plots
    fig_width = max(6, 6 * num_plots) # Ensure min width 6
    fig_height = 5
    fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, fig_height), squeeze=False) # squeeze=False ensures axes is always 2D

    plot_idx = 0
    for metric_key, cfg in active_plots.items():
        ax = axes[0, plot_idx] # Get the current subplot axis

        has_train = cfg.get('train') and cfg['train'] in df.columns and df[cfg['train']].notna().any()
        has_val = cfg.get('val') and cfg['val'] in df.columns and df[cfg['val']].notna().any()

        if has_train:
            ax.plot(df['epoch'], df[cfg['train']], label=f"Train {cfg['title']}", marker='o', linestyle='-', markersize=4)
        if has_val:
            ax.plot(df['epoch'], df[cfg['val']], label=f"Validation {cfg['title']}", marker='x', linestyle='--', markersize=4)

        ax.set_title(f"{cfg['title']} vs. Epoch")
        ax.set_xlabel('Epoch')
        ax.set_ylabel(cfg['title'])

        # Set y-limits if specified
        if cfg['ylim'] is not None:
            y_min, y_max = cfg['ylim']
            # Adjust limits dynamically for loss, especially for better visualization
            if metric_key == 'loss':
                 all_loss_values = []
                 if has_train: all_loss_values.extend(df[cfg['train']].dropna().tolist())
                 if has_val: all_loss_values.extend(df[cfg['val']].dropna().tolist())
                 if all_loss_values:
                      min_loss = min(all_loss_values)
                      max_loss = max(all_loss_values)
                      # Add padding, but prevent negative loss limits
                      ymin_padded = max(0.0, min_loss * 0.95) if min_loss >= 0 else min_loss * 1.05
                      ymax_padded = max_loss * 1.05 if max_loss >= 0 else max_loss * 0.95
                      # Ensure min < max, handle flat lines
                      if ymin_padded >= ymax_padded: ymax_padded = ymin_padded + 0.1 # Add minimal range if flat
                      ax.set_ylim(bottom=ymin_padded, top=ymax_padded)
            elif y_min is not None or y_max is not None:
                ax.set_ylim(bottom=y_min, top=y_max)


        ax.legend()
        ax.grid(True)
        plot_idx += 1

    plt.tight_layout(pad=2.0)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Plotter Error: Could not save training plot to {save_path}. Error: {e}")
    else:
        # Optionally show plot if save_path is None (e.g., for debugging)
        # plt.show()
        pass # Just close silently if no save path
    plt.close(fig) # Use fig object to close plot correctly


def generate_classification_analysis(true_labels, predictions, int_to_label, report_path=None, cm_path=None, prefix=""):
    """
    Generates and prints a classification report for multi-label data.
    Note: Confusion matrix plot is skipped for multi-label classification
    as standard confusion matrix is not applicable.
    """
    if classification_report is None or accuracy_score is None:
        print("Plotter Info: Skipping classification analysis because scikit-learn is not installed.")
        return

    if report_path is None:
        report_path = getattr(config, 'TEST_REPORT_PATH', None)
    if cm_path is None:
        # For multi-label, we generally skip the standard confusion matrix plot.
        # Setting cm_path to None to prevent plotting it.
        cm_path = None # Force skip CM plot for multi-label

    # Ensure inputs are numpy arrays and have the same shape
    if not isinstance(true_labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        print("Plotter Error: true_labels and predictions must be numpy arrays (multi-hot format).")
        return

    if true_labels.shape != predictions.shape:
        print(f"Plotter Error: Shape mismatch between true_labels {true_labels.shape} and predictions {predictions.shape}.")
        return

    if true_labels.size == 0:
        print("Plotter Info: true_labels and predictions are empty. Skipping classification analysis.")
        return

    # Use the loaded int_to_label mapping for target names
    if not int_to_label:
        print("Plotter Warning: int_to_label mapping not provided or empty. Using integer indices as names.")
        num_classes = true_labels.shape[1]
        label_names = [str(i) for i in range(num_classes)]
        # For reporting, we need the indices corresponding to the columns
        target_labels_for_report = list(range(num_classes))
    else:
        try:
            # Ensure int_to_label has entries for all columns in true_labels/predictions
            num_classes_data = true_labels.shape[1]
            if len(int_to_label) != num_classes_data:
                print(f"Plotter Warning: Mismatch between number of classes in data ({num_classes_data}) and int_to_label map ({len(int_to_label)}). Using integer indices as names.")
                label_names = [str(i) for i in range(num_classes_data)]
                target_labels_for_report = list(range(num_classes_data))
            else:
                # Sort labels by index to match column order
                sorted_labels = sorted(int_to_label.items())
                label_names = [label for index, label in sorted_labels]
                # Ensure target_labels for report are the indices 0 to N-1
                target_labels_for_report = [index for index, label in sorted_labels] # Should be 0 to N-1

        except Exception as e:
            print(f"Plotter Warning: Error processing int_to_label mapping ({e}). Using integer indices as names.")
            num_classes = true_labels.shape[1]
            label_names = [str(i) for i in range(num_classes)]
            target_labels_for_report = list(range(num_classes))


    try:
        # Generate classification report for multi-label
        # true_labels and predictions should be indicator matrices (multi-hot)
        report_str = classification_report(
            true_labels,
            predictions,
            target_names=label_names,
            zero_division=0, # Handle classes with no samples/predictions
            digits=4,
            # Multi-label classification_report works directly with indicator matrices
        )

        title = f"{prefix} Multi-Label Classification Report" if prefix else "Multi-Label Classification Report"

        # Calculate overall subset accuracy
        # Subset accuracy is the strict metric: prediction must match all true labels exactly
        subset_accuracy = accuracy_score(true_labels, predictions)


        full_report_output = f"\n--- {title} ---\n"
        full_report_output += f"Overall Subset Accuracy: {subset_accuracy:.4f}\n\n"
        full_report_output += report_str
        full_report_output += "\n-----------------------------------\n"

        print(full_report_output)

        if report_path:
            try:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(full_report_output)
                print(f"Classification report saved to {report_path}")
            except Exception as e:
                print(f"Plotter Error: Could not save classification report to {report_path}. Error: {e}")

    except Exception as e:
        print(f"Plotter Error: Could not generate classification report. Error: {e}")
        import traceback; traceback.print_exc();

    # --- Confusion Matrix (SKIP for Multi-Label) ---
    if cm_path: # This condition will be False due to cm_path=None above
        print("Plotter Info: Skipping confusion matrix plot as it's not directly applicable to multi-label classification.")
        # If you needed *per-class* binary CMs, you would iterate through each label.
        # This is beyond the scope of a standard CM function.
