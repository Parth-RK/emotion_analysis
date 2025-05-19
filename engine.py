import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import numpy as np
import os
import time
import sys

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    get_linear_schedule_with_warmup = None
    print("Warning: HuggingFace Transformers scheduler not found.")

try:
    # Import metrics functions, especially for multi-label
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    accuracy_score = None
    precision_recall_fscore_support = None
    print("Warning: scikit-learn not found. Cannot calculate detailed metrics.")


import config
try:
    from models import TransformerClassifier
except ImportError:
     print("ERROR: Could not import TransformerClassifier from models.py")
     sys.exit(1)


def initialize_model(model_type, n_classes):
    """Initializes the model based on type and number of classes."""
    print(f"\nInitializing model: {model_type} with {n_classes} classes")
    if model_type != 'Transformer':
        raise ValueError(f"Unsupported MODEL_TYPE '{model_type}'. Only 'Transformer' is supported now.")

    if not hasattr(config, 'TRANSFORMER_MODEL_NAME'):
         raise ValueError("config.TRANSFORMER_MODEL_NAME must be set.")

    # n_classes must be the total number of distinct labels for multi-label output
    # In GoEmotions, this is 28. This is handled in data_handler.prepare_data
    # and passed correctly here.
    model = TransformerClassifier(
        model_name=config.TRANSFORMER_MODEL_NAME,
        n_classes=n_classes
    )
    model.to(config.DEVICE)

    print(f"Model '{model_type}' ({config.TRANSFORMER_MODEL_NAME}) initialized and moved to {config.DEVICE}")

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
    except Exception as e:
        print(f"  Could not calculate parameter count: {e}")

    return model


def initialize_optimizer_scheduler(model, optimizer_type, scheduler_type, num_train_steps=None):
    """Initializes optimizer and learning rate scheduler."""
    print(f"\nInitializing Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    lr = config.LEARNING_RATE
    wd = config.WEIGHT_DECAY

    optimizer = None
    if optimizer_type == 'AdamW':
        # AdamW is commonly used with Transformers
        # Filter parameters to apply weight decay selectively
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': wd},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Check if any parameters are actually included before creating optimizer
        if not optimizer_grouped_parameters[0]['params'] and not optimizer_grouped_parameters[1]['params']:
             print("Warning: No trainable parameters found in model.")
             optimizer = torch.optim.AdamW([{'params': model.parameters(), 'weight_decay': 0.0}], lr=lr) # Create dummy optimizer
        else:
             optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        print(f"  Using AdamW with LR={lr}, Weight Decay={wd} (applied selectively)")
    elif optimizer_type == 'Adam':
        # Filter for parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=wd)
        print(f"  Using Adam with LR={lr}, Weight Decay={wd}")
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {optimizer_type}")

    scheduler = None
    if scheduler_type == 'linear_warmup':
        if get_linear_schedule_with_warmup is None:
             print("Warning: 'linear_warmup' requested, but Transformers library failed import. No scheduler used.")
        elif num_train_steps is None or num_train_steps <= 0:
            print(f"Warning: num_train_steps ({num_train_steps}) invalid for linear_warmup scheduler. No scheduler used.")
        else:
            num_warmup_steps = int(num_train_steps * config.WARMUP_PROPORTION)
            print(f"  Using Linear Warmup scheduler: Total steps={num_train_steps}, Warmup steps={num_warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
            )
    elif scheduler_type == 'reduce_on_plateau':
        patience = getattr(config, 'SCHEDULER_PATIENCE', 2)
        factor = getattr(config, 'SCHEDULER_FACTOR', 0.1)
        # For multi-label, we still monitor validation loss typically
        print(f"  Using ReduceLROnPlateau scheduler: Factor={factor}, Patience={patience}, Monitoring 'val_loss'")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    elif scheduler_type is None or scheduler_type.lower() == 'none':
         print("  No learning rate scheduler selected.")
    else:
        print(f"Warning: Scheduler type '{scheduler_type}' not implemented/recognized. No scheduler used.")

    return optimizer, scheduler


# --- Criterion will now be initialized in train.py and passed ---
# Removing global criterion = nn.BCEWithLogitsLoss()


def train_step(model, data_loader, optimizer, device, criterion, scheduler=None, grad_clip_value=None):
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Training", leave=False, unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        try:
            # input_ids and attention_mask are standard
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Labels are now multi-hot tensors (float)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs are raw logits from the final linear layer

        except KeyError as e:
             print(f"\nError: Missing key {e} in training batch {batch_idx}. Check Dataset __getitem__.")
             print(f"Batch keys: {batch.keys()}")
             raise # Raise error for critical data issue
        except Exception as e:
             print(f"\nError during forward pass in training batch {batch_idx}: {e}")
             # Print shapes for debugging input errors
             print(f"Input Shapes: ids={input_ids.shape if 'input_ids' in batch else 'N/A'}, mask={attention_mask.shape if 'attention_mask' in batch else 'N/A'}, labels={labels.shape if 'labels' in batch else 'N/A'}")
             raise # Raise error for critical model/data issue

        # Calculate loss using the passed criterion (BCEWithLogitsLoss with pos_weight)
        # Expected: outputs (logits float [batch_size, n_classes]), labels (multi-hot float [batch_size, n_classes])
        loss = criterion(outputs, labels)

        # Backpropagation and Optimization
        loss.backward()
        if grad_clip_value is not None and grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        # Scheduler step (only for linear warmup type)
        if scheduler and config.SCHEDULER_TYPE == 'linear_warmup':
            scheduler.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time
    print(f"  Train Avg. Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")

    return avg_loss


def evaluate_step(model, data_loader, device, criterion):
    """Performs evaluation on a data loader for multi-label classification."""
    if data_loader is None or len(data_loader) == 0:
        print("  Evaluation skipped: DataLoader is empty or None.")
        # Determine number of classes for empty arrays if possible
        num_classes_from_data = 0
        if data_loader and data_loader.dataset and len(data_loader.dataset) > 0:
             try:
                  # Access shape from the first item in the dataset
                  num_classes_from_data = data_loader.dataset[0]['labels'].shape[0]
             except Exception:
                  pass # Cannot determine, leave as 0


        return {'loss': float('nan'), 'accuracy': 0.0, 'precision_weighted': 0.0,
                'recall_weighted': 0.0, 'f1_weighted': 0.0,
                'predictions': np.array([]).reshape(0, num_classes_from_data), # Return empty numpy array of correct shape
                'true_labels': np.array([]).reshape(0, num_classes_from_data)} # Assume labels structure from first item


    model.eval()
    total_loss = 0.0
    all_preds_binary = [] # To store binary predictions (numpy arrays)
    all_labels_multihot = [] # To store true multi-hot labels (numpy arrays)

    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch")

    # Sigmoid activation for probabilities - needed *after* getting logits
    sigmoid = torch.nn.Sigmoid()
    # Use the prediction threshold from config for converting probabilities to binary predictions
    prediction_threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.5)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # Labels are multi-hot tensors (float)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # outputs are raw logits

            except KeyError as e:
                 print(f"\nError: Missing key {e} in evaluation batch {batch_idx}. Check Dataset __getitem__.")
                 print(f"Batch keys: {batch.keys()}")
                 raise
            except Exception as e:
                 print(f"\nError during forward pass in evaluation batch {batch_idx}: {e}")
                 print(f"Input Shapes: ids={input_ids.shape if 'input_ids' in batch else 'N/A'}, mask={attention_mask.shape if 'attention_mask' in batch else 'N/A'}, labels={labels.shape if 'labels' in batch else 'N/A'}")
                 raise

            # Calculate loss (for logging, not backprop). Use the passed criterion.
            # Note: pos_weight in criterion *only* affects the training loss calculation.
            # Evaluation loss is calculated the same way, but pos_weight doesn't change the eval metric values themselves.
            # We calculate loss here just to monitor it.
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get probabilities and convert to binary predictions using threshold
            probs = sigmoid(outputs)
            preds_binary = (probs > prediction_threshold).int() # Convert to int (0 or 1)

            # Store predictions and true labels (move to CPU and convert to numpy)
            all_preds_binary.append(preds_binary.cpu().numpy())
            all_labels_multihot.append(labels.cpu().numpy()) # labels were already float, now on cpu

            progress_bar.set_postfix({'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'})


    avg_loss = total_loss / len(data_loader)

    # Concatenate all batch results
    if all_preds_binary and all_labels_multihot:
        all_preds_np = np.vstack(all_preds_binary)
        all_labels_np = np.vstack(all_labels_multihot)
    else:
        # Handle case where data_loader was not empty but no batches were processed (e.g., errors)
        print("Warning: No data processed during evaluation step.")
        # Attempt to get class dim from dataset if available
        num_classes_from_data = 0
        if data_loader and data_loader.dataset and len(data_loader.dataset) > 0:
             try:
                  num_classes_from_data = data_loader.dataset[0]['labels'].shape[0]
             except Exception:
                  pass

        return {'loss': float('nan'), 'accuracy': 0.0, 'precision_weighted': 0.0,
                'recall_weighted': 0.0, 'f1_weighted': 0.0,
                'predictions': np.array([]).reshape(0, num_classes_from_data),
                'true_labels': np.array([]).reshape(0, num_classes_from_data)}


    elapsed_time = time.time() - start_time

    # Calculate multi-label metrics using scikit-learn
    accuracy = 0.0 # Subset accuracy
    precision, recall, f1 = 0.0, 0.0, 0.0 # Weighted P/R/F1

    if accuracy_score is not None and precision_recall_fscore_support is not None:
        try:
            # Subset accuracy: The set of predicted labels must exactly match the set of true labels.
            accuracy = accuracy_score(all_labels_np, all_preds_np)

            # Weighted metrics: Calculate metrics for each label, then average, weighted by support (# of true instances for each label).
            # Use all_labels_np (true labels) for support calculation
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels_np, all_preds_np, average='weighted', zero_division=0
            )
            # If needed, you can also get per-class metrics by removing 'average'
            # per_class_metrics = precision_recall_fscore_support(all_labels_np, all_preds_np, average=None, zero_division=0)


        except Exception as e:
            print(f"Warning: Error calculating multi-label metrics using scikit-learn: {e}")
            import traceback; traceback.print_exc()
            accuracy, precision, recall, f1 = float('nan'), float('nan'), float('nan'), float('nan') # Indicate failure

    print(f"  Eval Avg. Loss:  {avg_loss:.4f} | Sub. Acc: {accuracy:.4f} | F1 (W): {f1:.4f} | P (W): {precision:.4f} | R (W): {recall:.4f} | Time: {elapsed_time:.2f}s")

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy, # Subset Accuracy
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'predictions': all_preds_np, # Return numpy arrays for plotting/reporting
        'true_labels': all_labels_np
    }

    return metrics


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_save_path, criterion, metric_for_best=config.METRIC_FOR_BEST_MODEL):
    """Main training loop."""
    # History dict to store metrics per epoch
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_weighted': [], 'val_precision_weighted': [], 'val_recall_weighted': []}

    # Initialize best metric value based on the monitoring metric
    if metric_for_best == 'loss':
        best_metric_value = float('inf')
        is_better_op = lambda current, best: current < best
    elif metric_for_best in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
        best_metric_value = -float('inf')
        is_better_op = lambda current, best: current > best
    else:
        print(f"Warning: Unknown metric_for_best '{metric_for_best}'. Monitoring validation loss instead.")
        metric_for_best = 'loss'
        best_metric_value = float('inf')
        is_better_op = lambda current, best: current < best


    grad_clip_value = getattr(config, 'GRADIENT_CLIP_VALUE', None)

    print(f"\n--- Starting Training ---")
    print(f"Model Type: {config.MODEL_TYPE} ({config.TRANSFORMER_MODEL_NAME})")
    print(f"Epochs: {epochs}, Device: {device}")
    print(f"Optimizer: {config.OPTIMIZER_TYPE}, Scheduler: {config.SCHEDULER_TYPE}")
    print(f"Monitoring validation '{metric_for_best}' for best model.")
    print(f"Using Loss Function: {type(criterion).__name__} (with pos_weight={criterion.pos_weight.cpu().numpy() if hasattr(criterion, 'pos_weight') else 'None'})")
    if grad_clip_value: print(f"Using gradient clipping: {grad_clip_value}")
    print(f"Model checkpoints will be saved to: {model_save_path}")
    print(f"Using Prediction Threshold for metrics: {getattr(config, 'PREDICTION_THRESHOLD', 0.5)}")


    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train step - pass the criterion
        train_loss = train_step(model, train_loader, optimizer, device, criterion, scheduler, grad_clip_value)
        history['train_loss'].append(train_loss)

        # Evaluate step - pass the criterion (even though pos_weight doesn't affect eval metrics calculation, loss is calculated for logging)
        val_metrics = evaluate_step(model, val_loader, device, criterion)

        # Store validation metrics
        # Check if evaluation was skipped or failed
        if np.isnan(val_metrics['loss']): # Check loss as a primary indicator of evaluation success
             print("  Skipping validation metrics recording and best model check due to evaluation failure (loss is NaN).")
             # Append NaNs to history for plotting consistency
             history['val_loss'].append(float('nan'))
             history['val_accuracy'].append(float('nan'))
             history['val_f1_weighted'].append(float('nan'))
             history['val_precision_weighted'].append(float('nan'))
             history['val_recall_weighted'].append(float('nan'))

             # If scheduler monitors loss, step might need careful handling with NaN
             if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
                 # ReduceLROnPlateau handles inf/nan loss correctly by not stepping
                 scheduler.step(val_metrics['loss'])
             continue # Skip best model saving logic


        history['val_loss'].append(val_metrics.get('loss', float('nan')))
        history['val_accuracy'].append(val_metrics.get('accuracy', float('nan')))
        history['val_f1_weighted'].append(val_metrics.get('f1_weighted', float('nan')))
        history['val_precision_weighted'].append(val_metrics.get('precision_weighted', float('nan')))
        history['val_recall_weighted'].append(val_metrics.get('recall_weighted', float('nan')))

        # Scheduler step (only for ReduceLROnPlateau)
        if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
            # Scheduler steps based on the monitoring metric, which is usually validation loss
            monitor_value = val_metrics.get(getattr(config, 'SCHEDULER_MONITOR', 'loss'), val_metrics['loss']) # Default to loss if config not set
            scheduler.step(monitor_value)


        # Check for best model based on the monitoring metric
        current_metric_value = val_metrics.get(metric_for_best)

        if current_metric_value is None or np.isnan(current_metric_value):
             print(f"Warning: Metric '{metric_for_best}' not found or is NaN in validation metrics. Cannot save best model.")
             continue # Skip saving logic


        if is_better_op(current_metric_value, best_metric_value):
            print(f"  ✨ Validation '{metric_for_best}' improved ({best_metric_value:.4f} --> {current_metric_value:.4f}). Saving model...")
            best_metric_value = current_metric_value
            try:
                 os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                 # Save model state dictionary
                 torch.save(model.state_dict(), model_save_path)
                 print(f"     Model saved to {model_save_path}")
            except Exception as e:
                 print(f"     Error saving model to {model_save_path}: {e}")
        else:
            print(f"  Validation '{metric_for_best}' ({current_metric_value:.4f}) did not improve from best ({best_metric_value:.4f}).")


    end_training_time = time.time()
    total_training_time = end_training_time - start_training_time

    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Best validation '{metric_for_best}' achieved: {best_metric_value:.4f}")
    print(f"Model artifacts saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")

    return history


def load_trained_model(model_path, model_type, n_classes):
    """Loads a trained model state dictionary."""
    print(f"\nAttempting to load model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if model_type != 'Transformer':
         raise ValueError("load_trained_model currently only supports 'Transformer' type.")

    try:
        # Initialize model architecture first
        model = initialize_model(model_type, n_classes)

        # Load state dictionary
        # Use map_location to ensure it loads correctly regardless of available devices
        state_dict = torch.load(model_path, map_location=torch.device(config.DEVICE))

        # Load state dictionary into the model
        model.load_state_dict(state_dict)

        print(f"Model weights loaded successfully onto {config.DEVICE}.")

        # Set model to evaluation mode by default after loading
        model.eval()

        return model

    except FileNotFoundError:
        raise # Re-raise if file wasn't found after all
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Check for architecture mismatch (config settings vs saved model) or corrupted file.")
        import traceback; traceback.print_exc() # Print detailed error
        raise # Re-raise for calling function to handle
