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
import config
try:
    from models import TransformerClassifier
except ImportError:
     print("ERROR: Could not import TransformerClassifier from models.py")
     sys.exit(1)
def initialize_model(model_type, n_classes):
    print(f"\nInitializing model: {model_type} with {n_classes} classes")
    if model_type != 'Transformer':
        raise ValueError(f"Unsupported MODEL_TYPE '{model_type}'. Only 'Transformer' is supported now.")
    if not hasattr(config, 'TRANSFORMER_MODEL_NAME'):
         raise ValueError("config.TRANSFORMER_MODEL_NAME must be set.")
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
    print(f"\nInitializing Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    lr = config.LEARNING_RATE
    wd = config.WEIGHT_DECAY
    optimizer = None
    if optimizer_type == 'AdamW':
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': wd},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        print(f"  Using AdamW with LR={lr}, Weight Decay={wd} (applied selectively)")
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        print(f"  Using Adam with LR={lr}, Weight Decay={wd}")
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {optimizer_type}")
    scheduler = None
    if scheduler_type == 'linear_warmup':
        if get_linear_schedule_with_warmup is None:
             print("Warning: 'linear_warmup' requested, but Transformers library failed import. No scheduler used.")
        elif num_train_steps is None or num_train_steps <= 0:
            print("Warning: num_train_steps invalid for linear_warmup scheduler. No scheduler used.")
        else:
            num_warmup_steps = int(num_train_steps * config.WARMUP_PROPORTION)
            print(f"  Using Linear Warmup scheduler: Total steps={num_train_steps}, Warmup steps={num_warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
            )
    elif scheduler_type == 'reduce_on_plateau':
        patience = getattr(config, 'SCHEDULER_PATIENCE', 2)
        factor = getattr(config, 'SCHEDULER_FACTOR', 0.1)
        print(f"  Using ReduceLROnPlateau scheduler: Factor={factor}, Patience={patience}, Monitoring 'val_loss'")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    elif scheduler_type is None or scheduler_type.lower() == 'none':
         print("  No learning rate scheduler selected.")
    else:
        print(f"Warning: Scheduler type '{scheduler_type}' not implemented/recognized. No scheduler used.")
    return optimizer, scheduler
criterion = nn.CrossEntropyLoss()
print(f"\nUsing Loss Function: CrossEntropyLoss")
def train_step(model, data_loader, optimizer, device, scheduler=None, grad_clip_value=None):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Training", leave=False, unit="batch")
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except KeyError as e:
             print(f"\nError: Missing key {e} in training batch {batch_idx}. Check Dataset __getitem__.")
             print(f"Batch keys: {batch.keys()}")
             raise
        except Exception as e:
             print(f"\nError during forward pass in training batch {batch_idx}: {e}")
             print(f"Input Shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}")
             raise
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip_value is not None and grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        if scheduler and config.SCHEDULER_TYPE == 'linear_warmup':
            scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time
    print(f"  Train Avg. Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
    return avg_loss
def evaluate_step(model, data_loader, device):
    if data_loader is None or len(data_loader) == 0:
        print("  Evaluation skipped: DataLoader is empty or None.")
        return {'loss': float('nan'), 'accuracy': 0.0, 'precision_weighted': 0.0,
                'recall_weighted': 0.0, 'f1_weighted': 0.0,
                'predictions': [], 'true_labels': []}
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except KeyError as e:
                 print(f"\nError: Missing key {e} in evaluation batch {batch_idx}. Check Dataset __getitem__.")
                 print(f"Batch keys: {batch.keys()}")
                 raise
            except Exception as e:
                 print(f"\nError during forward pass in evaluation batch {batch_idx}: {e}")
                 print(f"Input Shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}")
                 raise
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'})
    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        accuracy = accuracy_score(all_labels_np, all_preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average='weighted', zero_division=0
        )
    except ImportError:
         print("Warning: scikit-learn not found. Cannot calculate detailed metrics (precision, recall, F1).")
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
    print(f"  Eval Avg. Loss:  {avg_loss:.4f} | Accuracy: {accuracy:.4f} | F1 (W): {f1:.4f} | Time: {elapsed_time:.2f}s")
    metrics = {'loss': avg_loss, 'accuracy': accuracy, 'precision_weighted': precision,
               'recall_weighted': recall, 'f1_weighted': f1,
               'predictions': all_preds, 'true_labels': all_labels}
    return metrics
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_save_path, metric_for_best=config.METRIC_FOR_BEST_MODEL):
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_weighted': []}
    best_metric_value = -float('inf') if metric_for_best != 'loss' else float('inf')
    grad_clip_value = getattr(config, 'GRADIENT_CLIP_VALUE', None)
    print(f"\n--- Starting Training ---")
    print(f"Model Type: {config.MODEL_TYPE} ({config.TRANSFORMER_MODEL_NAME})")
    print(f"Epochs: {epochs}, Device: {device}")
    print(f"Optimizer: {config.OPTIMIZER_TYPE}, Scheduler: {config.SCHEDULER_TYPE}")
    print(f"Monitoring validation '{metric_for_best}' for best model.")
    if grad_clip_value: print(f"Using gradient clipping: {grad_clip_value}")
    print(f"Model checkpoints will be saved to: {model_save_path}")
    start_training_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_step(model, train_loader, optimizer, device, scheduler, grad_clip_value)
        history['train_loss'].append(train_loss)
        val_metrics = evaluate_step(model, val_loader, device)
        if val_metrics['loss'] is float('nan'):
             print("  Skipping validation metrics recording and best model check.")
             continue
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler.step(val_metrics['loss'])
        current_metric_value = val_metrics.get(metric_for_best)
        if current_metric_value is None:
             print(f"Warning: Metric '{metric_for_best}' not found in validation metrics. Cannot save best model.")
             continue
        is_better = (current_metric_value < best_metric_value) if metric_for_best == 'loss' else (current_metric_value > best_metric_value)
        if is_better:
            print(f"  ✨ Validation '{metric_for_best}' improved ({best_metric_value:.4f} --> {current_metric_value:.4f}). Saving model...")
            best_metric_value = current_metric_value
            try:
                 os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                 torch.save(model.state_dict(), model_save_path)
                 print(f"     Model saved to {model_save_path}")
            except Exception as e: print(f"     Error saving model: {e}")
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
    print(f"\nAttempting to load model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if model_type != 'Transformer':
         raise ValueError("load_trained_model currently only supports 'Transformer' type.")
    try:
        model = initialize_model(model_type, n_classes)
        state_dict = torch.load(model_path, map_location=torch.device(config.DEVICE))
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully onto {config.DEVICE}.")
        model.eval()
        return model
    except FileNotFoundError: raise
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Check for architecture mismatch (config settings vs saved model) or corrupted file.")
        raise