import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# --- Use standard relative import ---
from . import config
# -----------------------------------

# Define a max gradient norm value for clipping
MAX_GRAD_NORM = 1.0

def loss_fn(outputs, targets):
    """
    Calculates the cross-entropy loss.
    """
    return nn.CrossEntropyLoss()(outputs, targets)

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training with gradient clipping.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, total=len(data_loader), desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # If using AutoModelForSequenceClassification, outputs might be an object:
        # loss = outputs.loss
        # logits = outputs.logits # Use logits for calculating metrics if needed
        # However, calculating loss directly often works too.
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()

        # --- Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        # -----------------------

        optimizer.step()
        if scheduler:
            scheduler.step()

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """
    Performs evaluation on the validation set.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(data_loader, total=len(data_loader), desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # If using AutoModelForSequenceClassification, outputs might be an object:
            # logits = outputs.logits
            # loss = loss_fn(logits, labels) # Calculate loss if not directly available
            loss = loss_fn(outputs, labels) # Assumes outputs are logits
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1) # Assumes outputs are logits

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    # Use output_dict=True for easier access to metrics
    report_dict = classification_report(all_labels, all_preds, target_names=config.EMOTIONS, zero_division=0, output_dict=True)
    # Generate the formatted string report as well
    report_str = classification_report(all_labels, all_preds, target_names=config.EMOTIONS, zero_division=0)


    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Validation Classification Report:")
    print(report_str) # Print the nicely formatted report directly

    # Return the dictionary version for potential programmatic use
    return avg_loss, accuracy, report_dict