import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def loss_fn(outputs, targets):
    """
    Calculates the cross-entropy loss.
    """
    # Consider adding class weights here if dealing with imbalance
    # weight = torch.tensor([...]).to(config.DEVICE)
    # return nn.CrossEntropyLoss(weight=weight)(outputs, targets)
    return nn.CrossEntropyLoss()(outputs, targets)

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, total=len(data_loader), desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Update scheduler (if using one)
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

    with torch.no_grad(): # Disable gradient calculations
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Get predictions (highest logit index)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.EMOTIONS, zero_division=0, output_dict=True)

    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Validation Classification Report:")
    # Print a formatted report
    print("{:<12} {:<10} {:<10} {:<10} {:<10}".format("Emotion", "Precision", "Recall", "F1-Score", "Support"))
    print("-" * 55)
    for emotion in config.EMOTIONS:
        metrics = report.get(emotion, {})
        print("{:<12} {:<10.3f} {:<10.3f} {:<10.3f} {:<10}".format(
            emotion,
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1-score', 0),
            metrics.get('support', 0)
        ))
    print("-" * 55)
    print("{:<12} {:<10.3f} {:<10.3f} {:<10.3f} {:<10}".format(
        "macro avg",
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score'],
        report['macro avg']['support']
    ))
    print("{:<12} {:<10.3f} {:<10.3f} {:<10.3f} {:<10}".format(
        "weighted avg",
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score'],
        report['weighted avg']['support']
    ))
    print("-" * 55)


    return avg_loss, accuracy, report # Return detailed report as dict