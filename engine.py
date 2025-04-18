import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
from tqdm import tqdm
import config
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import numpy as np

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def evaluate(model, data_loader, criterion, device):
    print("Evaluating...")
    total_loss, total_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Evaluation", leave=False):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            batch_loss = criterion(y_pred_logits, y)
            total_loss += batch_loss.item()

            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            total_acc += accuracy_fn(y, y_pred_class)

    avg_acc = total_acc / len(data_loader)
    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation Complete - Avg Loss: {avg_loss:.5f}, Avg Accuracy: {avg_acc:.2f}%")
    return avg_acc, avg_loss

def trainer(model, train_loader, optimizer, criterion, epochs, device, val_loader=None, model_save_path=None):
    history = {"train_loss": [], "train_acc": [], "epoch": []}
    if val_loader:
        history["val_loss"] = []
        history["val_acc"] = []

    best_val_loss = float('inf')
    model.to(device)
    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0, 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            loss = criterion(y_pred_logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            batch_acc = accuracy_fn(y, y_pred_class)
            epoch_loss += loss.item()
            epoch_acc += batch_acc
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}'})

        print(f"\n--- Evaluating Epoch {epoch} ---")
        train_acc, train_loss = evaluate(model, train_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["epoch"].append(epoch)

        log_message = f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%"

        if val_loader:
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            log_message += f" | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.2f}%"

            if val_loss < best_val_loss and model_save_path:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, model_save_path)
                log_message += " ✨ Best Model Saved ✨"

        print(log_message)
        print("-" * 50)

    print("Training Finished.")
    if model_save_path and os.path.exists(model_save_path) and val_loader:
         print(f"Loading best model from {model_save_path} based on validation loss.")
         load_checkpoint(model_save_path, model, optimizer, device)

    return model, pd.DataFrame(history)


def save_checkpoint(model, optimizer, epoch, filepath):
    print(f"Saving checkpoint to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print("Checkpoint saved.")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"Warning: Could not load optimizer state dict. {e}")
            print("Optimizer state will be reset.")
    else:
         print("Info: Optimizer state not loaded (not found or optimizer not provided).")

    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Checkpoint loaded. Model weights loaded from epoch {epoch}.")
    model.to(device)
    return checkpoint.get('epoch', 0)

def save_final_model(model, filepath):
    print(f"Saving final model state_dict to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print("Final model saved.")

def load_final_model(model, filepath, device='cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    print(f"Loading final model state_dict from {filepath}...")
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    print("Final model loaded.")

def plot_history(df, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker='o')
    if "val_acc" in df.columns:
        plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    plt.show()

def generate_test_report(model, data_loader, criterion, device, int_to_label_map, report_save_path=None, conf_matrix_save_path=None):
    """Evaluates the model on the test set and generates a report."""
    print("\n--- Generating Final Test Report ---")
    # Get predictions and true labels
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Test Evaluation"):
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            loss = criterion(y_pred_logits, y)
            total_loss += loss.item()
            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100

    print(f"\nTest Loss: {avg_loss:.5f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Generate Classification Report
    label_names = [int_to_label_map.get(i, str(i)) for i in sorted(int_to_label_map.keys())]
    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, digits=3)
    print("\nClassification Report:")
    print(report)

    if report_save_path:
        os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
        with open(report_save_path, 'w') as f:
            f.write(f"Test Loss: {avg_loss:.5f}\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Classification report saved to {report_save_path}")

    # Generate and Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=sorted(int_to_label_map.keys()))
    plt.figure(figsize=(max(8, len(label_names)*0.6), max(6, len(label_names)*0.5))) # Adjust size based on num classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if conf_matrix_save_path:
        os.makedirs(os.path.dirname(conf_matrix_save_path), exist_ok=True)
        plt.savefig(conf_matrix_save_path)
        print(f"Confusion matrix saved to {conf_matrix_save_path}")
    plt.show()