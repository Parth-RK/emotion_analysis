import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_on_left=False,
            truncation=True,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def trainmodel_with_early_stopping(tokenizer, max_len, X_train, y_train, X_val, y_val):
    # Define hyperparameters for grid search
    param_grid = {
        'learning_rate': [2e-5, 5e-5],
        'weight_decay': [0.0, 0.1]
    }

    best_model = None
    best_score = -1

    # K-fold cross validation
    kfold = KFold(n_splits=3)

    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_val[val_idx], y_val[val_idx]

        # Initialize dataset and dataloaders
        train_dataset = TextDataset(X_train_fold, y_train_fold, tokenizer, max_len)
        val_dataset = TextDataset(X_val_fold, y_val_fold, tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize model and optimizer
        model = ...
        optimizer = AdamW(model.parameters(), lr=param['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss()

        # Early stopping parameters
        early_stopping = EarlyStopping(patience=5, delta=0.1)

        # Training loop
        for epoch in range(10):
            model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=None,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                loss = loss_fn(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                # Print training metrics every few batches
                if batch_idx % 5 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {avg_loss:.4f}')

            # Validation after epoch
            model.eval()
            val_total_loss = 0
            val_preds = []

            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        token_type_ids=None,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    val_total_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    val_preds.extend(preds)

            # Calculate validation metrics
            avg_val_loss = val_total_loss / len(val_dataset)
            print(f'Validation Loss: {avg_val_loss:.4f}')

            score = avg_val_loss  # or any metric you want to track

            # Early stopping check
            if score < best_score:
                best_score = score
                best_model = model

            # If no improvement for 'patience' epochs, break
            if early_stopping.stop_early(score):
                print("Early stopping triggered!")
                break

    return best_model

# Usage example
tokenizer = ...
max_len = ...

X_train, y_train = ...  # your training data and labels
X_val, y_val = ...       # your validation data and labels

best_model = trainmodel_with_early_stopping(tokenizer, max_len, X_train, y_train, X_val, y_val)

# After finding the best model, you can retrain it on the full training set with the best hyperparameters
final_model = ...