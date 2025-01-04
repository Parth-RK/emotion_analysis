import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pickle
from sklearn.metrics import accuracy_score

# Load and preprocess data
data_path = "dataset/emotion_data.csv"  # Path to your dataset
df = pd.read_csv(data_path)

# Encode labels
label_map = {label: idx for idx, label in enumerate(df['sentiment'].unique())}
df['label'] = df['sentiment'].map(label_map)
print(label_map)
# Save label map
with open('label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['content'].values, df['label'].values, test_size=0.2, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model
model = EmotionClassifier(num_labels=len(label_map))
model.to(device)  # Move model to GPU if available

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=25):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)  # Move to GPU if available
            attention_mask = batch['attention_mask'].to(device)  # Move to GPU if available
            labels = batch['label'].to(device)  # Move to GPU if available
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        evaluate_model(model, val_loader)

# Evaluation
def evaluate_model(model, val_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, axis=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # print(classification_report(true_labels, predictions, labels=list(label_map.values()), target_names=list(label_map.keys())))
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")



# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion)

# Save the model
torch.save(model.state_dict(), "models/emotion_model.pth")
print("Model saved!")

# Predict on new data
def predict_emotion(texts):
    model.eval()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=encodings['input_ids'].to(device), attention_mask=encodings['attention_mask'].to(device))
        predictions = torch.argmax(outputs, axis=1).cpu().numpy()
    return [list(label_map.keys())[pred] for pred in predictions]

# Example usage
texts = ["I'm so happy today!", "This is terrible...", "I hate this"]
print(predict_emotion(texts))