import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle

# Load the label map
with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model definition
class EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# Load the model
model = EmotionClassifier(num_labels=len(label_map))
model.load_state_dict(torch.load("models/emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

# Predict function
def predict_emotion(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
        predictions = torch.argmax(outputs, axis=1).cpu().numpy()
    return [list(label_map.keys())[pred] for pred in predictions]

# Example usage
if __name__ == "__main__":
    texts = [
        "I'm so happy today!", 
        "This is terrible...",  
        "I'm feeling sad", 
        "I feel neutral about this situation.", 
        "I'm so excited!",
        #Custom text for prediction here
        
        ]
    print(predict_emotion(texts))