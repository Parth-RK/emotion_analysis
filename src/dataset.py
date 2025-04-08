import torch
from torch.utils.data import Dataset
from . import config
from .preprocess import clean_text

class EmotionDataset(Dataset):
    """
    PyTorch Dataset class for loading and tokenizing emotion data.
    """
    def __init__(self, dataframe, tokenizer, max_len, include_labels=True):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.content = dataframe['content'].apply(clean_text).values
        self.include_labels = include_labels
        if self.include_labels:
            self.labels = dataframe['sentiment'].map(config.EMOTION_TO_ID).values

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        text = str(self.content[index])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,    # Pad & truncate
            padding='max_length',       # Pad to max_len
            truncation=True,            # Truncate to max_len
            return_attention_mask=True, # Return attention mask
            return_tensors='pt',        # Return PyTorch tensors
        )

        item = {
            'text': text, # Keep original text for inspection if needed
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.include_labels:
             # Ensure label exists and is valid before converting
            label = self.labels[index]
            if label not in config.ID_TO_EMOTION:
                 raise ValueError(f"Invalid label ID '{label}' found at index {index}. Check EMOTION_TO_ID mapping and data.")
            item['labels'] = torch.tensor(label, dtype=torch.long)


        return item