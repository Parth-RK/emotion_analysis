import torch
from torch.utils.data import Dataset
from . import config
# Assuming clean_text is now applied *before* creating the dataset instance
# from .preprocess import clean_text # No longer needed here if pre-cleaned

class EmotionDataset(Dataset):
    """
    PyTorch Dataset class for loading and tokenizing emotion data.
    Assumes the input dataframe's 'content' column is already cleaned.
    """
    def __init__(self, dataframe, tokenizer, max_len, include_labels=True):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'content' and potentially 'sentiment' columns.
                                      Assumes 'content' is pre-cleaned.
            tokenizer: HuggingFace tokenizer instance.
            max_len (int): Maximum sequence length.
            include_labels (bool): Whether to expect and process labels ('sentiment' column).
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Assumes 'content' column already contains cleaned text
        self.content = dataframe['content'].values
        self.include_labels = include_labels
        if self.include_labels:
            # Map sentiment strings to IDs
            # Perform mapping here or ensure dataframe passed already has numeric 'label' column
            if 'sentiment' in dataframe.columns:
                # Use the imported config object
                self.labels = dataframe['sentiment'].map(config.EMOTION_TO_ID).values
            elif 'label' in dataframe.columns: # Allow passing dataframe with pre-mapped labels
                 self.labels = dataframe['label'].values
            else:
                raise ValueError("Dataframe must contain either 'sentiment' or 'label' column when include_labels=True")


    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        text = str(self.content[index]) # Text should already be cleaned

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
            'text': text, # Keep original (cleaned) text for inspection if needed
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.include_labels:
            label_id = self.labels[index]
            # Ensure label ID is valid according to config
            # Use the imported config object
            if label_id not in config.ID_TO_EMOTION:
                 raise ValueError(f"Invalid label ID '{label_id}' found at index {index}. Check EMOTION_TO_ID mapping and data.")
            item['labels'] = torch.tensor(label_id, dtype=torch.long)

        return item