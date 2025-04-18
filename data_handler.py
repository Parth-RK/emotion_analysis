import warnings
warnings.filterwarnings("ignore")

import os
import spacy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords as nltk_stopwords
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
import config
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

PAD_IDX = config.PAD_IDX
UNK_IDX = config.UNK_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX

SPACY_MODEL = config.SPACY_MODEL

class Vocabulary:
    def __init__(self, freq_threshold, max_size=None):
        self.itos = {PAD_IDX: config.PAD_TOKEN, UNK_IDX: config.UNK_TOKEN,
                     SOS_IDX: config.SOS_TOKEN, EOS_IDX: config.EOS_TOKEN}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        print("Building vocabulary...")
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in tqdm(sentence_list, desc="Counting Frequencies"):
            frequencies.update(sentence)

        if self.max_size is not None:
            limited_freq = frequencies.most_common(self.max_size - len(self.itos))
            frequencies = Counter(dict(limited_freq))

        for word, freq in tqdm(frequencies.items(), desc="Creating Mappings"):
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocabulary built. Size: {len(self.itos)}")

    def numericalize(self, text_tokens):
        return [self.stoi.get(token, UNK_IDX) for token in text_tokens]

    def save(self, filepath, n_class):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'stoi': self.stoi,
            'freq_threshold': self.freq_threshold,
            'n_class': n_class
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        print(f"Vocabulary (stoi) and n_class saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        stoi_loaded = loaded_data['stoi']
        freq_threshold = loaded_data.get('freq_threshold', config.MIN_FREQ)
        n_class = loaded_data.get('n_class')
        if n_class is None:
             raise ValueError("Number of classes (n_class) not found in vocabulary file.")

        vocab = cls(freq_threshold)
        itos_rebuilt = {}
        stoi_rebuilt = {}
        special_tokens = {config.PAD_TOKEN, config.UNK_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN}
        for token, index_str in stoi_loaded.items():
            index = int(index_str)
            itos_rebuilt[index] = token
            stoi_rebuilt[token] = index
            
        vocab.itos = itos_rebuilt
        vocab.stoi = stoi_rebuilt

        print(f"Vocabulary loaded from {filepath}. Size: {len(vocab.itos)}, n_class: {n_class}")
        return vocab, n_class

class TextPreprocessor:
    def __init__(self, use_stopwords=False):
        self.nlp = None
        self.stopwords = set(nltk_stopwords.words('english')) if use_stopwords else set()
        self._lazy_load_spacy()
        print(f"TextPreprocessor initialized. Stopwords {'enabled' if use_stopwords else 'disabled'}.")
        print("Dependency parser is ENABLED for negation handling.")

    def _lazy_load_spacy(self):
        if self.nlp is None:
            print(f"Loading spaCy model '{SPACY_MODEL}'...")
            try:
                self.nlp = spacy.load(SPACY_MODEL, disable=["ner"])
            except OSError:
                print(f"Spacy model '{SPACY_MODEL}' not found. Downloading...")
                spacy.cli.download(SPACY_MODEL)
                self.nlp = spacy.load(SPACY_MODEL, disable=["ner"])
            print("spaCy model loaded (with parser).")

    def clean_and_tokenize(self, text):
        text = str(text).lower()
        doc = self.nlp(text)
        tokens = []
        negated_indices = set()

        for token in doc:
            if token.dep_ == 'neg':
                head = token.head
                negated_indices.add(head.i)

        for token in doc:
            is_negated = token.i in negated_indices
            if (not token.is_stop and        # spaCy's default stop words
                not token.is_punct and       # Punctuation
                not token.is_space and       # Whitespace tokens
                token.lemma_ not in self.stopwords):

                lemma = token.lemma_
                if is_negated:
                    lemma += "_NEG"
                tokens.append(lemma)

        return tokens

    def preprocess_dataframe(self, df, text_column='text'):
        if text_column not in df.columns:
             raise ValueError(f"Input DataFrame must contain a '{text_column}' column.")
        df[text_column] = df[text_column].fillna('')

        print(f"Preprocessing DataFrame column '{text_column}'...")
        processed_texts = [self.clean_and_tokenize(text) for text in tqdm(df[text_column], desc="Processing Texts")]
        print("Preprocessing Done!")
        return processed_texts

class EmotionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        if len(self.sequences) != len(self.labels):
             raise ValueError("Sequences and labels must have the same length!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.long)
        text_list.append(processed_text)
        lengths.append(len(processed_text))

    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(label_list)
    return padded_texts, labels

def create_label_mappings(train_df, label_column='label'):
    unique_labels = sorted(train_df[label_column].astype(str).unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    print(f"Created mappings for {len(unique_labels)} unique string labels: {unique_labels}")
    return label_to_int, int_to_label

def create_placeholder_mappings(train_df, label_column='label'):
    unique_labels = sorted(train_df[label_column].unique())
    int_to_label = {i: f"label_{i}" for i in unique_labels}
    label_to_int = {v: k for k, v in int_to_label.items()}
    print(f"Using existing integer labels. Created placeholder mappings for {len(unique_labels)} labels.")
    print(f"Placeholder int_to_label map: {int_to_label}")
    return label_to_int, int_to_label

def to_native(obj):
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except ValueError:
             return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def save_label_map(int_to_label, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    str_keyed_map = {str(k): v for k, v in int_to_label.items()}
    
    with open(filepath, 'w') as f:
        json.dump(str_keyed_map, f, indent=4)
    print(f"Label map saved to {filepath}")

def load_and_prepare_data(train_path, val_path, test_path, label_map_save_path):
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Raw data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

        label_column = 'label'
        for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            if 'text' not in df.columns or label_column not in df.columns:
                raise ValueError(f"{df_name} DataFrame is missing 'text' or '{label_column}' column.")
            if df[label_column].isnull().any():
                print(f"Warning: Found NaN values in '{label_column}' of {df_name} data. Dropping rows.")
                df.dropna(subset=[label_column], inplace=True)
            if 'text' in df.columns:
                 df['text'] = df['text'].astype(str)


        label_to_int, int_to_label = None, None
        n_class = train_df[label_column].nunique()

        if ptypes.is_integer_dtype(train_df[label_column]):
            print(f"Detected integer labels in '{label_column}' column. Using them directly. n_class={n_class}")
            for df_name, df in [('Validation', val_df), ('Test', test_df)]:
                 if not ptypes.is_integer_dtype(df[label_column]):
                      try:
                           df[label_column] = df[label_column].astype(int)
                           print(f"Converted '{label_column}' in {df_name} to integer.")
                      except (ValueError, TypeError):
                           raise TypeError(f"Training labels are integers, but {df_name} labels in column '{label_column}' are not and cannot be converted to integer.")
            if not os.path.exists(label_map_save_path):
                print(f"Creating placeholder label map for integer labels at {label_map_save_path}")
                _, int_to_label = create_placeholder_mappings(train_df, label_column)
                save_label_map(int_to_label, label_map_save_path)

        elif ptypes.is_string_dtype(train_df[label_column]) or ptypes.is_object_dtype(train_df[label_column]):
            print(f"Detected string/object labels in '{label_column}' column. Creating mappings. n_class={n_class}")
            label_to_int, int_to_label = create_label_mappings(train_df, label_column)
            n_class = len(label_to_int)

            print("Mapping string labels to integers for all datasets...")
            for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
                original_labels = set(df[label_column].unique())
                df[label_column] = df[label_column].map(label_to_int)
                if df[label_column].isnull().any():
                    unmapped_labels = original_labels - set(label_to_int.keys())
                    print(f"Warning: Found labels in {df_name} set not present in training data mapping: {unmapped_labels}. Dropping rows with these unmappable labels.")
                    df.dropna(subset=[label_column], inplace=True)
                df[label_column] = df[label_column].astype(int)

            save_label_map(int_to_label, label_map_save_path)

        else:
             raise TypeError(f"Unsupported label type '{train_df[label_column].dtype}' in column '{label_column}'. Labels must be integers or strings.")

        print_data_summary(train_df, "Train", label_column)
        print_data_summary(val_df, "Validation", label_column)
        print_data_summary(test_df, "Test", label_column)
        print(f"Labels processed. '{label_column}' column now contains integer indices.")
        print(f"Final determined number of classes (n_class): {n_class}")

        return train_df, val_df, test_df, label_to_int, int_to_label, n_class

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Check file paths in config.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preparation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
def plot_class_distribution(df, label_column='label', title="Class Distribution", save_path=None):
    """Plots the distribution of classes in a DataFrame."""
    plt.figure(figsize=(10, 6))
    class_counts = df[label_column].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
    plt.show()

def plot_sequence_lengths(token_lists, title="Sequence Length Distribution (Before Padding)", save_path=None):
    """Plots the distribution of sequence lengths."""
    lengths = [len(tokens) for tokens in token_lists]
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50, kde=True)
    avg_len = np.mean(lengths)
    med_len = np.median(lengths)
    max_len = np.max(lengths)
    plt.title(f"{title}\nAvg: {avg_len:.2f}, Median: {med_len:.0f}, Max: {max_len:.0f}")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.axvline(avg_len, color='r', linestyle='dashed', linewidth=1, label=f'Avg Len ({avg_len:.2f})')
    plt.axvline(config.MAX_LENGTH, color='g', linestyle='dashed', linewidth=1, label=f'MAX_LENGTH ({config.MAX_LENGTH})')
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Sequence length plot saved to {save_path}")
    plt.show()

def print_data_summary(df, name, label_column='label'):
    """Prints a summary of the dataframe."""
    print(f"\n--- {name} Data Summary ---")
    print(f"Shape: {df.shape}")
    if label_column in df.columns:
        print("Label Distribution:")
        print(df[label_column].value_counts(normalize=True) * 100)
    else:
        print("Label column not found.")
    print("-" * (len(name) + 18))