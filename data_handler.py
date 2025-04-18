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
        if n_class is None or not isinstance(n_class, int):
             raise ValueError("n_class must be provided as an integer when saving vocabulary.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'stoi': self.stoi,
            'freq_threshold': self.freq_threshold,
            'n_class': n_class
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        print(f"Vocabulary (stoi) and n_class ({n_class}) saved to {filepath}")

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
             raise ValueError(f"Number of classes (n_class) not found in vocabulary file: {filepath}. Artifacts may be incomplete or corrupted.")

        vocab = cls(freq_threshold)
        itos_rebuilt = {int(idx): token for token, idx in stoi_loaded.items()}
        stoi_rebuilt = {token: int(idx) for token, idx in stoi_loaded.items()}

        for idx, token in [(PAD_IDX, config.PAD_TOKEN), (UNK_IDX, config.UNK_TOKEN), (SOS_IDX, config.SOS_TOKEN), (EOS_IDX, config.EOS_TOKEN)]:
             if idx not in itos_rebuilt:
                 itos_rebuilt[idx] = token
                 stoi_rebuilt[token] = idx

        vocab.itos = dict(sorted(itos_rebuilt.items()))
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
            is_negated = token.i in negated_indices or (token.head.i in negated_indices and token.dep_ != 'neg')
            if (token.lemma_ != "-PRON-" and
                not token.is_stop and
                not token.is_punct and
                not token.is_space and
                token.lemma_ not in self.stopwords):
                lemma = token.lemma_
                if is_negated:
                    lemma += "_NEG"
                tokens.append(lemma)
        return tokens

    def preprocess_dataframe(self, df, text_column='text'):
        if text_column not in df.columns:
             raise ValueError(f"Input DataFrame must contain a '{text_column}' column.")
        df[text_column] = df[text_column].fillna('').astype(str)

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
        sequence_data = self.sequences[idx]
        label_data = self.labels[idx]

        if not isinstance(sequence_data, list) or not all(isinstance(i, int) for i in sequence_data):
             raise TypeError(f"Sequence at index {idx} is not a list of integers: {sequence_data}")
        if not isinstance(label_data, (int, np.integer)):
             raise TypeError(f"Label at index {idx} is not an integer: {label_data}")

        sequence = torch.tensor(sequence_data, dtype=torch.long)
        label = torch.tensor(label_data, dtype=torch.long)
        return sequence, label


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_tensor = _label if isinstance(_label, torch.Tensor) else torch.tensor(_label, dtype=torch.long)
        label_list.append(label_tensor)
        text_tensor = _text if isinstance(_text, torch.Tensor) else torch.tensor(_text, dtype=torch.long)
        text_list.append(text_tensor)
        lengths.append(len(text_tensor))
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
    int_to_label = {int(i): f"label_{i}" for i in unique_labels}
    label_to_int = {v: k for k, v in int_to_label.items()}
    print(f"Using existing integer labels. Created placeholder mappings for {len(unique_labels)} labels.")
    print(f"Placeholder int_to_label map: {int_to_label}")
    return label_to_int, int_to_label

def save_label_map(int_to_label, filepath):
    if not isinstance(int_to_label, dict):
        raise TypeError("int_to_label must be a dictionary.")
    if not all(isinstance(k, int) for k in int_to_label.keys()):
         warnings.warn("Keys in int_to_label map are not all integers. Converting keys to string for JSON saving.")
    str_keyed_map = {str(k): v for k, v in int_to_label.items()}

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w') as f:
            json.dump(str_keyed_map, f, indent=4)
        print(f"Label map saved to {filepath}")
    except Exception as e:
        print(f"Error saving label map to {filepath}: {e}")
        raise

def load_label_map(label_map_path):
    if not os.path.exists(label_map_path):
        print(f"Warning: Label map file '{label_map_path}' not found. Returning None.")
        return None
    try:
        with open(label_map_path, 'r') as f:
            str_keyed_map = json.load(f)
        int_to_label = {int(k): v for k, v in str_keyed_map.items()}
        print(f"Label map loaded successfully from {label_map_path}")
        return int_to_label
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {label_map_path}. File might be corrupted.")
        return None
    except Exception as e:
        print(f"Error loading label map from {label_map_path}: {e}")
        return None


def load_and_prepare_data(train_path, val_path, test_path, label_map_path):
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
                 df['text'] = df['text'].fillna('').astype(str)

        train_label_dtype = train_df[label_column].dtype

        label_to_int = None
        int_to_label = None
        n_class = train_df[label_column].nunique()

        if ptypes.is_integer_dtype(train_label_dtype):
            print(f"Detected integer labels in '{label_column}' column (Type: {train_label_dtype}). Using them directly. n_class={n_class}")

            for df_name, df in [('Validation', val_df), ('Test', test_df)]:
                 if not ptypes.is_integer_dtype(df[label_column]):
                      try:
                           df[label_column] = df[label_column].astype(int)
                           print(f"Converted '{label_column}' in {df_name} to integer.")
                      except (ValueError, TypeError) as e:
                           raise TypeError(f"Training labels are integers, but {df_name} labels in column '{label_column}' (Type: {df[label_column].dtype}) could not be converted to integer: {e}")

            if os.path.exists(label_map_path):
                 print(f"Loading existing label map from {label_map_path} for integer labels.")
                 int_to_label = load_label_map(label_map_path)
                 if int_to_label is None:
                     print(f"Failed to load existing map. Creating placeholder map...")
                     _, int_to_label = create_placeholder_mappings(train_df, label_column)
                     save_label_map(int_to_label, label_map_path)
                 else:
                     train_ints = set(train_df[label_column].unique())
                     if set(int_to_label.keys()) != train_ints:
                         print(f"Warning: Mismatch between loaded label map keys {set(int_to_label.keys())} and training data integers {train_ints}. Recreating placeholder map.")
                         _, int_to_label = create_placeholder_mappings(train_df, label_column)
                         save_label_map(int_to_label, label_map_path)
            else:
                print(f"Creating and saving placeholder label map for integer labels at {label_map_path}")
                _, int_to_label = create_placeholder_mappings(train_df, label_column)
                save_label_map(int_to_label, label_map_path)

            label_to_int = {v: k for k, v in int_to_label.items()}


        elif ptypes.is_string_dtype(train_label_dtype) or ptypes.is_object_dtype(train_label_dtype):
            print(f"Detected string/object labels in '{label_column}' column (Type: {train_label_dtype}). Creating mappings. n_class={n_class}")
            label_to_int, int_to_label = create_label_mappings(train_df, label_column)
            n_class = len(label_to_int)

            print("Mapping string labels to integers for all datasets...")
            for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
                original_labels = set(df[label_column].unique())
                df[label_column] = df[label_column].map(label_to_int)
                if df[label_column].isnull().any():
                    unmapped_mask = df[label_column].isnull()
                    unmapped_original_values = set(df.loc[unmapped_mask, label_column].unique())
                    print(f"Warning: Found labels in {df_name} set not present in training data mapping: {unmapped_original_values}. Dropping rows with these unmappable labels.")
                    df.dropna(subset=[label_column], inplace=True)
                df[label_column] = df[label_column].astype(int)

            save_label_map(int_to_label, label_map_path)

        else:
             raise TypeError(f"Unsupported label type '{train_label_dtype}' in column '{label_column}'. Labels must be integers or strings/objects.")

        if int_to_label is None:
            raise RuntimeError(f"Label map (int_to_label) could not be created or loaded. Check data and paths.")

        print_data_summary(train_df, "Train", label_column)
        print_data_summary(val_df, "Validation", label_column)
        print_data_summary(test_df, "Test", label_column)
        print(f"Labels processed. '{label_column}' column now contains integer indices.")
        print(f"Final determined number of classes (n_class): {n_class}")
        print(f"Final integer-to-label mapping: {int_to_label}")

        return train_df, val_df, test_df, label_to_int, int_to_label, n_class

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Check file paths in config.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preparation: {e}")
        import traceback
        traceback.print_exc()
        raise

def plot_class_distribution(df, label_column='label', title="Class Distribution", save_path=None, int_to_label_map=None):
    plt.figure(figsize=(10, 6))

    if int_to_label_map and ptypes.is_integer_dtype(df[label_column]):
        label_series = df[label_column].map(int_to_label_map).fillna('Unknown')
        xlabel = "Class Label (String)"
    else:
        label_series = df[label_column]
        xlabel = "Class Label"

    class_counts = label_series.value_counts()
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")

def plot_sequence_lengths(token_lists, title="Sequence Length Distribution (Before Padding)", save_path=None):
    if not token_lists:
         print("Warning: Cannot plot sequence lengths, token_lists is empty.")
         return
    lengths = [len(tokens) for tokens in token_lists]
    if not lengths:
        print("Warning: Cannot plot sequence lengths, no valid lengths found.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=min(50, max(lengths) if lengths else 1), kde=True)
    avg_len = np.mean(lengths)
    med_len = np.median(lengths)
    max_len = np.max(lengths)
    plt.title(f"{title}\nAvg: {avg_len:.2f}, Median: {med_len:.0f}, Max: {max_len:.0f}")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.axvline(avg_len, color='r', linestyle='dashed', linewidth=1, label=f'Avg Len ({avg_len:.2f})')
    plt.axvline(config.MAX_LENGTH - 2, color='g', linestyle='dashed', linewidth=1, label=f'Effective Max Len ({config.MAX_LENGTH-2})')
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Sequence length plot saved to {save_path}")

def print_data_summary(df, name, label_column='label', int_to_label_map=None):
    print(f"\n--- {name} Data Summary ---")
    print(f"Shape: {df.shape}")
    if label_column in df.columns:
        print("Label Distribution (%):")
        if int_to_label_map and ptypes.is_integer_dtype(df[label_column]):
             label_series = df[label_column].map(int_to_label_map).fillna('Unknown')
             print(label_series.value_counts(normalize=True).mul(100).round(2))
        else:
             print(df[label_column].value_counts(normalize=True).mul(100).round(2))
    else:
        print(f"Label column '{label_column}' not found.")
    print("-" * (len(name) + 21))