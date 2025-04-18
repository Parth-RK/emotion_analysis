import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        print(f"Initializing LSTM with vocab_size={vocab_size}, embedding_dim={embedding_dim}, "
              f"hidden_dim={hidden_dim}, n_layers={n_layers}, n_class={n_class}, pad_idx={pad_idx}, dropout={dropout_prob}")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout_prob if n_layers > 1 else 0,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_indices):
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped)
        return out