from .attention import Attention
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=200,
            hidden_dim=128,
            num_classes=3,
            num_layers=1,
            embeddings=None,
            freeze_embeddings=True,
            bidirectional=True,
            dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.attention = Attention(lstm_output_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    # def forward(self, input_ids):
    #     """
    #     input_ids: (batch_size, seq_len)
    #     """
    #     embedded = self.embedding(input_ids)
    #     # embedded: (batch_size, seq_len, embed_dim)
    #
    #     lstm_out, _ = self.lstm(embedded)
    #     # lstm_out: (batch_size, seq_len, hidden_dim * directions)
    #
    #     # Attention pooling
    #     context, attn_weights = self.attention(lstm_out)
    #     # context: (batch_size, hidden_dim * directions)
    #
    #     context = self.dropout(context)
    #
    #     # Classification
    #     logits = self.fc(context)
    #     return logits

    def forward(self, x, mask=None, return_attention=False):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        context, attn_weights = self.attention(lstm_out, mask)
        logits = self.fc(context)

        if return_attention:
            return logits, attn_weights
        return logits
