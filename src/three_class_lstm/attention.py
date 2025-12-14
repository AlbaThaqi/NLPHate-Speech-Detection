import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out, mask=None):
        # lstm_out: (batch, seq_len, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)

        return context, weights

