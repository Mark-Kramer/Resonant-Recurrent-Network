import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from Utilities import timeseries_prepare_batch

def lstm_prepare_batch(batch, device):
    return timeseries_prepare_batch(batch, device)

class LSTMModel(nn.Module):
    """
    1-layer LSTM, input is always (B, T, 1).
    """

    def __init__(self, n_classes: int, hidden_size: int = 11, pool: str = "mean", dropout_p: float = 0.1):
        super().__init__()
        assert pool in {"mean", "last"}
        self.pool = pool

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, n_classes)

        self._reset_parameters(hidden_size)

    def _reset_parameters(self, H: int):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
                param.data[H:2*H] = 1.0   # forget gate bias (total = 1.0)
            elif "bias_hh" in name:
                nn.init.zeros_(param)


        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        out, _ = self.lstm(x)                      # (B, T, H)
        h = out.mean(dim=1) if self.pool == "mean" else out[:, -1, :]  # (B, H)
        h = self.layernorm(h)
        h = self.dropout(h)
        return self.fc(h)                          # (B, C)