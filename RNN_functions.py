import torch
import torch.nn as nn
from Utilities import timeseries_prepare_batch

def rnn_prepare_batch(batch, device):
    return timeseries_prepare_batch(batch, device)

class StandardRNN(nn.Module):
    """
    1-layer tanh RNN with hidden_size, time-series only.
    Expects x shaped (B, T, 1).
    """
    def __init__(self, n_classes: int, hidden_size: int = 11,
                 rho: float = 0.95, dropout_p: float = 0.1, pool: str = "mean"):
        super().__init__()
        assert pool in {"mean", "last"}
        self.hidden_size = int(hidden_size)
        self.rho = float(rho)
        self.pool = pool

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.dropout   = nn.Dropout(dropout_p)
        self.fc        = nn.Linear(self.hidden_size, n_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
                with torch.no_grad():
                    p.mul_(self.rho)
            elif "bias" in name:
                nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        out, _ = self.rnn(x)                    # (B, T, H)
        h = out.mean(dim=1) if self.pool == "mean" else out[:, -1, :]  # (B, H)
        h = self.layernorm(h)
        h = self.dropout(h)
        return self.fc(h)                       # (B, C)
