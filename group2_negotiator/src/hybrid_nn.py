import torch
import torch.nn as nn

from src.features import FEATURE_COLUMNS

N_FEATURES = len(FEATURE_COLUMNS)


class ConcessionNet(nn.Module):
    # Predicts the concession exponent e from domain/profile features
    # Softplus ensures output is always positive (e > 0)
    def __init__(self, n_features=N_FEATURES, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
