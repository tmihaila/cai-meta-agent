import torch
import torch.nn as nn

from src.agents import PORTFOLIO_NAMES
from src.features import FEATURE_COLUMNS

N_FEATURES = len(FEATURE_COLUMNS)
N_AGENTS = len(PORTFOLIO_NAMES)


class AgentScoreNet(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden_size=32, n_agents=N_AGENTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents),
        )

    def forward(self, x):
        return self.net(x)


class AgentScoreNet2Layer(nn.Module):
    def __init__(self, n_features=N_FEATURES, h1=32, h2=16, n_agents=N_AGENTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_agents),
        )

    def forward(self, x):
        return self.net(x)
