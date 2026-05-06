import torch.nn as nn


class ContextEncoder(nn.Module):
    """Shared context backbone used by all PPNet heads."""

    def __init__(self, context_dim, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, context):
        return self.mlp(context)
