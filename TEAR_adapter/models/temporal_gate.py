from __future__ import annotations

import torch
from torch import nn


class TemporalEligibilityGate(nn.Module):
    def __init__(self, metadata_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, metadata_matrix: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(metadata_matrix)).squeeze(-1)

