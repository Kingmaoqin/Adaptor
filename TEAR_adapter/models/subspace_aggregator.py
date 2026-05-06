from __future__ import annotations

import torch
from torch import nn


class RoleSubspaceAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_embeddings: torch.Tensor, role_probabilities: torch.Tensor) -> torch.Tensor:
        return torch.einsum("fr,bfd->brd", role_probabilities, feature_embeddings)

