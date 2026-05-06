from __future__ import annotations

import torch
from torch import nn


class SubspaceOrthogonalityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, subspaces: torch.Tensor) -> torch.Tensor:
        centered = subspaces - subspaces.mean(dim=0, keepdim=True)
        total = centered.new_tensor(0.0)
        num_roles = centered.shape[1]
        for left_index in range(num_roles):
            for right_index in range(left_index + 1, num_roles):
                covariance = centered[:, left_index, :].T @ centered[:, right_index, :] / max(centered.shape[0], 1)
                total = total + covariance.pow(2).mean()
        return total

