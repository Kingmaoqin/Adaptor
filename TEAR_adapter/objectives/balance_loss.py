from __future__ import annotations

import torch
from torch import nn


class LocalizedBalanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _rbf_kernel(left: torch.Tensor, right: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
        pairwise_distance = torch.cdist(left, right) ** 2
        return torch.exp(-gamma * pairwise_distance)

    def forward(self, confounding_subspace: torch.Tensor, intervention: torch.Tensor) -> torch.Tensor:
        unique_levels = torch.unique(intervention)
        if unique_levels.numel() <= 1:
            return confounding_subspace.new_tensor(0.0)
        total = confounding_subspace.new_tensor(0.0)
        num_pairs = 0
        for left_index in range(unique_levels.numel()):
            for right_index in range(left_index + 1, unique_levels.numel()):
                left_group = confounding_subspace[intervention == unique_levels[left_index]]
                right_group = confounding_subspace[intervention == unique_levels[right_index]]
                if left_group.shape[0] == 0 or right_group.shape[0] == 0:
                    continue
                xx = self._rbf_kernel(left_group, left_group).mean()
                yy = self._rbf_kernel(right_group, right_group).mean()
                xy = self._rbf_kernel(left_group, right_group).mean()
                total = total + xx + yy - 2.0 * xy
                num_pairs += 1
        if num_pairs == 0:
            return confounding_subspace.new_tensor(0.0)
        return total / num_pairs

