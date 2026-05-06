from __future__ import annotations

import torch
from torch import nn


class InterventionAssignmentHead(nn.Module):
    def __init__(self, embedding_dim: int, latent_proxy_dim: int, num_intervention_levels: int):
        super().__init__()
        input_dim = embedding_dim * 2 + latent_proxy_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_intervention_levels),
        )

    def forward(
        self,
        confounding_subspace: torch.Tensor,
        intervention_subspace: torch.Tensor,
        latent_proxy: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([confounding_subspace, intervention_subspace, latent_proxy], dim=-1))


class OutcomeRegressionHead(nn.Module):
    def __init__(self, embedding_dim: int, latent_proxy_dim: int, num_intervention_levels: int):
        super().__init__()
        input_dim = embedding_dim * 2 + latent_proxy_dim + num_intervention_levels
        hidden_dim = max(input_dim, 32)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_intervention_levels = num_intervention_levels

    def forward(
        self,
        confounding_subspace: torch.Tensor,
        outcome_subspace: torch.Tensor,
        latent_proxy: torch.Tensor,
        intervention_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([confounding_subspace, outcome_subspace, latent_proxy, intervention_one_hot], dim=-1)
        return self.network(inputs).squeeze(-1)

    def predict_all_levels(
        self,
        confounding_subspace: torch.Tensor,
        outcome_subspace: torch.Tensor,
        latent_proxy: torch.Tensor,
    ) -> torch.Tensor:
        predictions = []
        identity = torch.eye(self.num_intervention_levels, device=confounding_subspace.device, dtype=confounding_subspace.dtype)
        for level_index in range(self.num_intervention_levels):
            level_one_hot = identity[level_index].unsqueeze(0).expand(confounding_subspace.shape[0], -1)
            predictions.append(
                self.forward(confounding_subspace, outcome_subspace, latent_proxy, level_one_hot)
            )
        return torch.stack(predictions, dim=1)

