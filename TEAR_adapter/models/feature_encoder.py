from __future__ import annotations

import torch
from torch import nn


class FeatureEmbeddingEncoder(nn.Module):
    def __init__(self, num_features: int, metadata_dim: int, embedding_dim: int):
        super().__init__()
        self.value_scale = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.05)
        self.value_bias = nn.Parameter(torch.zeros(num_features, embedding_dim))
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.output_normalization = nn.LayerNorm(embedding_dim)

    def forward(self, covariates: torch.Tensor, metadata_matrix: torch.Tensor) -> torch.Tensor:
        metadata_embedding = self.metadata_projection(metadata_matrix).unsqueeze(0)
        value_embedding = covariates.unsqueeze(-1) * self.value_scale.unsqueeze(0) + self.value_bias.unsqueeze(0)
        return self.output_normalization(value_embedding + metadata_embedding)

