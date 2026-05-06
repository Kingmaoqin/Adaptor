"""
SharedReprAdapter — T-10
========================
Ablation adapter: uses the same feature encoder as TemporalRole, but
collapses all role-specific subspaces into a single shared representation
via mean pooling (no role routing).

Architecture:
  z_shared_i = (1/p) Σ_j e_ij   ∈ R^(embedding_dim)

This ablation answers: "Does the role routing structure add value, or
is the benefit just from having a learned feature encoder + metadata fusion?"

If SharedReprAdapter ≈ TemporalRole: the router doesn't matter.
If SharedReprAdapter << TemporalRole: role routing is a key contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.schemas import FeatureMetadata, metadata_to_matrix
from ..models.feature_encoder import FeatureEmbeddingEncoder
from ..utils.seed import seed_everything
from .base import AdapterOutput, BaseAdapter
from .temporal_role_adapter import AdapterTrainingConfig


class SharedReprAdapter(BaseAdapter):
    """
    No role routing: all features share a single learned representation.
    Uses the same FeatureEmbeddingEncoder as TemporalRole but without
    the HardEligibilityRoleRouter.

    confounding_repr = mean of all per-feature embeddings (n, embedding_dim)
    All other repr = zeros.

    Parameters
    ----------
    config : AdapterTrainingConfig
        Uses embedding_dim, total_epochs, learning_rate from the same config.
        (lambda_balance, lambda_orthogonality, lambda_gate are ignored.)
    """

    def __init__(self, config: Optional[AdapterTrainingConfig] = None):
        self.config = config or AdapterTrainingConfig()
        self._encoder: Optional[FeatureEmbeddingEncoder] = None
        self._meta_dim: int = 0

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "SharedReprAdapter":
        seed_everything(self.config.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        M = metadata_to_matrix(metadata)  # (p, meta_dim)
        self._meta_dim = M.shape[1]
        n, p = X.shape

        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        M_t = torch.tensor(M, dtype=torch.float32, device=dev)

        encoder = FeatureEmbeddingEncoder(
            num_features=p,
            metadata_dim=self._meta_dim,
            embedding_dim=self.config.embedding_dim,
        ).to(dev)

        # Simple reconstruction target: each embedding predicts x_j variance
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.config.learning_rate)
        # Precompute target: Y_std for outcome regression signal
        Y_t = torch.tensor(outcome, dtype=torch.float32, device=dev).unsqueeze(1)
        I_t = torch.tensor(
            intervention.astype(np.int64), dtype=torch.long, device=dev
        )
        num_K = int(intervention.max()) + 1

        # Simple training: predict outcome and treatment from shared embedding
        out_head = nn.Linear(self.config.embedding_dim, num_K).to(dev)
        treat_head = nn.Linear(self.config.embedding_dim, num_K).to(dev)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(out_head.parameters()) + list(treat_head.parameters()),
            lr=self.config.learning_rate,
        )

        for _ in range(self.config.total_epochs):
            encoder.train()
            optimizer.zero_grad(set_to_none=True)

            # E: (n, p, embedding_dim)
            E = encoder(X_t, M_t)
            # Shared repr: mean over features → (n, embedding_dim)
            z = E.mean(dim=1)

            loss_Y = nn.functional.mse_loss(out_head(z), Y_t.expand(-1, num_K))
            loss_I = nn.functional.cross_entropy(treat_head(z), I_t)
            (loss_Y + loss_I).backward()
            optimizer.step()

        self._encoder = encoder
        self._encoder.eval()
        self._device = dev
        self._M_t = M_t
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        assert self._encoder is not None, "Call fit() before transform()"
        M = metadata_to_matrix(metadata)
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        M_t = torch.tensor(M, dtype=torch.float32, device=self._device)
        n, p = X.shape

        with torch.no_grad():
            E = self._encoder(X_t, M_t)          # (n, p, embedding_dim)
            z = E.mean(dim=1).cpu().numpy()        # (n, embedding_dim)

        zero_p = np.zeros((n, p), dtype=np.float32)
        zero_e = np.zeros((n, self.config.embedding_dim), dtype=np.float32)

        return AdapterOutput(
            confounding_repr=z,
            intervention_repr=zero_e,
            outcome_repr=zero_e,
            proxy_repr=zero_e,
            remainder_repr=zero_e,
            role_probabilities=np.full((p, 5), 0.2, dtype=np.float32),
            eligibility_gate=np.ones(p, dtype=np.float32),
        )
