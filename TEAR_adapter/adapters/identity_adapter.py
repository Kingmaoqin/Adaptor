from __future__ import annotations

from typing import List

import numpy as np

from ..data.schemas import FeatureMetadata
from .base import AdapterOutput, BaseAdapter


class IdentityAdapter(BaseAdapter):
    """
    Baseline: passes raw X through unchanged.
    adjusted_features = X (no routing, no gate).
    role_probabilities = uniform (1/5 per role).
    eligibility_gate = 1.0 for all features.
    """

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "IdentityAdapter":
        self._n_features = X.shape[1]
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        n, p = X.shape
        zero = np.zeros((n, p), dtype=np.float32)
        return AdapterOutput(
            confounding_repr=X.astype(np.float32),
            intervention_repr=zero,
            outcome_repr=zero,
            proxy_repr=zero,
            remainder_repr=zero,
            role_probabilities=np.full((p, 5), 0.2, dtype=np.float32),
            eligibility_gate=np.ones(p, dtype=np.float32),
        )
