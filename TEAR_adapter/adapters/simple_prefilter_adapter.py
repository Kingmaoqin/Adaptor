from __future__ import annotations

from typing import List

import numpy as np

from ..data.schemas import FeatureMetadata
from .base import AdapterOutput, BaseAdapter


class SimplePrefilterAdapter(BaseAdapter):
    """
    Baseline: deterministic pre-index filter using metadata only.
    A feature is included in the adjustment set iff:
        relative_time < 0  AND  post_intervention_keyword < 0.5

    adjusted_features = X[:, eligible_mask]  (hard filter, no learned routing)
    Role probabilities are deterministic from metadata (no training).
    """

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "SimplePrefilterAdapter":
        self._eligible_mask = np.array(
            [
                (m.relative_time < 0.0 and m.post_intervention_keyword < 0.5)
                for m in metadata
            ],
            dtype=bool,
        )
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        n, p = X.shape
        gate = self._eligible_mask.astype(np.float32)

        # Adjustment set: eligible features only
        adj = X * gate[np.newaxis, :]  # broadcast mask; zero out ineligible cols

        # Role probabilities: hard assignment from eligibility
        role_probs = np.zeros((p, 5), dtype=np.float32)
        role_probs[self._eligible_mask, 0] = 1.0    # confounding
        role_probs[~self._eligible_mask, 4] = 1.0   # remainder

        zero = np.zeros((n, p), dtype=np.float32)
        return AdapterOutput(
            confounding_repr=adj.astype(np.float32),
            intervention_repr=zero,
            outcome_repr=zero,
            proxy_repr=zero,
            remainder_repr=(X * (1 - gate)[np.newaxis, :]).astype(np.float32),
            role_probabilities=role_probs,
            eligibility_gate=gate,
        )
