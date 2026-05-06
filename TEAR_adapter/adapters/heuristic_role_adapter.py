from __future__ import annotations

from typing import List

import numpy as np

from ..data.schemas import FeatureMetadata
from .base import AdapterOutput, BaseAdapter


class HeuristicRoleAdapter(BaseAdapter):
    """
    Deterministic rule-based role assignment using metadata + data statistics.
    No learned parameters, no LLM.

    Rules (applied in order):
      1. relative_time >= 0 OR post_intervention_keyword >= 0.5
             → remainder  (post-intervention / ineligible)
      2. Among pre-index features: SMD > smd_threshold
             → confounding  (associated with treatment selection = genuine confounder)
      3. Among pre-index features: SMD <= smd_threshold
             → intervention predictor  (not predictive of treatment; exclude from z^C
                                        to avoid diluting the adjustment set)
      4. Otherwise → confounding (conservative default)

    Design rationale: in causal inference, confounders are variables that predict
    BOTH treatment assignment AND outcome. High SMD indicates association with
    treatment — exactly the confounders we want in z^(C). Low-SMD pre-index
    features are essentially noise from a confounding perspective.

    adjusted_features = X masked to confounding columns (hard mask).
    """

    def __init__(self, smd_threshold: float = 0.1):
        self.smd_threshold = smd_threshold
        self._role_labels: np.ndarray | None = None
        self._role_probs: np.ndarray | None = None
        self._gate: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "HeuristicRoleAdapter":
        p = X.shape[1]
        levels = np.unique(intervention)
        smd = self._compute_smd(X, intervention, levels)

        gate = np.array(
            [
                float(m.relative_time < 0.0 and m.post_intervention_keyword < 0.5)
                for m in metadata
            ],
            dtype=np.float32,
        )
        role_probs = np.zeros((p, 5), dtype=np.float32)
        for j, m in enumerate(metadata):
            if gate[j] < 0.5:
                role_probs[j, 4] = 1.0  # remainder (post-intervention / ineligible)
            elif smd[j] > self.smd_threshold:
                role_probs[j, 0] = 1.0  # confounding (high SMD = predicts treatment)
            else:
                role_probs[j, 1] = 1.0  # intervention predictor (low SMD = near-noise)

        self._gate = gate
        self._role_probs = role_probs
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        n, p = X.shape
        conf_mask = self._role_probs[:, 0]   # 1.0 for confounders
        int_mask  = self._role_probs[:, 1]
        out_mask  = self._role_probs[:, 2]
        rem_mask  = self._role_probs[:, 4]

        def _weighted(mask: np.ndarray) -> np.ndarray:
            return (X * mask[np.newaxis, :]).astype(np.float32)

        zero = np.zeros((n, p), dtype=np.float32)
        return AdapterOutput(
            confounding_repr=_weighted(conf_mask),
            intervention_repr=_weighted(int_mask),
            outcome_repr=_weighted(out_mask),
            proxy_repr=zero,
            remainder_repr=_weighted(rem_mask),
            role_probabilities=self._role_probs,
            eligibility_gate=self._gate,
        )

    @staticmethod
    def _compute_smd(
        X: np.ndarray,
        intervention: np.ndarray,
        levels: np.ndarray,
    ) -> np.ndarray:
        """Mean pairwise absolute standardized mean difference per feature."""
        p = X.shape[1]
        global_std = X.std(axis=0) + 1e-8
        smd_max = np.zeros(p, dtype=np.float32)
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                mask_i = intervention == levels[i]
                mask_j = intervention == levels[j]
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    continue
                diff = np.abs(X[mask_i].mean(axis=0) - X[mask_j].mean(axis=0))
                smd_max = np.maximum(smd_max, (diff / global_std).astype(np.float32))
        return smd_max
