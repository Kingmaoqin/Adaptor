"""
MetadataFilterAdapter — T-11
============================
Rule-based adapter: keeps only pre-intervention features using temporal metadata,
without any learned parameters.

Routing rule:
  - relative_time < 0  (pre-intervention) → confounding_repr  (kept)
  - relative_time >= 0 (post-intervention) → remainder_repr   (zeroed out)
  - relative_time == None / missing        → confounding_repr  (kept by default)

This is the strongest possible "simple" baseline for temporal routing.
It answers: "Does TemporalRole's learned routing add value beyond a hard
time-based rule?"

Expected: MetadataFilterAdapter < TemporalRole when:
  - Post-intervention features carry confounding signal (unusual)
  - The timing boundary is noisy / features near t=0 need soft treatment
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..data.schemas import FeatureMetadata
from .base import AdapterOutput, BaseAdapter


class MetadataFilterAdapter(BaseAdapter):
    """
    Hard temporal filter: drops post-intervention features using metadata.
    No learned parameters — deterministic rule-based routing.

    Parameters
    ----------
    time_threshold : float
        Features with relative_time >= time_threshold are treated as
        post-intervention and zeroed. Default: 0.0 (strict pre/post split).
    keep_ambiguous : bool
        If True (default), features with missing/None relative_time are kept.
        If False, they are zeroed.
    """

    def __init__(self, time_threshold: float = 0.0, keep_ambiguous: bool = True):
        self.time_threshold = time_threshold
        self.keep_ambiguous = keep_ambiguous
        self._pre_mask: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "MetadataFilterAdapter":
        p = X.shape[1]
        self._pre_mask = np.ones(p, dtype=bool)
        for j, m in enumerate(metadata):
            t = getattr(m, "relative_time", None)
            if t is None:
                self._pre_mask[j] = self.keep_ambiguous
            else:
                self._pre_mask[j] = (t < self.time_threshold)
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        assert self._pre_mask is not None, "Call fit() before transform()"
        n, p = X.shape
        X = X.astype(np.float32)

        # Apply mask: keep pre-intervention features in confounding_repr
        conf = X * self._pre_mask[np.newaxis, :].astype(np.float32)  # (n, p)
        rem  = X * (~self._pre_mask[np.newaxis, :]).astype(np.float32)

        zero = np.zeros((n, p), dtype=np.float32)

        # Role probabilities: hard 1.0 assignment
        role_probs = np.zeros((p, 5), dtype=np.float32)
        role_probs[self._pre_mask,  0] = 1.0  # confounding
        role_probs[~self._pre_mask, 4] = 1.0  # remainder (post-intervention)

        # Gate: 1.0 for pre, 0.0 for post
        gate = self._pre_mask.astype(np.float32)

        return AdapterOutput(
            confounding_repr=conf,
            intervention_repr=zero,
            outcome_repr=zero,
            proxy_repr=zero,
            remainder_repr=rem,
            role_probabilities=role_probs,
            eligibility_gate=gate,
        )
