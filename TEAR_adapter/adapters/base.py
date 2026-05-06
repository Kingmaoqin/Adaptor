from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.schemas import FeatureMetadata


@dataclass
class AdapterOutput:
    """Unified output returned by every adapter after fit_transform."""

    # Role-specific representations, shape (n_samples, embedding_dim each)
    confounding_repr: np.ndarray    # z^(C)
    intervention_repr: np.ndarray   # z^(I)
    outcome_repr: np.ndarray        # z^(Y)
    proxy_repr: np.ndarray          # z^(P)
    remainder_repr: np.ndarray      # z^(R)  post-intervention / ineligible

    # Per-feature diagnostics, shape (n_features,)
    role_probabilities: np.ndarray  # (p, 5)  soft role assignment
    eligibility_gate: np.ndarray    # (p,)    temporal eligibility score

    @property
    def adjusted_features(self) -> np.ndarray:
        """Default downstream input: z^(C) only."""
        return self.confounding_repr

    @property
    def full_repr(self) -> np.ndarray:
        """Extended downstream input: [z^(C) | z^(I) | z^(Y)]."""
        return np.concatenate(
            [self.confounding_repr, self.intervention_repr, self.outcome_repr],
            axis=1,
        )

    def repr_for_estimator(self, output_mode: str = "confounding") -> np.ndarray:
        """
        Select which representation to pass to a downstream estimator.

        output_mode:
          'confounding'  → z^(C)                        (default)
          'full'         → [z^(C), z^(I), z^(Y)]
          'raw_conf_int' → [z^(C), z^(I)]               (propensity-only)
        """
        if output_mode == "full":
            return self.full_repr
        if output_mode == "raw_conf_int":
            return np.concatenate([self.confounding_repr, self.intervention_repr], axis=1)
        return self.confounding_repr


class BaseAdapter(ABC):
    """
    Unified interface for all covariate routing adapters.

    Usage
    -----
    adapter = SomeAdapter(config)
    adapter.fit(X, metadata, intervention, outcome)
    out = adapter.transform(X, metadata)
    # or one-shot:
    out = adapter.fit_transform(X, metadata, intervention, outcome)

    Downstream estimators receive out.adjusted_features (z^C)
    or out.full_repr ([z^C, z^I, z^Y]) depending on their needs.
    """

    def fit_transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> AdapterOutput:
        self.fit(X, metadata, intervention, outcome)
        return self.transform(X, metadata)

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "BaseAdapter":
        ...

    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        ...
