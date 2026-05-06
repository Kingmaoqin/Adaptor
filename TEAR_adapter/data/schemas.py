from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROLE_NAMES: Tuple[str, ...] = (
    "confounding",
    "intervention",
    "outcome",
    "proxy",
    "remainder",
)
ORACLE_ROLE_NAMES: Tuple[str, ...] = (
    "confounding",
    "intervention",
    "outcome",
    "proxy",
    "post_intervention",
)


@dataclass
class FeatureMetadata:
    feature_index: int
    feature_name: str
    description: str
    oracle_role: str
    relative_time: float
    measurement_window: str
    post_intervention_keyword: float
    missingness_pre_index: float
    always_missing_pre_index: float


@dataclass
class SyntheticDataset:
    covariates: np.ndarray
    intervention: np.ndarray
    outcome: np.ndarray
    potential_outcomes: np.ndarray
    role_labels: List[str]
    metadata: List[FeatureMetadata]
    latent_acuity: np.ndarray
    intervention_logits: np.ndarray
    intervention_probabilities: np.ndarray
    oracle_effects: np.ndarray


@dataclass
class AuditFeatureSummary:
    feature_index: int
    feature_name: str
    weighted_smd: float
    attribution_intervention: float
    attribution_outcome: float
    relative_time: float
    oracle_role: Optional[str] = None


@dataclass
class OverlapSummary:
    quantiles_by_level: Dict[int, List[float]]
    minimum_propensity: float
    trim_fraction: float


@dataclass
class AuditPacket:
    feature_summaries: List[AuditFeatureSummary]
    overlap_summary: OverlapSummary
    exemplar_indices: List[int]
    cycle_index: int


@dataclass
class RolePriorSuggestion:
    role_prior: np.ndarray
    ordering_pairs: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class AuditSuggestion:
    role_prior: np.ndarray
    exclusion_strength: np.ndarray
    ordering_pairs: List[Tuple[int, int]]
    unit_risk: np.ndarray
    confidence: float
    source: str = "unknown"
    updated_feature_count: int = 0


def metadata_to_matrix(metadata: Sequence[FeatureMetadata]) -> np.ndarray:
    rows = []
    for item in metadata:
        rows.append(
            [
                item.relative_time,
                item.post_intervention_keyword,
                item.missingness_pre_index,
                item.always_missing_pre_index,
                1.0 if item.measurement_window == "baseline" else 0.0,
                1.0 if item.measurement_window == "ambiguous" else 0.0,
                1.0 if item.measurement_window == "post_index" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)
