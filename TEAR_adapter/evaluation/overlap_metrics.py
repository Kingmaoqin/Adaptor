from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data.schemas import OverlapSummary


def summarize_overlap(propensity_scores: np.ndarray, trim_threshold: float) -> OverlapSummary:
    quantiles_by_level: Dict[int, List[float]] = {}
    for level_index in range(propensity_scores.shape[1]):
        quantiles_by_level[level_index] = np.quantile(
            propensity_scores[:, level_index],
            [0.05, 0.25, 0.50, 0.75, 0.95],
        ).tolist()
    minimum_propensity = float(propensity_scores.min())
    trim_fraction = float((propensity_scores.min(axis=1) < trim_threshold).mean())
    return OverlapSummary(
        quantiles_by_level=quantiles_by_level,
        minimum_propensity=minimum_propensity,
        trim_fraction=trim_fraction,
    )

