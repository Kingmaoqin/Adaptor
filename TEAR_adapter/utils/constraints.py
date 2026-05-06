from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def clip_exclusion_strength(
    exclusion_strength: np.ndarray,
    metadata_gate: np.ndarray,
) -> np.ndarray:
    upper = np.clip(1.0 - metadata_gate, 0.0, 1.0)
    return np.clip(exclusion_strength, 0.0, upper)


def filter_temporal_orderings(
    raw_pairs: Iterable[Tuple[int, int]],
    relative_times: Sequence[float],
    max_pairs: int = 500,
) -> List[Tuple[int, int]]:
    valid_pairs: List[Tuple[int, int]] = []
    for earlier_idx, later_idx in raw_pairs:
        if relative_times[earlier_idx] < 0.0 and relative_times[later_idx] >= -0.5:
            valid_pairs.append((int(earlier_idx), int(later_idx)))
            if len(valid_pairs) >= max_pairs:
                break
    return valid_pairs

