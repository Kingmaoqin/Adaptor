from __future__ import annotations

from typing import Iterable, List

import numpy as np


def compute_gate_precision_recall(
    gate_scores: np.ndarray,
    role_labels: Iterable[str],
    threshold: float = 0.2,
) -> dict:
    labels = np.asarray([1 if label == "post_intervention" else 0 for label in role_labels], dtype=np.int64)
    predicted_post = (gate_scores < threshold).astype(np.int64)
    true_positive = int(((predicted_post == 1) & (labels == 1)).sum())
    false_positive = int(((predicted_post == 1) & (labels == 0)).sum())
    false_negative = int(((predicted_post == 0) & (labels == 1)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {"precision": precision, "recall": recall}

