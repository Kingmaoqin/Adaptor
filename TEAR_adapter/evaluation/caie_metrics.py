from __future__ import annotations

import numpy as np


def compute_caie_rmse(predicted_potential_outcomes: np.ndarray, true_potential_outcomes: np.ndarray) -> float:
    pred_effects = predicted_potential_outcomes - predicted_potential_outcomes[:, :1]
    true_effects = true_potential_outcomes - true_potential_outcomes[:, :1]
    return float(np.sqrt(np.mean((pred_effects - true_effects) ** 2)))

