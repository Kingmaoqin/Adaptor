from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DataSplit:
    covariates: np.ndarray
    intervention: np.ndarray
    outcome: np.ndarray
    potential_outcomes: Optional[np.ndarray] = None


def random_split(
    covariates: np.ndarray,
    intervention: np.ndarray,
    outcome: np.ndarray,
    potential_outcomes: Optional[np.ndarray],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, DataSplit]:
    generator = np.random.default_rng(seed)
    indices = np.arange(len(outcome))
    generator.shuffle(indices)
    test_size = int(len(indices) * test_fraction)
    val_size = int(len(indices) * val_fraction)
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    def take(array: Optional[np.ndarray], selected: np.ndarray) -> Optional[np.ndarray]:
        if array is None:
            return None
        return array[selected]

    return {
        "train": DataSplit(covariates[train_indices], intervention[train_indices], outcome[train_indices], take(potential_outcomes, train_indices)),
        "val": DataSplit(covariates[val_indices], intervention[val_indices], outcome[val_indices], take(potential_outcomes, val_indices)),
        "test": DataSplit(covariates[test_indices], intervention[test_indices], outcome[test_indices], take(potential_outcomes, test_indices)),
    }


def make_kfold_indices(num_samples: int, num_folds: int, seed: int) -> Tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    generator.shuffle(indices)
    return tuple(np.array_split(indices, num_folds))

