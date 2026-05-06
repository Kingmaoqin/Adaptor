from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List

import numpy as np

from ..data.schemas import FeatureMetadata
from ..data.splitters import make_kfold_indices
from .orbit_trainer import OrbitTrainer, OrbitTrainingConfig, OrbitVariantConfig


@dataclass
class CrossFitResult:
    potential_outcomes: np.ndarray
    propensity_scores: np.ndarray
    role_probabilities: np.ndarray
    eligibility_gate: np.ndarray
    pairwise_effects: Dict[str, float]
    pairwise_standard_errors: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    audit_histories: List[List[Dict[str, Any]]]


class CrossFittedDREstimator:
    def __init__(self, training_config: OrbitTrainingConfig, num_folds: int = 5):
        self.training_config = training_config
        self.num_folds = num_folds

    def fit_predict(
        self,
        covariates: np.ndarray,
        intervention: np.ndarray,
        outcome: np.ndarray,
        metadata: List[FeatureMetadata],
        role_labels: List[str],
        variant_config: OrbitVariantConfig,
        potential_outcomes: np.ndarray,
    ) -> CrossFitResult:
        folds = make_kfold_indices(len(outcome), self.num_folds, seed=self.training_config.seed)
        num_levels = int(intervention.max()) + 1
        predicted_potential_outcomes = np.zeros((len(outcome), num_levels), dtype=np.float32)
        predicted_propensity = np.zeros((len(outcome), num_levels), dtype=np.float32)
        role_probabilities = []
        eligibility_gates = []
        fold_metrics: List[Dict[str, float]] = []
        audit_histories: List[List[Dict[str, Any]]] = []
        for fold_index, holdout_indices in enumerate(folds):
            train_mask = np.ones(len(outcome), dtype=bool)
            train_mask[holdout_indices] = False
            trainer_config = OrbitTrainingConfig(**self.training_config.__dict__)
            trainer_config.seed = self.training_config.seed + fold_index
            trainer = OrbitTrainer(trainer_config)
            result = trainer.fit(
                train_covariates=covariates[train_mask],
                train_intervention=intervention[train_mask],
                train_outcome=outcome[train_mask],
                metadata=metadata,
                role_labels=role_labels,
                variant_config=variant_config,
                eval_potential_outcomes=potential_outcomes[train_mask],
            )
            role_probabilities.append(result.role_probabilities)
            eligibility_gates.append(result.eligibility_gate)
            audit_histories.append(result.audit_history)
            holdout_predictions = result.predictor.predict(covariates[holdout_indices])
            predicted_potential_outcomes[holdout_indices] = holdout_predictions["potential_outcomes"]
            predicted_propensity[holdout_indices] = holdout_predictions["propensity_scores"]
            fold_metrics.append(result.metrics)

        pairwise_effects: Dict[str, float] = {}
        pairwise_standard_errors: Dict[str, float] = {}
        for left_level, right_level in combinations(range(num_levels), 2):
            mu_left = predicted_potential_outcomes[:, left_level]
            mu_right = predicted_potential_outcomes[:, right_level]
            e_left = np.clip(predicted_propensity[:, left_level], 1e-3, 1.0)
            e_right = np.clip(predicted_propensity[:, right_level], 1e-3, 1.0)
            residual = outcome - predicted_potential_outcomes[np.arange(len(outcome)), intervention]
            pseudo_outcome = (
                (intervention == left_level).astype(np.float32) / e_left
                - (intervention == right_level).astype(np.float32) / e_right
            ) * residual + mu_left - mu_right
            key = f"{left_level}_vs_{right_level}"
            pairwise_effects[key] = float(np.mean(pseudo_outcome))
            pairwise_standard_errors[key] = float(np.std(pseudo_outcome, ddof=1) / np.sqrt(len(pseudo_outcome)))

        return CrossFitResult(
            potential_outcomes=predicted_potential_outcomes,
            propensity_scores=predicted_propensity,
            role_probabilities=np.mean(np.stack(role_probabilities, axis=0), axis=0),
            eligibility_gate=np.mean(np.stack(eligibility_gates, axis=0), axis=0),
            pairwise_effects=pairwise_effects,
            pairwise_standard_errors=pairwise_standard_errors,
            fold_metrics=fold_metrics,
            audit_histories=audit_histories,
        )
