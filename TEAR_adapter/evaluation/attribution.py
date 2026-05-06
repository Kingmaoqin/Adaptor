from __future__ import annotations

import numpy as np
import torch


def compute_assignment_weight_attribution(value_scale: torch.Tensor, assignment_weight: torch.Tensor) -> np.ndarray:
    scale_summary = value_scale.detach().abs().mean(dim=-1)
    confounding_weight = assignment_weight.detach().abs().mean(dim=0)
    attributions = (scale_summary * confounding_weight.mean()).cpu().numpy()
    attributions /= np.maximum(attributions.sum(), 1e-8)
    return attributions.astype(np.float32)


def compute_route_weighted_smd(covariates: np.ndarray, intervention: np.ndarray, confounding_mass: np.ndarray) -> np.ndarray:
    num_features = covariates.shape[1]
    scores = np.zeros(num_features, dtype=np.float32)
    unique_levels = np.unique(intervention)
    for feature_index in range(num_features):
        max_smd = 0.0
        feature_values = covariates[:, feature_index]
        for left_index in range(len(unique_levels)):
            for right_index in range(left_index + 1, len(unique_levels)):
                left_values = feature_values[intervention == unique_levels[left_index]]
                right_values = feature_values[intervention == unique_levels[right_index]]
                if len(left_values) == 0 or len(right_values) == 0:
                    continue
                pooled_std = np.sqrt(0.5 * (left_values.var() + right_values.var()) + 1e-8)
                max_smd = max(max_smd, abs(left_values.mean() - right_values.mean()) / pooled_std)
        scores[feature_index] = confounding_mass[feature_index] * max_smd
    return scores

