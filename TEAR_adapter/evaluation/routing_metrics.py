from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..data.schemas import AuditFeatureSummary, AuditPacket
from .attribution import compute_route_weighted_smd
from .overlap_metrics import summarize_overlap


def compute_misrouting_score(role_probabilities: np.ndarray, role_labels: Iterable[str]) -> dict:
    labels = list(role_labels)
    confounding_mass = role_probabilities[:, 0]
    post_mask = np.asarray([label == "post_intervention" for label in labels], dtype=np.float32)
    conf_mask = np.asarray([label == "confounding" for label in labels], dtype=np.float32)
    epsilon_post = float((confounding_mass * post_mask).sum())
    epsilon_miss = float(((1.0 - confounding_mass) * conf_mask).sum())
    return {
        "epsilon_post": epsilon_post,
        "epsilon_miss": epsilon_miss,
        "epsilon_total": epsilon_post + epsilon_miss,
        "post_intervention_recall": float(np.mean(confounding_mass[post_mask == 1.0] < 0.1)) if post_mask.sum() > 0 else 0.0,
    }


def build_audit_packet(
    covariates: np.ndarray,
    intervention: np.ndarray,
    propensity_scores: np.ndarray,
    role_probabilities: np.ndarray,
    metadata: list,
    attribution_intervention: np.ndarray,
    attribution_outcome: np.ndarray,
    cycle_index: int,
    trim_threshold: float,
    top_k: int = 20,
) -> AuditPacket:
    weighted_smd = compute_route_weighted_smd(covariates, intervention, role_probabilities[:, 0])
    top_indices = np.argsort(weighted_smd)[::-1][:top_k]
    feature_summaries: List[AuditFeatureSummary] = []
    for feature_index in top_indices.tolist():
        feature_metadata = metadata[feature_index]
        feature_summaries.append(
            AuditFeatureSummary(
                feature_index=feature_index,
                feature_name=feature_metadata.feature_name,
                weighted_smd=float(weighted_smd[feature_index]),
                attribution_intervention=float(attribution_intervention[feature_index]),
                attribution_outcome=float(attribution_outcome[feature_index]),
                relative_time=float(feature_metadata.relative_time),
                oracle_role=feature_metadata.oracle_role,
            )
        )
    exemplar_indices = np.argsort(propensity_scores.min(axis=1))[: min(10, propensity_scores.shape[0])].tolist()
    return AuditPacket(
        feature_summaries=feature_summaries,
        overlap_summary=summarize_overlap(propensity_scores, trim_threshold=trim_threshold),
        exemplar_indices=exemplar_indices,
        cycle_index=cycle_index,
    )
