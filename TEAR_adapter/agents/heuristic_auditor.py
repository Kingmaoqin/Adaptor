from __future__ import annotations

import numpy as np

from ..data.schemas import AuditPacket, AuditSuggestion


class HeuristicAuditor:
    def suggest(
        self,
        packet: AuditPacket,
        current_role_prior: np.ndarray,
        metadata_gate: np.ndarray,
        unit_risk_template: np.ndarray,
    ) -> AuditSuggestion:
        updated_role_prior = current_role_prior.copy()
        exclusion_strength = np.zeros(current_role_prior.shape[0], dtype=np.float32)
        for feature in packet.feature_summaries:
            role_prior = updated_role_prior[feature.feature_index]
            if feature.weighted_smd > 0.15 and feature.relative_time >= 0.0:
                role_prior[:] = np.asarray([0.01, 0.08, 0.08, 0.08, 0.75], dtype=np.float32)
            elif feature.weighted_smd > 0.10 and feature.attribution_intervention > feature.attribution_outcome:
                role_prior[:] = np.asarray([0.05, 0.65, 0.10, 0.10, 0.10], dtype=np.float32)
            elif feature.weighted_smd > 0.10 and feature.attribution_outcome >= feature.attribution_intervention:
                role_prior[:] = np.asarray([0.10, 0.10, 0.55, 0.15, 0.10], dtype=np.float32)
            elif feature.weighted_smd < 0.10 and feature.relative_time < 0.0:
                role_prior[:] = np.asarray([0.60, 0.10, 0.10, 0.10, 0.10], dtype=np.float32)
            updated_role_prior[feature.feature_index] = role_prior / role_prior.sum()
            exclusion_strength[feature.feature_index] = min(float(feature.weighted_smd), float(1.0 - metadata_gate[feature.feature_index]))
        confidence = max(0.0, 1.0 - min(packet.overlap_summary.trim_fraction * 2.0, 0.8))
        return AuditSuggestion(
            role_prior=updated_role_prior,
            exclusion_strength=exclusion_strength,
            ordering_pairs=[],
            unit_risk=unit_risk_template.astype(np.float32),
            confidence=float(confidence),
            source="heuristic",
            updated_feature_count=len(packet.feature_summaries),
        )
