from __future__ import annotations

import numpy as np

from ..data.schemas import AuditPacket, AuditSuggestion, FeatureMetadata


class OracleAuditor:
    role_to_index = {
        "confounding": 0,
        "intervention": 1,
        "outcome": 2,
        "proxy": 3,
        "post_intervention": 4,
    }

    def suggest(
        self,
        packet: AuditPacket,
        metadata: list[FeatureMetadata],
        unit_risk_template: np.ndarray,
    ) -> AuditSuggestion:
        role_prior = np.zeros((len(metadata), 5), dtype=np.float32)
        exclusion_strength = np.zeros(len(metadata), dtype=np.float32)
        ordering_pairs = []
        for item in metadata:
            role_prior[item.feature_index, self.role_to_index[item.oracle_role]] = 1.0
            if item.oracle_role == "post_intervention":
                exclusion_strength[item.feature_index] = 1.0
            if item.relative_time < 0:
                for later_item in metadata:
                    if later_item.relative_time >= 0:
                        ordering_pairs.append((item.feature_index, later_item.feature_index))
        return AuditSuggestion(
            role_prior=role_prior,
            exclusion_strength=exclusion_strength,
            ordering_pairs=ordering_pairs,
            unit_risk=unit_risk_template.astype(np.float32),
            confidence=1.0,
            source="oracle",
            updated_feature_count=len(metadata),
        )
