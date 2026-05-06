from __future__ import annotations

import numpy as np

from ..data.schemas import AuditPacket, AuditSuggestion, FeatureMetadata
from .heuristic_auditor import HeuristicAuditor
from .llm_client import StructuredLLMClient
from .oracle_auditor import OracleAuditor
from .prompts import AUDIT_SYSTEM_PROMPT, ROLE_TEMPLATE_LIBRARY, build_audit_prompt


class LLMAuditorAgent:
    def __init__(self, llm_client: StructuredLLMClient | None = None, fallback_to_heuristic: bool = True):
        self.llm_client = llm_client or StructuredLLMClient()
        self.fallback_to_heuristic = fallback_to_heuristic
        self.heuristic_auditor = HeuristicAuditor()
        self.last_successful_suggestion: AuditSuggestion | None = None

    def suggest(
        self,
        packet: AuditPacket,
        current_role_prior: np.ndarray,
        metadata_gate: np.ndarray,
        unit_risk_template: np.ndarray,
        metadata: list[FeatureMetadata],
    ) -> AuditSuggestion:
        if not self.llm_client.available():
            if self.fallback_to_heuristic:
                suggestion = self.heuristic_auditor.suggest(packet, current_role_prior, metadata_gate, unit_risk_template)
                suggestion.source = "heuristic_fallback_unavailable"
                return suggestion
            raise RuntimeError("LLM auditor requested but LOCAL_LLM_ENDPOINT is unavailable.")
        try:
            response = self._request_suggestion(packet, metadata, current_role_prior, metadata_gate)
            role_prior, exclusion_strength, updated_feature_count = self._decode_sparse_updates(
                response=response,
                current_role_prior=current_role_prior,
                metadata_gate=metadata_gate,
            )
            confidence = float(response.get("confidence", 0.5))
            unit_risk_scale = float(response.get("unit_risk_scale", 1.0))
            suggestion = AuditSuggestion(
                role_prior=role_prior,
                exclusion_strength=exclusion_strength,
                ordering_pairs=[(int(left), int(right)) for left, right in response.get("ordering_pairs", [])],
                unit_risk=np.clip(unit_risk_template * unit_risk_scale, 0.0, 1.0).astype(np.float32),
                confidence=max(0.0, min(confidence, 1.0)),
                source="llm",
                updated_feature_count=updated_feature_count,
            )
            self.last_successful_suggestion = suggestion
            return suggestion
        except Exception:
            if self.last_successful_suggestion is not None:
                suggestion = AuditSuggestion(
                    role_prior=self.last_successful_suggestion.role_prior.copy(),
                    exclusion_strength=self.last_successful_suggestion.exclusion_strength.copy(),
                    ordering_pairs=list(self.last_successful_suggestion.ordering_pairs),
                    unit_risk=unit_risk_template.astype(np.float32),
                    confidence=self.last_successful_suggestion.confidence,
                    source="llm_reuse_last_success",
                    updated_feature_count=self.last_successful_suggestion.updated_feature_count,
                )
                return suggestion
            if self.fallback_to_heuristic:
                suggestion = self.heuristic_auditor.suggest(packet, current_role_prior, metadata_gate, unit_risk_template)
                suggestion.source = "heuristic_fallback_error"
                return suggestion
            raise

    def _request_suggestion(
        self,
        packet: AuditPacket,
        metadata: list[FeatureMetadata],
        current_role_prior: np.ndarray,
        metadata_gate: np.ndarray,
    ) -> dict:
        primary_prompt = build_audit_prompt(
            packet,
            metadata=metadata,
            current_role_prior=current_role_prior,
            metadata_gate=metadata_gate,
            max_updates=4,
            max_ordering_pairs=32,
        )
        try:
            return self.llm_client.chat_json(AUDIT_SYSTEM_PROMPT, primary_prompt, max_tokens=160)
        except Exception:
            compact_prompt = build_audit_prompt(
                packet,
                metadata=metadata,
                current_role_prior=current_role_prior,
                metadata_gate=metadata_gate,
                max_updates=2,
                max_ordering_pairs=8,
            )
            return self.llm_client.chat_json(AUDIT_SYSTEM_PROMPT, compact_prompt, max_tokens=96)

    def _decode_sparse_updates(
        self,
        response: dict,
        current_role_prior: np.ndarray,
        metadata_gate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if "role_prior" in response and "exclusion_strength" in response:
            role_prior = np.asarray(response["role_prior"], dtype=np.float32)
            exclusion_strength = np.asarray(response["exclusion_strength"], dtype=np.float32)
            role_prior = role_prior / np.maximum(role_prior.sum(axis=1, keepdims=True), 1e-8)
            return role_prior, exclusion_strength, int(role_prior.shape[0])

        updated_role_prior = current_role_prior.copy()
        exclusion_strength = np.zeros(current_role_prior.shape[0], dtype=np.float32)
        feature_updates = response.get("feature_updates", [])
        updated_feature_count = 0
        for update in feature_updates:
            if isinstance(update, dict):
                feature_index = int(update["feature_index"])
                role_name = str(update.get("role_name", "remainder"))
                update_strength = float(update.get("update_strength", 0.8))
                raw_exclusion = float(update.get("exclusion_strength", 0.0))
            else:
                feature_index = int(update[0])
                role_name = str(update[1]) if len(update) > 1 else "remainder"
                update_strength = float(update[2]) if len(update) > 2 else 0.8
                raw_exclusion = float(update[3]) if len(update) > 3 else 0.0
            if feature_index < 0 or feature_index >= current_role_prior.shape[0]:
                continue
            role_name = self._expand_role_name(role_name)
            template = np.asarray(ROLE_TEMPLATE_LIBRARY.get(role_name, ROLE_TEMPLATE_LIBRARY["remainder"]), dtype=np.float32)
            template = template / template.sum()
            update_strength = float(np.clip(update_strength, 0.0, 1.0))
            updated_role_prior[feature_index] = (
                (1.0 - update_strength) * updated_role_prior[feature_index]
                + update_strength * template
            )
            updated_role_prior[feature_index] = updated_role_prior[feature_index] / max(
                float(updated_role_prior[feature_index].sum()),
                1e-8,
            )
            exclusion_strength[feature_index] = float(np.clip(max(raw_exclusion, 1.0 - metadata_gate[feature_index]), 0.0, 1.0))
            updated_feature_count += 1
        return updated_role_prior, exclusion_strength, updated_feature_count

    @staticmethod
    def _expand_role_name(role_name: str) -> str:
        role_name = role_name.strip().lower()
        if "|" in role_name:
            role_name = role_name.split("|", 1)[0].strip()
        return {
            "c": "confounding",
            "i": "intervention",
            "o": "outcome",
            "p": "proxy",
            "r": "remainder",
        }.get(role_name, role_name)


def get_auditor(mode: str):
    if mode == "heuristic":
        return HeuristicAuditor()
    if mode == "llm":
        return LLMAuditorAgent()
    if mode == "oracle":
        return OracleAuditor()
    return None
