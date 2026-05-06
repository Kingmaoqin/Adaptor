from __future__ import annotations

from typing import List

import numpy as np

from ..data.schemas import FeatureMetadata, RolePriorSuggestion
from .llm_client import StructuredLLMClient
from .prompts import ROLE_PRIOR_SYSTEM_PROMPT, ROLE_TEMPLATE_LIBRARY, build_role_prior_prompt


class RolePriorAgent:
    def __init__(self, llm_client: StructuredLLMClient | None = None, max_ordering_pairs: int = 500):
        self.llm_client = llm_client or StructuredLLMClient()
        self.max_ordering_pairs = max_ordering_pairs

    def suggest(self, metadata: List[FeatureMetadata], use_llm: bool) -> RolePriorSuggestion:
        if use_llm and self.llm_client.available():
            try:
                response = self.llm_client.chat_json(
                    ROLE_PRIOR_SYSTEM_PROMPT,
                    build_role_prior_prompt(metadata, max_ordering_pairs=self.max_ordering_pairs),
                    max_tokens=1200,
                )
                role_prior = self._decode_role_prior_response(response, metadata)
                ordering_pairs = [
                    (int(left), int(right))
                    for left, right in response.get("ordering_pairs", [])[: self.max_ordering_pairs]
                ]
                return RolePriorSuggestion(role_prior=role_prior, ordering_pairs=ordering_pairs)
            except Exception:
                return self._heuristic_prior(metadata)
        return self._heuristic_prior(metadata)

    def _decode_role_prior_response(self, response: dict, metadata: List[FeatureMetadata]) -> np.ndarray:
        if "role_prior" in response:
            role_prior = np.asarray(response["role_prior"], dtype=np.float32)
            if role_prior.shape == (len(metadata), 5):
                return role_prior / np.maximum(role_prior.sum(axis=1, keepdims=True), 1e-8)
        role_prior = np.zeros((len(metadata), 5), dtype=np.float32)
        feature_roles = response.get("feature_roles", [])
        assigned_indices = set()
        for update in feature_roles:
            if isinstance(update, dict):
                feature_index = int(update["feature_index"])
                role_name = str(update["role_name"])
            else:
                feature_index = int(update[0])
                role_name = str(update[1])
            role_name = self._expand_role_name(role_name)
            template = np.asarray(ROLE_TEMPLATE_LIBRARY.get(role_name, ROLE_TEMPLATE_LIBRARY["remainder"]), dtype=np.float32)
            if 0 <= feature_index < len(metadata):
                role_prior[feature_index] = template / template.sum()
                assigned_indices.add(feature_index)
        if len(assigned_indices) != len(metadata):
            heuristic = self._heuristic_prior(metadata).role_prior
            for feature_index in range(len(metadata)):
                if feature_index not in assigned_indices:
                    role_prior[feature_index] = heuristic[feature_index]
        return role_prior

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

    def _heuristic_prior(self, metadata: List[FeatureMetadata]) -> RolePriorSuggestion:
        role_prior = np.zeros((len(metadata), 5), dtype=np.float32)
        ordering_pairs = []
        for item in metadata:
            base = np.asarray([0.25, 0.20, 0.20, 0.20, 0.15], dtype=np.float32)
            if item.relative_time >= 0.0 or item.post_intervention_keyword > 0.5:
                base = np.asarray([0.02, 0.08, 0.10, 0.10, 0.70], dtype=np.float32)
            elif item.measurement_window == "baseline":
                base = np.asarray([0.50, 0.10, 0.15, 0.15, 0.10], dtype=np.float32)
            elif item.measurement_window == "ambiguous":
                base = np.asarray([0.20, 0.25, 0.15, 0.25, 0.15], dtype=np.float32)
            role_prior[item.feature_index] = base / base.sum()
        pre_indices = [item.feature_index for item in metadata if item.relative_time < 0]
        post_indices = [item.feature_index for item in metadata if item.relative_time >= 0]
        for pre_idx in pre_indices:
            for post_idx in post_indices:
                ordering_pairs.append((pre_idx, post_idx))
                if len(ordering_pairs) >= self.max_ordering_pairs:
                    break
            if len(ordering_pairs) >= self.max_ordering_pairs:
                break
        return RolePriorSuggestion(role_prior=role_prior, ordering_pairs=ordering_pairs)
