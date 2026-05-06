from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np

from ..data.schemas import AuditPacket, FeatureMetadata, ROLE_NAMES


ROLE_PRIOR_SYSTEM_PROMPT = (
    "ORBIT+ role prior agent. JSON only. "
    "Roles: c=confounding, i=intervention, o=outcome, p=proxy, r=remainder. "
    "r means post-index or otherwise ineligible."
)


AUDIT_SYSTEM_PROMPT = (
    "ORBIT+ auditor. JSON only. "
    "Revise only flagged features. "
    "Roles: c=confounding, i=intervention, o=outcome, p=proxy, r=remainder. "
    "Choose one role only for each updated feature."
)


ROLE_GUIDE = {
    "c": "baseline severity, physiology, comorbidity",
    "i": "assignment logic, escalation, readiness",
    "o": "downstream outcome risk",
    "p": "hidden acuity proxy, density, monitoring",
    "r": "post-index response, dose change, recovery",
}


ROLE_TEMPLATE_LIBRARY = {
    "confounding": [0.82, 0.05, 0.05, 0.05, 0.03],
    "intervention": [0.08, 0.74, 0.06, 0.08, 0.04],
    "outcome": [0.10, 0.06, 0.70, 0.10, 0.04],
    "proxy": [0.10, 0.08, 0.08, 0.68, 0.06],
    "remainder": [0.02, 0.05, 0.05, 0.08, 0.80],
}


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _semantic_hints(feature_name: str, description: str) -> list[str]:
    text = f"{feature_name} {description}".lower()
    hints: list[str] = []
    if any(token in text for token in ("post_index", "response", "clearance", "adjustment", "recovery", "after intervention")):
        hints.append("post")
    if any(token in text for token in ("propensity", "readiness", "escalation", "activation", "assignment")):
        hints.append("assign")
    if any(token in text for token in ("mortality", "hypoperfusion", "oxygenation", "metabolic", "coagulation", "organ dysfunction")):
        hints.append("outcome")
    if any(token in text for token in ("proxy", "surrogate", "density", "monitoring", "hidden", "latent")):
        hints.append("proxy")
    if any(token in text for token in ("baseline", "pre_index", "admission", "comorbidity", "apache", "sofa", "creatinine", "lactate")):
        hints.append("base")
    return hints


def _top_role_name(role_prior: Sequence[float]) -> str:
    return ROLE_NAMES[int(np.asarray(role_prior).argmax())]


def build_role_prior_prompt(metadata: Iterable[FeatureMetadata], max_ordering_pairs: int) -> str:
    payload = []
    for item in metadata:
        payload.append(
            [
                item.feature_index,
                item.feature_name,
                _round_float(item.relative_time, 1),
                item.measurement_window,
                _round_float(item.post_intervention_keyword, 1),
                _round_float(item.always_missing_pre_index, 1),
                _semantic_hints(item.feature_name, item.description),
            ]
        )
    return (
        "Input rows=[idx,name,time,window,post_kw,always_missing,hints]. "
        "Choose one role per idx. "
        "Prefer r if time>=0 or window=post_index or hints contain post. "
        "Prefer i for assign cues, o for outcome cues, p for proxy cues, c for baseline severity. "
        f"Return JSON {{\"feature_roles\":[[idx,\"c|i|o|p|r\"],...],\"ordering_pairs\":[[pre,post],...]}} with at most {max_ordering_pairs} ordering_pairs. "
        f"Rows: {json.dumps(payload, separators=(',', ':'))}"
    )


def build_audit_prompt(
    packet: AuditPacket,
    metadata: Sequence[FeatureMetadata],
    current_role_prior: np.ndarray,
    metadata_gate: np.ndarray,
    max_updates: int = 4,
    max_ordering_pairs: int = 64,
) -> str:
    feature_payload = []
    metadata_by_index = {item.feature_index: item for item in metadata}
    for feature in packet.feature_summaries[:max_updates]:
        item = metadata_by_index[feature.feature_index]
        role_prior = current_role_prior[feature.feature_index]
        feature_payload.append(
            [
                feature.feature_index,
                feature.feature_name,
                _round_float(feature.relative_time, 1),
                item.measurement_window,
                _round_float(metadata_gate[feature.feature_index], 3),
                _round_float(feature.weighted_smd, 3),
                _round_float(feature.attribution_intervention, 3),
                _round_float(feature.attribution_outcome, 3),
                _top_role_name(role_prior)[0],
                _semantic_hints(item.feature_name, item.description),
            ]
        )
    return (
        "Rows=[idx,name,time,window,gate,smd,aI,aY,current,hints]. "
        "Revise only rows needing change. "
        "Rules: r for post or time>=0; i for assignment logic; o for downstream risk; p for proxy/density; c for baseline severity. "
        "If smd is high and role should be r, use high exclusion_strength. "
        "Role must be exactly one of \"c\",\"i\",\"o\",\"p\",\"r\". Never use \"|\" or multiple roles. "
        "Do not echo input rows. "
        "Valid example: {\"feature_updates\":[[7,\"r\",0.95,1.0]],\"ordering_pairs\":[],\"unit_risk_scale\":1.0,\"confidence\":0.7}. "
        f"Return JSON {{\"feature_updates\":[[idx,\"c|i|o|p|r\",strength,exclusion],...],\"ordering_pairs\":[[pre,post],...],\"unit_risk_scale\":x,\"confidence\":y}} with at most {max_updates} feature_updates and {max_ordering_pairs} ordering_pairs. "
        f"Overlap={{\"min_prop\":{_round_float(packet.overlap_summary.minimum_propensity, 3)},\"trim\":{_round_float(packet.overlap_summary.trim_fraction, 3)}}}. "
        f"Rows: {json.dumps(feature_payload, separators=(',', ':'))}"
    )
