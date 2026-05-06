from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


@dataclass
class AuditEnergyBreakdown:
    route_energy: torch.Tensor
    temporal_energy: torch.Tensor
    support_energy: torch.Tensor
    total_energy: torch.Tensor


class IterativeStructuredAuditEnergy(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        role_probabilities: torch.Tensor,
        role_prior: torch.Tensor,
        confounding_gate: torch.Tensor,
        exclusion_strength: torch.Tensor,
        ordering_pairs: Iterable[Tuple[int, int]],
        unit_risk: torch.Tensor,
        trim_indicator: torch.Tensor,
        confidence: torch.Tensor,
    ) -> AuditEnergyBreakdown:
        route_energy = torch.sum(
            role_probabilities * (role_probabilities.clamp_min(1e-8).log() - role_prior.clamp_min(1e-8).log())
        )
        temporal_energy = torch.sum(exclusion_strength * role_probabilities[:, 0])
        temporal_score = role_probabilities[:, 0] * confounding_gate
        for earlier_index, later_index in ordering_pairs:
            temporal_energy = temporal_energy + torch.relu(
                temporal_score[later_index] - temporal_score[earlier_index] + self.margin
            )
        support_energy = confidence * torch.sum(unit_risk * trim_indicator)
        total_energy = route_energy + temporal_energy + support_energy
        return AuditEnergyBreakdown(
            route_energy=route_energy,
            temporal_energy=temporal_energy,
            support_energy=support_energy,
            total_energy=total_energy,
        )

