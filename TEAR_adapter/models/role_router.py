from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class HardEligibilityRoleRouter(nn.Module):
    def __init__(self, num_features: int, num_roles: int):
        super().__init__()
        self.role_logits = nn.Parameter(torch.zeros(num_features, num_roles))

    def forward(
        self,
        eligibility_gate: torch.Tensor,
        prior_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.role_logits
        if prior_logits is not None:
            logits = logits + prior_logits
        raw_probabilities = torch.softmax(logits, dim=-1)
        confounding_probability = eligibility_gate * raw_probabilities[:, 0]
        residual_mass = 1.0 - confounding_probability
        non_confounding_raw = raw_probabilities[:, 1:]
        non_confounding_raw = non_confounding_raw / non_confounding_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        non_confounding_probability = residual_mass.unsqueeze(-1) * non_confounding_raw
        return torch.cat([confounding_probability.unsqueeze(-1), non_confounding_probability], dim=-1)

