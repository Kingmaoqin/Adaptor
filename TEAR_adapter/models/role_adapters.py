"""
Role-Specific Light Adapters (Phase 6 — Modification I)
========================================================
Attacks the architectural bottleneck identified in Phase 5:
    cos(C, Y) ≈ 0.99 even under strong entropy/disentangle penalties.

Root cause: the shared FeatureEmbeddingEncoder produces one embedding e_j
per feature; all role subspaces are weighted averages of the *same* base
embeddings → necessarily highly correlated regardless of routing weights.

Fix: keep the shared backbone, but add lightweight per-role residual
adapters so each visible role (C, I, Y) receives embeddings in a
*different* direction of representation space.

    e_j^(k) = u_j + α_k * MLP_k(u_j)    for k ∈ {C, I, Y}

where u_j is the shared embedding (n, p, d) and MLP_k is a small bottleneck
(d → d/2 → d). α_k is a learnable scalar gate initialized to 0 so the model
starts from the shared-encoder equilibrium and only diverges when doing so
helps the downstream objective.

Hidden roles P, R keep the shared embedding (no adapter) since they are
never exposed to the downstream estimator.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LightRoleAdapter(nn.Module):
    """
    Lightweight bottleneck MLP with learnable residual scaling.

    alpha_init=0 starts near identity (safe warm start);
    alpha_init=1 starts with the adapter fully active (breaks the shared-
    encoder equilibrium — required when task losses don't push alpha off 0).
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_ratio: int = 2,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        hidden = max(4, embedding_dim // bottleneck_ratio)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding_dim),
        )
        # Small random init on final layer — enough to break symmetry
        nn.init.normal_(self.net[-1].weight, std=0.1)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (n, p, d) shared embeddings → (n, p, d) role-adapted embeddings."""
        return u + self.alpha * self.net(u)


class LinearLowRankRoleAdapter(nn.Module):
    """
    E1-var2: Linear low-rank residual adapter (no nonlinearity).

    e^(k) = u + alpha_k * (U_k V_k^T) u,  U_k in R^{d x r}, V_k in R^{d x r}

    Compared to LightRoleAdapter (MLP with GELU), this variant has no
    nonlinear gating — so it cannot synthesize new representational
    directions outside the linear span of u. Hypothesis: this blocks the
    channel by which adapter_I's MLP leaked Y-predictive directions into
    z^I under E1 (observed ε_post +35%).

    rank r controls capacity; r = d // bottleneck_ratio is the natural
    match to the MLP hidden dim.
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_ratio: int = 2,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        rank = max(4, embedding_dim // bottleneck_ratio)
        self.down = nn.Linear(embedding_dim, rank, bias=False)
        self.up = nn.Linear(rank, embedding_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.1)
        nn.init.normal_(self.up.weight, std=0.1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u + self.alpha * self.up(self.down(u))


class RoleSpecificEncoderStack(nn.Module):
    """
    Bank of 3 light adapters for visible roles {C, I, Y}.
    Hidden roles {P, R} reuse the shared embedding directly.

    Outputs per-role embeddings of shape (n, p, 5, d), ordered
    [C, I, Y, P, R] to match the existing role index convention.

    adapter_type:
      "mlp"            — LightRoleAdapter (E1, bottleneck + GELU)
      "linear_lowrank" — LinearLowRankRoleAdapter (E1-var2, no nonlinearity)
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_ratio: int = 2,
        adapter_type: str = "mlp",
    ):
        super().__init__()
        if adapter_type == "mlp":
            cls = LightRoleAdapter
        elif adapter_type == "linear_lowrank":
            cls = LinearLowRankRoleAdapter
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
        self.adapter_type = adapter_type
        self.adapter_C = cls(embedding_dim, bottleneck_ratio)
        self.adapter_I = cls(embedding_dim, bottleneck_ratio)
        self.adapter_Y = cls(embedding_dim, bottleneck_ratio)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (n, p, d) shared embeddings.
        returns: (n, p, 5, d) with per-role embeddings along role axis.
        """
        e_C = self.adapter_C(u)
        e_I = self.adapter_I(u)
        e_Y = self.adapter_Y(u)
        # P and R reuse the shared embedding (hidden, not exposed downstream)
        return torch.stack([e_C, e_I, e_Y, u, u], dim=2)  # (n, p, 5, d)


def per_role_aggregation(
    role_probs: torch.Tensor,       # (p, 5)
    role_embeddings: torch.Tensor,  # (n, p, 5, d)
) -> torch.Tensor:
    """
    Aggregate per-role embeddings by role weights.

    z^k[i, :] = Σ_j role_probs[j, k] * role_embeddings[i, j, k, :]
    returns: (n, 5, d)
    """
    # Expand role_probs from (p, 5) to (1, p, 5, 1) for broadcasting
    weights = role_probs.unsqueeze(0).unsqueeze(-1)  # (1, p, 5, 1)
    weighted = weights * role_embeddings             # (n, p, 5, d)
    return weighted.sum(dim=1)                       # (n, 5, d)
