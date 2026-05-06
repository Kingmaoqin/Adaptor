"""
Safe Eligibility Role Router (修改 A)
======================================
修复旧版 HardEligibilityRoleRouter 的 full-mode post-treatment leakage 问题。

旧版问题：
  - 只 gate confounding 路由 (role 0)
  - 当 h_j → 0 时，feature 的质量流向 ALL non-confounding roles，包括 I(1) 和 Y(2)
  - 在 full 模式 [z^C, z^I, z^Y] 下，post-intervention 特征仍可进入下游

修复原理：
  定义 S_vis = {C=0, I=1, Y=2}（下游可见子空间）
       S_hid = {P=3, R=4}（隐藏子空间，不暴露给下游）

  新路由规则：

    对 k ∈ S_vis:
        g_j^k = h_j * (~g_j^k / Σ_{k'∈S_vis} ~g_j^{k'})
              = h_j * vis_normalized_j^k

    对 m ∈ S_hid:
        g_j^m = (1 - h_j) * (~g_j^m / Σ_{m'∈S_hid} ~g_j^{m'})
              = (1 - h_j) * hid_normalized_j^m

  保证：
    (1) Σ_k g_j^k = h_j + (1-h_j) = 1  (probability sum = 1)
    (2) 当 h_j → 0 时，Σ_{k∈S_vis} g_j^k → 0  (no leakage into any visible subspace)
    (3) 当 h_j → 1 时，Σ_{m∈S_hid} g_j^m → 0  (eligible features maximally visible)

旧版保留为 HardEligibilityRoleRouter（legacy ablation，不覆盖）。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# visible role indices (C, I, Y)
_VIS_IDX = slice(0, 3)
# hidden role indices (P, R)
_HID_IDX = slice(3, 5)


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Martins & Astudillo (ICML 2016) sparsemax: argmin over simplex of
    ||p - z||^2. Piecewise-linear, yields exactly-zero components when
    logits are dominated. Drop-in replacement for softmax along `dim`.

    Used for Phase 6 E2 (sparse routing) to give each feature a truly
    sparse role assignment, rather than the always-dense softmax.
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    K = z.size(dim)
    rng_shape = [1] * z.dim()
    rng_shape[dim] = K
    rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype).view(rng_shape)

    cumsum = z_sorted.cumsum(dim=dim)
    # condition: 1 + k * z_(k) > sum_{j<=k} z_(j), k = 1..K
    cond = (rng * z_sorted) > (cumsum - 1)
    k_z = cond.sum(dim=dim, keepdim=True).clamp_min(1)
    # tau = (sum_{j<=k_z} z_(j) - 1) / k_z
    tau = (cumsum.gather(dim, (k_z - 1)) - 1) / k_z.to(cumsum.dtype)
    return (z - tau).clamp_min(0.0)


class SafeEligibilityRoleRouter(nn.Module):
    """
    Safe role router: temporal eligibility gate h_j controls ALL
    downstream-visible subspace routing (C, I, Y), not only confounding.

    Parameters
    ----------
    num_features : int
        Number of features p.
    num_roles : int
        Must be 5: {confounding, intervention, outcome, proxy, remainder}.
    """

    def __init__(
        self,
        num_features: int,
        num_roles: int = 5,
        init_std: float = 0.0,
        routing_mode: str = "softmax",
    ):
        super().__init__()
        assert num_roles == 5, "SafeEligibilityRoleRouter requires exactly 5 roles"
        assert routing_mode in {"softmax", "sparsemax"}
        self.routing_mode = routing_mode
        # init_std > 0 breaks the uniform-softmax saddle point so that entropy
        # and disentanglement losses have non-zero gradients from epoch 1.
        # Sparsemax with init=0 already has nonzero gradient (no saddle), but
        # init_std > 0 still helps break feature-index symmetry.
        if init_std > 0.0:
            self.role_logits = nn.Parameter(torch.randn(num_features, num_roles) * init_std)
        else:
            self.role_logits = nn.Parameter(torch.zeros(num_features, num_roles))

    def forward(
        self,
        eligibility_gate: torch.Tensor,          # (p,)  in [0, 1]
        prior_logits: Optional[torch.Tensor] = None,  # (p, 5) optional additive bias
    ) -> torch.Tensor:
        """
        Returns role probabilities of shape (p, 5).

        g_j^k for k in {C,I,Y} = h_j * (raw[j,k] / sum_{k'∈{C,I,Y}} raw[j,k'])
        g_j^m for m in {P,R}   = (1-h_j) * (raw[j,m] / sum_{m'∈{P,R}} raw[j,m'])
        """
        logits = self.role_logits
        if prior_logits is not None:
            logits = logits + prior_logits

        if self.routing_mode == "sparsemax":
            raw = sparsemax(logits, dim=-1)    # (p, 5), sparse
        else:
            raw = torch.softmax(logits, dim=-1)  # (p, 5), dense
        h = eligibility_gate                  # (p,)

        # --- Visible roles {C, I, Y} (indices 0-2) ---
        vis_raw = raw[:, _VIS_IDX]                                  # (p, 3)
        vis_sum = vis_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8) # (p, 1)
        vis_normalized = vis_raw / vis_sum                           # (p, 3)
        g_vis = h.unsqueeze(-1) * vis_normalized                     # (p, 3)

        # --- Hidden roles {P, R} (indices 3-4) ---
        hid_raw = raw[:, _HID_IDX]                                  # (p, 2)
        hid_sum = hid_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8) # (p, 1)
        hid_normalized = hid_raw / hid_sum                           # (p, 2)
        g_hid = (1.0 - h).unsqueeze(-1) * hid_normalized            # (p, 2)

        return torch.cat([g_vis, g_hid], dim=-1)  # (p, 5), sums to 1


class LegacyHardEligibilityRoleRouter(nn.Module):
    """
    Legacy router — identical to the original HardEligibilityRoleRouter.
    Kept for ablation comparison (old full / old confounding conditions).
    Only gates the confounding role; I/Y subspaces are NOT gated by h_j.
    """

    def __init__(self, num_features: int, num_roles: int = 5):
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
        raw = torch.softmax(logits, dim=-1)
        g_conf = eligibility_gate * raw[:, 0]
        residual = 1.0 - g_conf
        non_conf_raw = raw[:, 1:]
        non_conf_raw = non_conf_raw / non_conf_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        g_non_conf = residual.unsqueeze(-1) * non_conf_raw
        return torch.cat([g_conf.unsqueeze(-1), g_non_conf], dim=-1)
