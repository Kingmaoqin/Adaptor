"""
Gumbel-Softmax Role Router (改进方向 4)
=========================================
用 Gumbel-Softmax 替代标准 softmax，支持退火硬路由。

三种模式：
  1. soft       — 标准 softmax（等同 SafeEligibilityRoleRouter）
  2. gumbel     — Gumbel-Softmax with temperature annealing
  3. late_hard  — soft 前 50% epochs → annealed Gumbel 后 50%

动机：标准 softmax 分配柔和权重，子空间之间有信息泄露。
     硬路由（或接近硬路由）让每个特征只属于一个子空间，减少 inter-subspace leakage。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


_VIS_IDX = slice(0, 3)
_HID_IDX = slice(3, 5)


class GumbelSafeRoleRouter(nn.Module):
    """
    Safe role router with Gumbel-Softmax option.

    Uses the same S_vis / S_hid partitioning as SafeEligibilityRoleRouter,
    but applies Gumbel-Softmax (optionally annealed) to the raw logits
    before the vis/hid split.

    Parameters
    ----------
    num_features : int
    num_roles : int
        Must be 5.
    tau_start : float
        Initial Gumbel temperature (high = soft).
    tau_end : float
        Final Gumbel temperature (low = hard).
    hard : bool
        If True, use straight-through hard Gumbel in forward pass.
    """

    def __init__(
        self,
        num_features: int,
        num_roles: int = 5,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        hard: bool = False,
    ):
        super().__init__()
        assert num_roles == 5
        self.role_logits = nn.Parameter(torch.zeros(num_features, num_roles))
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.hard = hard
        # Current temperature (updated externally by training loop)
        self._tau: float = tau_start

    def set_temperature(self, progress: float):
        """
        Set temperature based on training progress.

        Parameters
        ----------
        progress : float in [0, 1]
            0 = start of training, 1 = end.
        """
        self._tau = self.tau_start * (self.tau_end / self.tau_start) ** progress

    @property
    def current_tau(self) -> float:
        return self._tau

    def forward(
        self,
        eligibility_gate: torch.Tensor,
        prior_logits: Optional[torch.Tensor] = None,
        use_gumbel: bool = True,
    ) -> torch.Tensor:
        """
        Returns role probabilities of shape (p, 5).
        """
        logits = self.role_logits
        if prior_logits is not None:
            logits = logits + prior_logits

        if use_gumbel and self.training:
            raw = F.gumbel_softmax(logits, tau=self._tau, hard=self.hard, dim=-1)
        else:
            raw = torch.softmax(logits, dim=-1)

        h = eligibility_gate

        vis_raw = raw[:, _VIS_IDX]
        vis_sum = vis_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        vis_normalized = vis_raw / vis_sum
        g_vis = h.unsqueeze(-1) * vis_normalized

        hid_raw = raw[:, _HID_IDX]
        hid_sum = hid_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hid_normalized = hid_raw / hid_sum
        g_hid = (1.0 - h).unsqueeze(-1) * hid_normalized

        return torch.cat([g_vis, g_hid], dim=-1)
