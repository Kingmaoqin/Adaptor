"""
Gate Ranking Loss — T-06
========================
Augments BCE gate supervision with a pairwise ranking constraint:
pre-intervention features should have higher gate values than
post-intervention features.

Math:
    L_rank = (1 / |pre||post|) Σ_{j∈pre, k∈post} max(0, margin - h_j + h_k)

Intuition:
    For every (pre, post) feature pair, we want h_pre > h_post + margin.
    If satisfied, contributes 0. If violated, contributes a hinge loss.

This is more efficient than the O(p²) naive pair enumeration:
    L_rank = max(0, margin - mean(h_pre) + mean(h_post))

Parameters
----------
margin : float, default 0.1
    Minimum required gap between average h_pre and h_post.
reduction : str, default 'mean'
    'mean' or 'sum'.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GateRankingLoss(nn.Module):
    """
    Efficiently computes ranking loss using group means:
        L_rank = max(0, margin - mean_{j∈pre}(h_j) + mean_{k∈post}(h_k))

    Parameters
    ----------
    margin : float
        Desired minimum margin between pre and post gate means.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        gate: torch.Tensor,       # (p,) eligibility gate values ∈ [0, 1]
        gate_target: torch.Tensor,  # (p,) binary targets: 1=pre, 0=post
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        gate : (p,) tensor, h_j values
        gate_target : (p,) tensor, 1 if pre-intervention, 0 if post-intervention

        Returns
        -------
        scalar loss
        """
        pre_mask  = gate_target > 0.5   # (p,)
        post_mask = gate_target <= 0.5  # (p,)

        n_pre  = pre_mask.sum().clamp_min(1)
        n_post = post_mask.sum().clamp_min(1)

        h_pre_mean  = gate[pre_mask].sum()  / n_pre
        h_post_mean = gate[post_mask].sum() / n_post

        return torch.clamp(self.margin - h_pre_mean + h_post_mean, min=0.0)
