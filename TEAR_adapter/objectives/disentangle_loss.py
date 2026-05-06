"""
Routing Disentanglement Losses
===============================
Two losses to break the degenerate equilibrium where all role subspaces
are identical (routing entropy ≈ max):

1. Entropy penalty: minimizes H(softmax(role_logits)) within each partition
   (visible {C,I,Y} and hidden {P,R}), forcing sharper routing.

2. Adversarial disentanglement via gradient reversal:
   - z_out should NOT predict treatment → adversarial T-head on z_out
   - z_int should NOT predict outcome  → adversarial Y-head on z_int
   This guides the model on WHICH role each feature should be assigned to.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ── Gradient Reversal Layer (Ganin et al. 2016) ────────────────────────────

class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal: forward = identity, backward = -alpha * grad."""
    return _GradientReversalFn.apply(x, alpha)


# ── Routing Entropy Penalty ─────────────────────────────────────────────────

def routing_entropy_loss(role_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute average entropy of the within-partition softmax distributions.

    Parameters
    ----------
    role_logits : (p, 5)
        Raw logits from the router (before eligibility gating).

    Returns
    -------
    loss : scalar
        Mean entropy across features and partitions. Minimizing this
        sharpens the role assignment within {C,I,Y} and {P,R}.
    """
    # Visible partition: roles 0,1,2 (C,I,Y)
    vis_probs = torch.softmax(role_logits[:, :3], dim=-1)  # (p, 3)
    ent_vis = -(vis_probs * torch.log(vis_probs + 1e-8)).sum(dim=-1).mean()

    # Hidden partition: roles 3,4 (P,R)
    hid_probs = torch.softmax(role_logits[:, 3:], dim=-1)  # (p, 2)
    ent_hid = -(hid_probs * torch.log(hid_probs + 1e-8)).sum(dim=-1).mean()

    return ent_vis + ent_hid


# ── Adversarial Disentanglement Heads ───────────────────────────────────────

class AdversarialDisentangleHeads(nn.Module):
    """
    Two adversarial heads:
      adv_T_from_out: predicts treatment from z_out (should FAIL → z_out is T-free)
      adv_Y_from_int: predicts outcome from z_int  (should FAIL → z_int is Y-free)

    During forward pass: heads try to minimize prediction error (standard).
    Gradient reversal on the representation: pushes z_out/z_int to be
    LESS informative about the "wrong" target.
    """

    def __init__(self, embedding_dim: int, num_levels: int):
        super().__init__()
        self.adv_T_from_out = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, num_levels),
        )
        self.adv_Y_from_int = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(
        self,
        z_int: torch.Tensor,
        z_out: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns combined adversarial loss.

        The gradient reversal ensures:
          - z_out learns to NOT encode treatment information
          - z_int learns to NOT encode outcome information
        """
        # z_out should not predict T
        z_out_rev = gradient_reversal(z_out, alpha)
        loss_adv_T = F.cross_entropy(self.adv_T_from_out(z_out_rev), T)

        # z_int should not predict Y
        z_int_rev = gradient_reversal(z_int, alpha)
        loss_adv_Y = F.mse_loss(self.adv_Y_from_int(z_int_rev).squeeze(-1), Y)

        return loss_adv_T + loss_adv_Y
