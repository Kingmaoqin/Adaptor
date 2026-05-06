"""
Soft-Mask Adapter (改进方向 1)
===============================
用 routing 学到的 mask 直接过滤原始高维空间 X，
而不是输出低维嵌入 z^(C)。

两种输出模式：
  1. confounding_soft_mask:  X_tilde = X ⊙ g^C
  2. safe_visible_mask:      X_tilde = X ⊙ m_vis, where m_j = Σ_{k∈S_vis} g_j^k

动机：低维 z^(C) (dim=32 from p=100) 可能让 TARNet / DragonNet / RLearner
     "营养不良"。Soft-mask 保留原始特征维度，仅去毒。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

from ..data.schemas import FeatureMetadata, metadata_to_matrix
from ..models.temporal_gate import TemporalEligibilityGate
from ..models.role_router_safe import SafeEligibilityRoleRouter
from ..models.feature_encoder import FeatureEmbeddingEncoder
from ..models.subspace_aggregator import RoleSubspaceAggregator
from ..models.outcome_heads import TreatmentSpecificOutcomeHeads
from ..objectives.balance_loss import LocalizedBalanceLoss
from ..objectives.orthogonality_loss import SubspaceOrthogonalityLoss
from ..utils.seed import seed_everything
from .base import AdapterOutput, BaseAdapter


@dataclass
class SoftMaskAdapterConfig:
    embedding_dim: int = 32
    total_epochs: int = 40
    warmup_fraction: float = 0.25
    learning_rate: float = 1e-3
    lambda_balance: float = 0.5
    lambda_orthogonality: float = 0.1
    lambda_gate: float = 0.5
    device: str = "auto"
    seed: int = 42


class SoftMaskTemporalRoleAdapter(BaseAdapter):
    """
    Trains same safe router as SafeTemporalRoleAdapter,
    but outputs X ⊙ g^C (confounding mask) or X ⊙ m_vis (visible mask)
    instead of low-dim z^(C).

    The routing is still learned via the full encoder + router pipeline;
    the mask is just the per-feature confounding probability from the router.
    """

    def __init__(
        self,
        config: Optional[SoftMaskAdapterConfig] = None,
        gate_source: str = "learned",
    ):
        self.config = config or SoftMaskAdapterConfig()
        self.gate_source = gate_source
        self._core = None
        self._device = None

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "SoftMaskTemporalRoleAdapter":
        seed_everything(self.config.seed)
        device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else "cpu"
        self._device = device
        dev = torch.device(device)

        meta_np = metadata_to_matrix(metadata)
        n, p = X.shape
        num_levels = int(intervention.max()) + 1
        meta_dim = meta_np.shape[1]

        # Reuse _SafeAdapterCore from safe adapter
        from .temporal_role_adapter_safe import _SafeAdapterCore
        core = _SafeAdapterCore(p, meta_dim, num_levels, self.config.embedding_dim,
                                outcome_head_mode="treatment_specific").to(dev)
        optimizer = torch.optim.Adam(core.parameters(), lr=self.config.learning_rate)
        balance_loss_fn = LocalizedBalanceLoss()
        orth_loss_fn = SubspaceOrthogonalityLoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        I_t = torch.tensor(intervention, dtype=torch.long, device=dev)
        Y_t = torch.tensor(outcome, dtype=torch.float32, device=dev)
        M_t = torch.tensor(meta_np, dtype=torch.float32, device=dev)

        gate_target = (
            (M_t[:, 0] < 0.0).float()
            * (M_t[:, 1] < 0.5).float()
            * (M_t[:, 3] < 0.5).float()
            * (M_t[:, 6] < 0.5).float()
        )

        num_warmup = max(1, int(self.config.total_epochs * self.config.warmup_fraction))
        num_prewarm = max(0, int(self.config.total_epochs * 0.10))
        if self.config.lambda_gate > 0 and self.gate_source == "learned" and num_prewarm > 0:
            gate_opt = torch.optim.Adam(core.gate.parameters(), lr=self.config.learning_rate * 2)
            for _ in range(num_prewarm):
                core.train(); gate_opt.zero_grad(set_to_none=True)
                rg = core.gate(M_t)
                F.binary_cross_entropy(rg.clamp(1e-6, 1 - 1e-6), gate_target).backward()
                gate_opt.step()

        for epoch in range(1, self.config.total_epochs + 1):
            core.train(); optimizer.zero_grad(set_to_none=True)
            out = core(X_t, M_t, gate_source=self.gate_source)
            loss_I = F.cross_entropy(out["intervention_logits"], I_t)
            loss_Y = core.outcome_loss(out["z_conf"], out["z_out"], Y_t, I_t)
            loss_bal = balance_loss_fn(out["z_conf"], I_t)
            loss_orth = orth_loss_fn(out["subspaces"])
            total = loss_I + loss_Y
            if epoch > num_warmup:
                total = total + self.config.lambda_balance * loss_bal
                total = total + self.config.lambda_orthogonality * loss_orth
            if self.config.lambda_gate > 0 and self.gate_source == "learned":
                total = total + self.config.lambda_gate * F.binary_cross_entropy(
                    out["raw_gate"].clamp(1e-6, 1 - 1e-6), gate_target)
            total.backward(); optimizer.step()

        self._core = core
        self._metadata = metadata
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> "SoftMaskAdapterOutput":
        assert self._core is not None
        dev = torch.device(self._device)
        meta_np = metadata_to_matrix(metadata)
        self._core.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=dev)
            M_t = torch.tensor(meta_np, dtype=torch.float32, device=dev)
            out = self._core(X_t, M_t, gate_source=self.gate_source)

        rp = out["role_probabilities"].detach().cpu().numpy()  # (p, 5)
        gate = out["eligibility_gate"].detach().cpu().numpy()    # (p,)

        # Confounding mask: g^C per feature
        g_conf = rp[:, 0]                     # (p,)
        # Safe visible mask: sum of C+I+Y
        g_vis  = rp[:, 0:3].sum(axis=1)        # (p,)

        # Masked feature outputs
        X_conf_masked = X * g_conf[np.newaxis, :]   # (n, p) ⊙ (1, p)
        X_vis_masked  = X * g_vis[np.newaxis, :]     # (n, p) ⊙ (1, p)

        return SoftMaskAdapterOutput(
            confounding_repr=out["z_conf"].detach().cpu().numpy(),
            intervention_repr=out["z_int"].detach().cpu().numpy(),
            outcome_repr=out["z_out"].detach().cpu().numpy(),
            proxy_repr=out["z_proxy"].detach().cpu().numpy(),
            remainder_repr=out["z_remainder"].detach().cpu().numpy(),
            role_probabilities=rp,
            eligibility_gate=gate,
            confounding_soft_mask=X_conf_masked,
            safe_visible_mask=X_vis_masked,
        )


class SoftMaskAdapterOutput(AdapterOutput):
    """Extended output with soft-masked original-space representations."""

    def __init__(self, confounding_soft_mask: np.ndarray,
                 safe_visible_mask: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.confounding_soft_mask = confounding_soft_mask
        self.safe_visible_mask = safe_visible_mask

    def repr_for_estimator(self, output_mode: str = "confounding") -> np.ndarray:
        if output_mode == "confounding_soft_mask":
            return self.confounding_soft_mask
        if output_mode == "safe_visible_mask":
            return self.safe_visible_mask
        if output_mode == "safe_full":
            return np.concatenate(
                [self.confounding_repr, self.intervention_repr, self.outcome_repr], axis=1)
        return self.confounding_repr
