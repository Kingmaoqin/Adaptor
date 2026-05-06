from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.schemas import FeatureMetadata, metadata_to_matrix
from ..models.feature_encoder import FeatureEmbeddingEncoder
from ..models.role_router import HardEligibilityRoleRouter
from ..models.subspace_aggregator import RoleSubspaceAggregator
from ..models.temporal_gate import TemporalEligibilityGate
from ..objectives.balance_loss import LocalizedBalanceLoss
from ..objectives.orthogonality_loss import SubspaceOrthogonalityLoss
from ..objectives.gate_ranking_loss import GateRankingLoss
from ..utils.seed import seed_everything
from .base import AdapterOutput, BaseAdapter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdapterTrainingConfig:
    embedding_dim: int = 32
    total_epochs: int = 40
    warmup_fraction: float = 0.25
    learning_rate: float = 1e-3
    lambda_balance: float = 0.5
    lambda_orthogonality: float = 0.1
    lambda_gate: float = 0.1
    lambda_rank: float = 0.0          # T-06: gate ranking loss weight (0 = disabled)
    rank_margin: float = 0.1          # T-06: margin for ranking loss
    gate_pretrain_epochs: int = 0     # T-05: standalone gate pretraining epochs (0 = disabled)
    device: str = "auto"
    seed: int = 42


# ---------------------------------------------------------------------------
# Internal model (encoder + gate + router + aggregator only)
# ---------------------------------------------------------------------------

class _AdapterCore(torch.nn.Module):
    """Stripped-down core: no proxy bridge, no nuisance heads, no audit."""

    def __init__(
        self,
        num_features: int,
        metadata_dim: int,
        num_intervention_levels: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = FeatureEmbeddingEncoder(num_features, metadata_dim, embedding_dim)
        self.gate = TemporalEligibilityGate(metadata_dim)
        self.router = HardEligibilityRoleRouter(num_features, 5)
        self.aggregator = RoleSubspaceAggregator()

        # Lightweight nuisance heads for training signal only
        # (not exported — downstream estimators use the subspaces directly)
        self.intervention_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, num_intervention_levels),
        )
        self.outcome_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2 + num_intervention_levels, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, 1),
        )
        self._num_levels = num_intervention_levels

    def forward(
        self,
        covariates: torch.Tensor,
        metadata_matrix: torch.Tensor,
        gate_source: str = "learned",
    ) -> dict:
        embeddings = self.encoder(covariates, metadata_matrix)
        raw_gate = self.gate(metadata_matrix)

        if gate_source == "oracle_metadata":
            eligibility = (
                (metadata_matrix[:, 0] < 0.0)
                & (metadata_matrix[:, 1] < 0.5)
                & (metadata_matrix[:, 3] < 0.5)
                & (metadata_matrix[:, 6] < 0.5)
            ).to(metadata_matrix.dtype)
        else:
            eligibility = raw_gate

        role_probs = self.router(eligibility)
        subspaces = self.aggregator(embeddings, role_probs)
        z_conf = subspaces[:, 0, :]   # (n, d)
        z_int  = subspaces[:, 1, :]
        z_out  = subspaces[:, 2, :]

        intervention_logits = self.intervention_head(
            torch.cat([z_conf, z_int], dim=-1)
        )
        identity = torch.eye(self._num_levels, device=covariates.device, dtype=covariates.dtype)
        potential_outcomes = torch.stack(
            [
                self.outcome_head(
                    torch.cat([z_conf, z_out, identity[k].unsqueeze(0).expand(z_conf.shape[0], -1)], dim=-1)
                ).squeeze(-1)
                for k in range(self._num_levels)
            ],
            dim=1,
        )
        return {
            "raw_gate": raw_gate,
            "eligibility_gate": eligibility,
            "role_probabilities": role_probs,
            "subspaces": subspaces,
            "z_conf": z_conf,
            "z_int": z_int,
            "z_out": z_out,
            "z_proxy": subspaces[:, 3, :],
            "z_remainder": subspaces[:, 4, :],
            "intervention_logits": intervention_logits,
            "potential_outcomes": potential_outcomes,
        }


# ---------------------------------------------------------------------------
# Adapter-A: TemporalRoleAdapter  (minimal)
# ---------------------------------------------------------------------------

class TemporalRoleAdapter(BaseAdapter):
    """
    Adapter-A — minimal learned adapter.

    Components:  metadata-aware feature encoder
                 temporal eligibility gate  (with auxiliary BCE supervision)
                 5-way hard-constrained role router
                 role subspace aggregator

    Training procedure (3-phase):
      Phase 0 — Gate pre-warmup (10% of total_epochs, gate params only, 2× LR):
                 Pure BCE(h_j, h_j*) — gives gate a head start before main-task
                 gradient competition.  Critical for convergence at large p.
      Phase 1 — Warmup (warmup_fraction of total_epochs):
                 L_I + L_Y + λ_gate * L_gate  (no balance / orth regularisers)
      Phase 2 — Full training (remaining epochs):
                 L_I + L_Y + λ_bal + λ_orth + λ_gate * L_gate

    Training loss (phase 2):
        L = L_I + L_Y + λ_bal * L_bal + λ_orth * L_orth + λ_gate * BCE(h, h*)

    Recommended hyperparameters (from D3 study):
        lambda_gate = 0.5  (p=40 features)  →  gate_F1 ≈ 1.0
        lambda_gate = 0.5  (p=100, with 2-phase)  →  gate_F1 ≈ 0.42, h_post ≈ 0.25

    No LLM, no proxy bridge, no audit energy.
    """

    def __init__(
        self,
        config: Optional[AdapterTrainingConfig] = None,
        gate_source: str = "learned",
    ):
        self.config = config or AdapterTrainingConfig()
        self.gate_source = gate_source
        self._core: Optional[_AdapterCore] = None
        self._metadata_matrix_np: Optional[np.ndarray] = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "TemporalRoleAdapter":
        seed_everything(self.config.seed)
        device = self._resolve_device()
        self._device = device
        dev = torch.device(device)

        meta_np = metadata_to_matrix(metadata)
        self._metadata_matrix_np = meta_np

        n, p = X.shape
        num_levels = int(intervention.max()) + 1
        meta_dim = meta_np.shape[1]

        core = _AdapterCore(p, meta_dim, num_levels, self.config.embedding_dim).to(dev)
        optimizer = torch.optim.Adam(core.parameters(), lr=self.config.learning_rate)
        balance_loss_fn = LocalizedBalanceLoss()
        orth_loss_fn = SubspaceOrthogonalityLoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        I_t = torch.tensor(intervention, dtype=torch.long, device=dev)
        Y_t = torch.tensor(outcome, dtype=torch.float32, device=dev)
        M_t = torch.tensor(meta_np, dtype=torch.float32, device=dev)

        # Gate supervision target from metadata
        gate_target = (
            (M_t[:, 0] < 0.0).float()
            * (M_t[:, 1] < 0.5).float()
            * (M_t[:, 3] < 0.5).float()
            * (M_t[:, 6] < 0.5).float()
        )

        num_warmup = max(1, int(self.config.total_epochs * self.config.warmup_fraction))
        # Gate pre-warm: a few epochs of pure BCE to give the gate a head start.
        # This prevents the main-task gradient from drowning the gate signal early on.
        num_gate_prewarm = max(0, int(self.config.total_epochs * 0.10))
        if self.config.lambda_gate > 0.0 and self.gate_source == "learned" and num_gate_prewarm > 0:
            gate_optimizer = torch.optim.Adam(core.gate.parameters(), lr=self.config.learning_rate * 2)
            for _ in range(num_gate_prewarm):
                core.train()
                gate_optimizer.zero_grad(set_to_none=True)
                raw_gate = core.gate(M_t)
                raw_clamped = raw_gate.clamp(1e-6, 1.0 - 1e-6)
                loss_gate = F.binary_cross_entropy(raw_clamped, gate_target)
                loss_gate.backward()
                gate_optimizer.step()

        for epoch in range(1, self.config.total_epochs + 1):
            core.train()
            optimizer.zero_grad(set_to_none=True)

            out = core(X_t, M_t, gate_source=self.gate_source)
            factual = out["potential_outcomes"].gather(1, I_t.unsqueeze(-1)).squeeze(-1)

            loss_I = F.cross_entropy(out["intervention_logits"], I_t)
            loss_Y = F.mse_loss(factual, Y_t)
            loss_bal = balance_loss_fn(out["z_conf"], I_t)
            loss_orth = orth_loss_fn(out["subspaces"])

            total = loss_I + loss_Y
            if epoch > num_warmup:
                total = total + self.config.lambda_balance * loss_bal
                total = total + self.config.lambda_orthogonality * loss_orth

            if self.config.lambda_gate > 0.0 and self.gate_source == "learned":
                raw_clamped = out["raw_gate"].clamp(1e-6, 1.0 - 1e-6)
                loss_gate = F.binary_cross_entropy(raw_clamped, gate_target)
                total = total + self.config.lambda_gate * loss_gate

            # T-06: gate ranking loss (pre features should have higher gate than post)
            if self.config.lambda_rank > 0.0 and self.gate_source == "learned":
                rank_loss_fn = GateRankingLoss(margin=self.config.rank_margin)
                loss_rank = rank_loss_fn(out["raw_gate"].squeeze(-1), gate_target)
                total = total + self.config.lambda_rank * loss_rank

            total.backward()
            optimizer.step()

        self._core = core
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        assert self._core is not None, "Call fit() before transform()."
        dev = torch.device(self._device)
        meta_np = metadata_to_matrix(metadata)

        self._core.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=dev)
            M_t = torch.tensor(meta_np, dtype=torch.float32, device=dev)
            out = self._core(X_t, M_t, gate_source=self.gate_source)

        def _np(key: str) -> np.ndarray:
            return out[key].detach().cpu().numpy()

        return AdapterOutput(
            confounding_repr=_np("z_conf"),
            intervention_repr=_np("z_int"),
            outcome_repr=_np("z_out"),
            proxy_repr=_np("z_proxy"),
            remainder_repr=_np("z_remainder"),
            role_probabilities=_np("role_probabilities"),
            eligibility_gate=_np("eligibility_gate"),
        )

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Adapter-B: RobustTemporalRoleAdapter  (+ heuristic audit loop)
# ---------------------------------------------------------------------------

class RobustTemporalRoleAdapter(TemporalRoleAdapter):
    """
    Adapter-B — adds a heuristic structured audit loop on top of Adapter-A.

    Every `audit_interval` epochs after warmup:
      1. Compute routing-weighted SMD per feature.
      2. Update role prior for flagged features using SMD + relative_time rules.
      3. Apply updated prior as additive logit bias to the router.

    No LLM. No oracle. Pure statistics + metadata.
    """

    def __init__(
        self,
        config: Optional[AdapterTrainingConfig] = None,
        gate_source: str = "learned",
        audit_interval: int = 10,
        smd_flag_threshold: float = 0.1,
        audit_top_k: int = 20,
    ):
        super().__init__(config=config, gate_source=gate_source)
        self.audit_interval = audit_interval
        self.smd_flag_threshold = smd_flag_threshold
        self.audit_top_k = audit_top_k

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "RobustTemporalRoleAdapter":
        from ..evaluation.attribution import compute_route_weighted_smd

        seed_everything(self.config.seed)
        device = self._resolve_device()
        self._device = device
        dev = torch.device(device)

        meta_np = metadata_to_matrix(metadata)
        self._metadata_matrix_np = meta_np

        n, p = X.shape
        num_levels = int(intervention.max()) + 1
        meta_dim = meta_np.shape[1]

        core = _AdapterCore(p, meta_dim, num_levels, self.config.embedding_dim).to(dev)
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
        # Uniform role prior to start
        current_prior_logits: Optional[torch.Tensor] = None

        for epoch in range(1, self.config.total_epochs + 1):
            core.train()
            optimizer.zero_grad(set_to_none=True)

            out = core(X_t, M_t, gate_source=self.gate_source)
            factual = out["potential_outcomes"].gather(1, I_t.unsqueeze(-1)).squeeze(-1)

            loss_I = F.cross_entropy(out["intervention_logits"], I_t)
            loss_Y = F.mse_loss(factual, Y_t)
            loss_bal = balance_loss_fn(out["z_conf"], I_t)
            loss_orth = orth_loss_fn(out["subspaces"])

            total = loss_I + loss_Y
            total = total + self.config.lambda_balance * loss_bal
            total = total + self.config.lambda_orthogonality * loss_orth

            if self.config.lambda_gate > 0.0 and self.gate_source == "learned":
                raw_clamped = out["raw_gate"].clamp(1e-6, 1.0 - 1e-6)
                total = total + self.config.lambda_gate * F.binary_cross_entropy(raw_clamped, gate_target)

            # Inject heuristic prior as router logit bias (post-warmup)
            if current_prior_logits is not None and epoch > num_warmup:
                prior_logits_t = torch.tensor(current_prior_logits, dtype=torch.float32, device=dev)
                probs_biased = core.router(out["eligibility_gate"], prior_logits=prior_logits_t)
                kl_prior = torch.sum(
                    probs_biased * (probs_biased.clamp_min(1e-8).log() - out["role_probabilities"].clamp_min(1e-8).log())
                )
                total = total + 0.01 * kl_prior

            total.backward()
            optimizer.step()

            # Periodic heuristic audit
            if epoch > num_warmup and epoch % self.audit_interval == 0:
                core.eval()
                with torch.no_grad():
                    refresh = core(X_t, M_t, gate_source=self.gate_source)
                role_np = refresh["role_probabilities"].detach().cpu().numpy()
                gate_np = refresh["eligibility_gate"].detach().cpu().numpy()
                weighted_smd = compute_route_weighted_smd(X, intervention, role_np[:, 0])
                top_idx = np.argsort(weighted_smd)[::-1][: self.audit_top_k]
                current_prior_logits = self._heuristic_prior_update(
                    role_np, gate_np, weighted_smd, top_idx, metadata
                )
                core.train()

        self._core = core
        return self

    @staticmethod
    def _heuristic_prior_update(
        role_probs: np.ndarray,
        gate: np.ndarray,
        weighted_smd: np.ndarray,
        top_idx: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> np.ndarray:
        """Return log-prior logit bias (p, 5) from heuristic rules."""
        p = role_probs.shape[0]
        log_prior = np.zeros((p, 5), dtype=np.float32)
        for j in top_idx:
            m = metadata[j]
            if m.relative_time >= 0.0 or m.post_intervention_keyword >= 0.5:
                # Post-intervention → push to remainder
                log_prior[j] = np.log([0.02, 0.05, 0.05, 0.08, 0.80])
            elif weighted_smd[j] > 0.15:
                # Very imbalanced, pre-index → likely intervention predictor
                log_prior[j] = np.log([0.05, 0.70, 0.10, 0.10, 0.05])
            elif weighted_smd[j] > 0.10:
                # Moderately imbalanced, pre-index → split confounder/intervention
                log_prior[j] = np.log([0.30, 0.40, 0.10, 0.15, 0.05])
            else:
                # Low SMD, pre-index → confounding
                log_prior[j] = np.log([0.65, 0.10, 0.10, 0.10, 0.05])
        return log_prior
