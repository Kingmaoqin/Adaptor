"""
SafeTemporalRoleAdapter (修改 A + B + C + D)
=============================================
新主版本 adapter，修复旧版的四类问题：

  A. Safe router: post-intervention features 被全部 gate 到隐藏子空间 {P,R}，
     无法进入 {C,I,Y} 任意可见子空间。
  B. safe_full 输出模式: [z^C, z^I, z^Y] 基于 safe router，不含 post 泄露。
     full_legacy: 保留旧版，仅用于对照。
  C. Treatment-specific outcome heads: 每个 treatment level 独立头，
     避免 factual treatment-conditional shortcut。
  D. Metadata 模式: static_only（默认）vs static_plus_behavior（消融）。
     behavior-derived stats 只在 training split 内计算。

Cross-fitting 边界（修改 E）：
  本 adapter 的 fit() 接口不变；cross-fitting 场景下应使用
  orbit_adapter.downstream.crossfit_adapter 中的 FoldwiseAdapterDML/RLearner
  进行 fold-level adapter re-fit。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional

from ..data.schemas import FeatureMetadata, metadata_to_matrix
from ..models.feature_encoder import FeatureEmbeddingEncoder
from ..models.temporal_gate import TemporalEligibilityGate
from ..models.subspace_aggregator import RoleSubspaceAggregator
from ..models.role_router_safe import SafeEligibilityRoleRouter
from ..models.role_adapters import RoleSpecificEncoderStack, per_role_aggregation
from ..models.outcome_heads import TreatmentSpecificOutcomeHeads, FactualSingleOutcomeHead
from ..objectives.balance_loss import LocalizedBalanceLoss
from ..objectives.orthogonality_loss import SubspaceOrthogonalityLoss
from ..objectives.disentangle_loss import (
    routing_entropy_loss, AdversarialDisentangleHeads,
)
from ..utils.seed import seed_everything
from .base import AdapterOutput, BaseAdapter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SafeAdapterConfig:
    embedding_dim: int = 32
    total_epochs: int = 40
    warmup_fraction: float = 0.25
    gate_prewarm_fraction: float = 0.10
    learning_rate: float = 1e-3
    lambda_balance: float = 0.5
    lambda_orthogonality: float = 0.1
    lambda_gate: float = 0.5
    # Modification C: outcome head mode
    outcome_head_mode: str = "treatment_specific"  # or "single_head" (legacy)
    # Modification D: metadata mode
    metadata_mode: str = "static_only"  # or "static_plus_behavior"
    # Modification F: routing entropy penalty (sharpen within-partition role assignment)
    lambda_entropy: float = 0.0
    # Modification G: adversarial disentanglement (z_out ⊥ T, z_int ⊥ Y)
    lambda_disentangle: float = 0.0
    # Modification H: break uniform-softmax saddle by randomly initializing role logits
    router_init_std: float = 0.0
    # Modification I: role-specific light encoder adapters (attacks cos(C,Y)≈1)
    use_role_adapters: bool = False
    role_adapter_bottleneck: int = 2
    # Modification I-var2: adapter_type = "mlp" (E1) or "linear_lowrank" (E1-var2)
    role_adapter_type: str = "mlp"
    # Modification J (E2): routing_mode = "softmax" (default) or "sparsemax"
    routing_mode: str = "softmax"
    record_history: bool = False
    device: str = "auto"
    seed: int = 42


# ---------------------------------------------------------------------------
# Metadata helpers (Modification D)
# ---------------------------------------------------------------------------

def _static_metadata_matrix(metadata: List[FeatureMetadata]) -> np.ndarray:
    """7-column static schema metadata (same as original metadata_to_matrix)."""
    return metadata_to_matrix(metadata)


def _behavior_metadata_matrix(
    metadata: List[FeatureMetadata],
    X_train: np.ndarray,
) -> np.ndarray:
    """
    7 static columns + 2 behavior-derived columns (computed ONLY from training split):
      col 7: per-feature mean (normalized) in training data
      col 8: per-feature zero-fraction in training data

    Risk: these columns can encode assignment-correlated statistics
    (e.g., post-intervention features have higher mean in treated arm if
    the intervention changes the measurement).
    """
    static = metadata_to_matrix(metadata)  # (p, 7)
    feat_mean = X_train.mean(axis=0, keepdims=True).T.astype(np.float32)      # (p, 1)
    feat_mean_norm = (feat_mean - feat_mean.mean()) / (feat_mean.std() + 1e-8)
    zero_frac = (X_train == 0).mean(axis=0, keepdims=True).T.astype(np.float32)  # (p, 1)
    return np.concatenate([static, feat_mean_norm, zero_frac], axis=1)  # (p, 9)


# ---------------------------------------------------------------------------
# Internal safe model core
# ---------------------------------------------------------------------------

class _SafeAdapterCore(torch.nn.Module):
    """
    Core with SafeEligibilityRoleRouter + configurable outcome head.
    """

    def __init__(
        self,
        num_features: int,
        metadata_dim: int,
        num_levels: int,
        embedding_dim: int,
        outcome_head_mode: str = "treatment_specific",
        use_disentangle: bool = False,
        router_init_std: float = 0.0,
        use_role_adapters: bool = False,
        role_adapter_bottleneck: int = 2,
        role_adapter_type: str = "mlp",
        routing_mode: str = "softmax",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.outcome_head_mode = outcome_head_mode
        self.use_role_adapters = use_role_adapters

        self.encoder = FeatureEmbeddingEncoder(num_features, metadata_dim, embedding_dim)
        self.gate = TemporalEligibilityGate(metadata_dim)
        # Modification A: safe router (+ Modification H: init_std, + Modification J: routing_mode)
        self.router = SafeEligibilityRoleRouter(
            num_features, 5, init_std=router_init_std, routing_mode=routing_mode,
        )
        self.aggregator = RoleSubspaceAggregator()
        # Modification I: role-specific light encoders (active when flag set)
        self.role_adapters = (
            RoleSpecificEncoderStack(
                embedding_dim, role_adapter_bottleneck,
                adapter_type=role_adapter_type,
            )
            if use_role_adapters else None
        )

        # Propensity head (unchanged)
        self.intervention_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, num_levels),
        )

        # Modification C: outcome head selection
        if outcome_head_mode == "treatment_specific":
            self.outcome_module = TreatmentSpecificOutcomeHeads(embedding_dim, num_levels)
        else:
            self.outcome_module = FactualSingleOutcomeHead(embedding_dim, num_levels)

        # Modification G: adversarial disentanglement heads
        self.adv_heads = (
            AdversarialDisentangleHeads(embedding_dim, num_levels)
            if use_disentangle else None
        )

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

        role_probs = self.router(eligibility)  # (p, 5) — safe routing

        # Modification I: per-role embedding path (breaks shared-encoder bottleneck)
        if self.role_adapters is not None:
            role_embeddings = self.role_adapters(embeddings)      # (n, p, 5, d)
            subspaces = per_role_aggregation(role_probs, role_embeddings)  # (n, 5, d)
        else:
            subspaces = self.aggregator(embeddings, role_probs)   # (n, 5, d)

        z_conf = subspaces[:, 0, :]
        z_int  = subspaces[:, 1, :]
        z_out  = subspaces[:, 2, :]

        intervention_logits = self.intervention_head(
            torch.cat([z_conf, z_int], dim=-1)
        )
        potential_outcomes = self.outcome_module.forward(z_conf, z_out)  # (n, K)

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

    def outcome_loss(
        self,
        z_conf: torch.Tensor,
        z_out: torch.Tensor,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Route to correct loss depending on head mode."""
        if self.outcome_head_mode == "treatment_specific":
            return self.outcome_module.treatment_specific_loss(z_conf, z_out, Y, T)
        else:
            return self.outcome_module.factual_loss(z_conf, z_out, Y, T)


# ---------------------------------------------------------------------------
# SafeTemporalRoleAdapter
# ---------------------------------------------------------------------------

class SafeTemporalRoleAdapter(BaseAdapter):
    """
    Safe adapter: fixes post-treatment leakage, factual shortcut, metadata risk.

    Modifications vs. TemporalRoleAdapter (legacy):
      A. SafeEligibilityRoleRouter: h_j gates ALL visible subspaces {C,I,Y}
      B. new output modes: 'safe_full', 'confounding', 'full_legacy'
      C. TreatmentSpecificOutcomeHeads (or 'single_head' ablation)
      D. metadata_mode: 'static_only' (default) or 'static_plus_behavior'
    """

    def __init__(
        self,
        config: Optional[SafeAdapterConfig] = None,
        gate_source: str = "learned",
    ):
        self.config = config or SafeAdapterConfig()
        self.gate_source = gate_source
        self._core: Optional[_SafeAdapterCore] = None
        self._metadata_matrix_np: Optional[np.ndarray] = None
        self._device: Optional[str] = None
        self._X_train_for_behavior: Optional[np.ndarray] = None
        self.history_: list[dict] = []

    def _make_meta_matrix(
        self,
        metadata: List[FeatureMetadata],
        X_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.config.metadata_mode == "static_plus_behavior" and X_ref is not None:
            return _behavior_metadata_matrix(metadata, X_ref)
        return _static_metadata_matrix(metadata)

    def fit(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "SafeTemporalRoleAdapter":
        seed_everything(self.config.seed)
        device = self._resolve_device()
        self._device = device
        dev = torch.device(device)

        # Modification D: metadata construction
        self._X_train_for_behavior = X.copy()
        meta_np = self._make_meta_matrix(metadata, X_ref=X)
        self._metadata_matrix_np = meta_np
        self._metadata_list = metadata

        n, p = X.shape
        num_levels = int(intervention.max()) + 1
        meta_dim = meta_np.shape[1]

        core = _SafeAdapterCore(
            p, meta_dim, num_levels, self.config.embedding_dim,
            outcome_head_mode=self.config.outcome_head_mode,
            use_disentangle=(self.config.lambda_disentangle > 0.0),
            router_init_std=self.config.router_init_std,
            use_role_adapters=self.config.use_role_adapters,
            role_adapter_bottleneck=self.config.role_adapter_bottleneck,
            role_adapter_type=self.config.role_adapter_type,
            routing_mode=self.config.routing_mode,
        ).to(dev)
        optimizer = torch.optim.Adam(core.parameters(), lr=self.config.learning_rate)
        balance_loss_fn = LocalizedBalanceLoss()
        orth_loss_fn = SubspaceOrthogonalityLoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        I_t = torch.tensor(intervention, dtype=torch.long, device=dev)
        Y_t = torch.tensor(outcome, dtype=torch.float32, device=dev)
        M_t = torch.tensor(meta_np, dtype=torch.float32, device=dev)

        # Gate supervision target from STATIC metadata columns (0,1,3,6)
        # (always based on static columns, regardless of metadata_mode)
        gate_target = (
            (M_t[:, 0] < 0.0).float()
            * (M_t[:, 1] < 0.5).float()
            * (M_t[:, 3] < 0.5).float()
            * (M_t[:, 6] < 0.5).float()
        )

        num_warmup = max(1, int(self.config.total_epochs * self.config.warmup_fraction))
        # Gate pre-warmup (10% epochs, gate params only, 2× LR)
        num_gate_prewarm = max(0, int(self.config.total_epochs * self.config.gate_prewarm_fraction))
        if self.config.lambda_gate > 0.0 and self.gate_source == "learned" and num_gate_prewarm > 0:
            gate_opt = torch.optim.Adam(core.gate.parameters(), lr=self.config.learning_rate * 2)
            for _ in range(num_gate_prewarm):
                core.train()
                gate_opt.zero_grad(set_to_none=True)
                raw_gate = core.gate(M_t)
                loss_g = F.binary_cross_entropy(raw_gate.clamp(1e-6, 1 - 1e-6), gate_target)
                loss_g.backward()
                gate_opt.step()

        for epoch in range(1, self.config.total_epochs + 1):
            core.train()
            optimizer.zero_grad(set_to_none=True)

            out = core(X_t, M_t, gate_source=self.gate_source)

            loss_I = F.cross_entropy(out["intervention_logits"], I_t)
            # Modification C: use treatment-specific or single-head loss
            loss_Y = core.outcome_loss(out["z_conf"], out["z_out"], Y_t, I_t)
            loss_bal = balance_loss_fn(out["z_conf"], I_t)
            loss_orth = orth_loss_fn(out["subspaces"])

            total = loss_I + loss_Y
            if epoch > num_warmup:
                total = total + self.config.lambda_balance * loss_bal
                total = total + self.config.lambda_orthogonality * loss_orth

            if self.config.lambda_gate > 0.0 and self.gate_source == "learned":
                raw_clamped = out["raw_gate"].clamp(1e-6, 1 - 1e-6)
                total = total + self.config.lambda_gate * F.binary_cross_entropy(
                    raw_clamped, gate_target
                )

            # Modification F: routing entropy penalty (sharpen within-partition routing)
            if self.config.lambda_entropy > 0.0 and epoch > num_warmup:
                loss_ent = routing_entropy_loss(core.router.role_logits)
                total = total + self.config.lambda_entropy * loss_ent

            # Modification G: adversarial disentanglement
            if self.config.lambda_disentangle > 0.0 and core.adv_heads is not None \
                    and epoch > num_warmup:
                Y_f = Y_t.float()
                loss_adv = core.adv_heads(
                    out["z_int"], out["z_out"], I_t, Y_f, alpha=1.0
                )
                total = total + self.config.lambda_disentangle * loss_adv

            total.backward()
            optimizer.step()

            if self.config.record_history:
                with torch.no_grad():
                    gate_np = out["eligibility_gate"].detach().cpu().numpy()
                    role_np = out["role_probabilities"].detach().cpu().numpy()
                    post_mask = np.asarray([m.oracle_role == "post_intervention" for m in metadata], dtype=bool)
                    pre_mask = ~post_mask
                    if post_mask.any():
                        post_h = float(gate_np[post_mask].mean())
                        epsilon_post = float(role_np[post_mask, 0:3].sum())
                        post_vis_mass = float(role_np[post_mask, 0:3].sum(axis=1).mean())
                    else:
                        post_h = 0.0
                        epsilon_post = 0.0
                        post_vis_mass = 0.0
                    pre_h = float(gate_np[pre_mask].mean()) if pre_mask.any() else 0.0
                    self.history_.append(
                        {
                            "epoch": int(epoch),
                            "mean_post_h": post_h,
                            "mean_pre_h": pre_h,
                            "epsilon_post": epsilon_post,
                            "post_visible_mass": post_vis_mass,
                        }
                    )

        self._core = core
        return self

    def transform(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
    ) -> AdapterOutput:
        assert self._core is not None, "Call fit() before transform()."
        dev = torch.device(self._device)
        # Modification D: use same metadata construction mode
        meta_np = self._make_meta_matrix(metadata, X_ref=self._X_train_for_behavior)

        self._core.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=dev)
            M_t = torch.tensor(meta_np, dtype=torch.float32, device=dev)
            out = self._core(X_t, M_t, gate_source=self.gate_source)

        def _np(key: str) -> np.ndarray:
            return out[key].detach().cpu().numpy()

        return SafeAdapterOutput(
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
# SafeAdapterOutput: adds safe_full and full_legacy output modes
# ---------------------------------------------------------------------------

class SafeAdapterOutput(AdapterOutput):
    """
    Extended output with safe_full and full_legacy output modes.

    Modification B:
      'confounding'  → z^(C)                                (safe, same as before)
      'safe_full'    → [z^(C), z^(I), z^(Y)] from safe router  (no post leakage)
      'full_legacy'  → [z^(C), z^(I), z^(Y)] — same data but label for comparison
      'raw_conf_int' → [z^(C), z^(I)]                          (deprecated)
    """

    def repr_for_estimator(self, output_mode: str = "confounding") -> np.ndarray:
        if output_mode in ("safe_full", "full"):
            return np.concatenate(
                [self.confounding_repr, self.intervention_repr, self.outcome_repr],
                axis=1,
            )
        if output_mode == "full_legacy":
            # Same computation, different label — downstream knows it came from legacy router
            return np.concatenate(
                [self.confounding_repr, self.intervention_repr, self.outcome_repr],
                axis=1,
            )
        if output_mode == "raw_conf_int":
            return np.concatenate([self.confounding_repr, self.intervention_repr], axis=1)
        return self.confounding_repr


# ---------------------------------------------------------------------------
# Leakage diagnostic helper
# ---------------------------------------------------------------------------

def compute_leakage_metrics(
    role_probabilities: np.ndarray,  # (p, 5)
    role_labels: List[str],
) -> dict:
    """
    Compute post-treatment leakage diagnostics.

    Returns
    -------
    epsilon_post : float
        Total probability mass of post-intervention features in {C,I,Y}.
    epsilon_miss : float
        Total probability mass of true confounders NOT in C.
    post_vis_mass_mean : float
        Mean visible-subspace mass per post-intervention feature.
    post_conf_mass_mean : float
        Mean confounding mass per post-intervention feature (C only).
    post_int_mass_mean : float
        Mean intervention-subspace mass per post-intervention feature.
    post_out_mass_mean : float
        Mean outcome-subspace mass per post-intervention feature.
    """
    role_arr = np.asarray(role_labels)
    post_mask = role_arr == "post_intervention"
    conf_mask = role_arr == "confounding"

    g_vis = role_probabilities[:, 0:3].sum(axis=1)   # (p,)  mass in {C,I,Y}
    g_conf = role_probabilities[:, 0]                  # (p,)

    if post_mask.sum() == 0:
        epsilon_post = 0.0
        post_vis_mass_mean = 0.0
        post_conf_mass_mean = 0.0
        post_int_mass_mean = 0.0
        post_out_mass_mean = 0.0
    else:
        post_probs = role_probabilities[post_mask]  # (n_post, 5)
        epsilon_post = float(post_probs[:, 0:3].sum())
        post_vis_mass_mean = float(post_probs[:, 0:3].sum(axis=1).mean())
        post_conf_mass_mean = float(post_probs[:, 0].mean())
        post_int_mass_mean = float(post_probs[:, 1].mean())
        post_out_mass_mean = float(post_probs[:, 2].mean())

    if conf_mask.sum() == 0:
        epsilon_miss = 0.0
    else:
        epsilon_miss = float((1.0 - g_conf[conf_mask]).sum())

    epsilon_total = epsilon_post + epsilon_miss

    return {
        "epsilon_post": round(epsilon_post, 4),
        "epsilon_miss": round(epsilon_miss, 4),
        "epsilon_total": round(epsilon_total, 4),
        "post_vis_mass_mean": round(post_vis_mass_mean, 4),
        "post_conf_mass_mean": round(post_conf_mass_mean, 4),
        "post_int_mass_mean": round(post_int_mass_mean, 4),
        "post_out_mass_mean": round(post_out_mass_mean, 4),
    }
