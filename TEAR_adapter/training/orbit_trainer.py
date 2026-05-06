from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..agents.audit_agent import HeuristicAuditor, LLMAuditorAgent, OracleAuditor
from ..agents.role_prior_agent import RolePriorAgent
from ..data.schemas import FeatureMetadata, metadata_to_matrix
from ..evaluation.attribution import compute_assignment_weight_attribution
from ..evaluation.caie_metrics import compute_caie_rmse
from ..evaluation.routing_metrics import build_audit_packet, compute_misrouting_score
from ..models.feature_encoder import FeatureEmbeddingEncoder
from ..models.nuisance_heads import InterventionAssignmentHead, OutcomeRegressionHead
from ..models.proxy_bridge import ProxyBridgeMLP
from ..models.role_router import HardEligibilityRoleRouter
from ..models.subspace_aggregator import RoleSubspaceAggregator
from ..models.temporal_gate import TemporalEligibilityGate
from ..objectives.audit_energy import IterativeStructuredAuditEnergy
from ..objectives.balance_loss import LocalizedBalanceLoss
from ..objectives.bridge_loss import ProxyBridgeBottleneckLoss
from ..objectives.orthogonality_loss import SubspaceOrthogonalityLoss
from ..utils.constraints import clip_exclusion_strength, filter_temporal_orderings
from ..utils.seed import seed_everything
from .audit_loop import DecayingAuditScheduler
from .warmup_trainer import WarmupSchedule


@dataclass
class OrbitVariantConfig:
    name: str
    gate_enabled: bool = True
    gate_source: str = "learned"
    routing_enabled: bool = True
    llm_init_enabled: bool = False
    audit_mode: str = "none"
    proxy_bridge_enabled: bool = True
    bridge_bottleneck_enabled: bool = True
    audit_corruption_mode: str = "none"
    corruption_fraction: float = 0.0
    confidence_override: Optional[float] = None
    adversarial_top_k: int = 5


@dataclass
class OrbitTrainingConfig:
    embedding_dim: int = 32
    latent_proxy_dim: int = 8
    bridge_depth: int = 2
    total_epochs: int = 40
    warmup_fraction: float = 0.25
    audit_interval: int = 10
    learning_rate: float = 1e-3
    lambda_balance: float = 0.5
    lambda_orthogonality: float = 0.1
    lambda_audit_base: float = 0.1
    lambda_bridge: float = 0.01
    lambda_gate: float = 0.1
    trim_threshold: float = 0.05
    audit_top_k: int = 20
    device: str = "auto"
    seed: int = 42


@dataclass
class OrbitTrainingResult:
    potential_outcomes: np.ndarray
    propensity_scores: np.ndarray
    role_probabilities: np.ndarray
    eligibility_gate: np.ndarray
    training_history: List[Dict[str, float]]
    audit_history: List[Dict[str, float]]
    metrics: Dict[str, float]
    predictor: "OrbitPredictor"


@dataclass
class OrbitPredictor:
    system: nn.Module
    metadata_matrix: np.ndarray
    variant_config: OrbitVariantConfig
    role_prior: np.ndarray
    device: str

    def predict(self, covariates: np.ndarray) -> Dict[str, np.ndarray]:
        device = torch.device(self.device)
        self.system.eval()
        with torch.no_grad():
            covariate_tensor = torch.tensor(covariates, dtype=torch.float32, device=device)
            metadata_tensor = torch.tensor(self.metadata_matrix, dtype=torch.float32, device=device)
            prior_tensor = torch.log(torch.tensor(self.role_prior, dtype=torch.float32, device=device).clamp_min(1e-8))
            outputs = self.system(covariate_tensor, metadata_tensor, self.variant_config, role_prior=prior_tensor)
        return {
            "potential_outcomes": outputs["predicted_potential_outcomes"].detach().cpu().numpy(),
            "propensity_scores": outputs["intervention_probabilities"].detach().cpu().numpy(),
            "role_probabilities": outputs["role_probabilities"].detach().cpu().numpy(),
            "eligibility_gate": outputs["eligibility_gate"].detach().cpu().numpy(),
        }


class OrbitSystem(nn.Module):
    def __init__(self, num_features: int, metadata_dim: int, num_intervention_levels: int, config: OrbitTrainingConfig):
        super().__init__()
        self.encoder = FeatureEmbeddingEncoder(num_features, metadata_dim, config.embedding_dim)
        self.temporal_gate = TemporalEligibilityGate(metadata_dim)
        self.router = HardEligibilityRoleRouter(num_features, 5)
        self.aggregator = RoleSubspaceAggregator()
        self.proxy_bridge = ProxyBridgeMLP(
            input_dim=config.embedding_dim,
            latent_dim=config.latent_proxy_dim,
            num_intervention_levels=num_intervention_levels,
            depth=config.bridge_depth,
        )
        self.intervention_head = InterventionAssignmentHead(config.embedding_dim, config.latent_proxy_dim, num_intervention_levels)
        self.outcome_head = OutcomeRegressionHead(config.embedding_dim, config.latent_proxy_dim, num_intervention_levels)

    def forward(
        self,
        covariates: torch.Tensor,
        metadata_matrix: torch.Tensor,
        variant_config: OrbitVariantConfig,
        role_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feature_embeddings = self.encoder(covariates, metadata_matrix)
        raw_gate = self.temporal_gate(metadata_matrix)
        if variant_config.gate_enabled:
            if variant_config.gate_source == "oracle_metadata":
                # metadata_matrix columns:
                # 0=relative_time, 1=post_intervention_keyword, 3=always_missing_pre_index, 6=post_index_window
                eligibility_gate = (
                    (metadata_matrix[:, 0] < 0.0)
                    & (metadata_matrix[:, 1] < 0.5)
                    & (metadata_matrix[:, 3] < 0.5)
                    & (metadata_matrix[:, 6] < 0.5)
                ).to(metadata_matrix.dtype)
            else:
                eligibility_gate = raw_gate
        else:
            eligibility_gate = torch.ones_like(raw_gate)

        if not variant_config.routing_enabled:
            if variant_config.gate_enabled:
                confounding_mass = eligibility_gate
                remainder_mass = 1.0 - confounding_mass
            else:
                confounding_mass = torch.ones_like(eligibility_gate)
                remainder_mass = torch.zeros_like(eligibility_gate)
            role_probabilities = torch.stack(
                [
                    confounding_mass,
                    torch.zeros_like(confounding_mass),
                    torch.zeros_like(confounding_mass),
                    torch.zeros_like(confounding_mass),
                    remainder_mass,
                ],
                dim=-1,
            )
        else:
            role_probabilities = self.router(eligibility_gate, prior_logits=role_prior)

        subspaces = self.aggregator(feature_embeddings, role_probabilities)
        confounding_subspace = subspaces[:, 0, :]
        intervention_subspace = subspaces[:, 1, :]
        outcome_subspace = subspaces[:, 2, :]
        proxy_subspace = subspaces[:, 3, :]

        if variant_config.proxy_bridge_enabled:
            bridge_output = self.proxy_bridge(proxy_subspace)
            latent_proxy = bridge_output.latent_proxy
        else:
            bridge_output = None
            latent_proxy = torch.zeros(confounding_subspace.shape[0], self.proxy_bridge.encoder[-1].out_features, device=covariates.device)

        intervention_logits = self.intervention_head(confounding_subspace, intervention_subspace, latent_proxy)
        intervention_probabilities = torch.softmax(intervention_logits, dim=-1)
        predicted_potential_outcomes = self.outcome_head.predict_all_levels(confounding_subspace, outcome_subspace, latent_proxy)
        return {
            "feature_embeddings": feature_embeddings,
            "raw_gate": raw_gate,
            "eligibility_gate": eligibility_gate,
            "role_probabilities": role_probabilities,
            "subspaces": subspaces,
            "confounding_subspace": confounding_subspace,
            "proxy_subspace": proxy_subspace,
            "bridge_output": bridge_output,
            "intervention_logits": intervention_logits,
            "intervention_probabilities": intervention_probabilities,
            "predicted_potential_outcomes": predicted_potential_outcomes,
        }


class OrbitTrainer:
    def __init__(self, training_config: OrbitTrainingConfig):
        self.training_config = training_config

    def _resolve_device(self) -> str:
        if self.training_config.device != "auto":
            return self.training_config.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _apply_audit_corruption(self, suggestion, packet, variant_config: OrbitVariantConfig):
        if suggestion is None:
            return None
        generator = np.random.default_rng(self.training_config.seed + int(packet.cycle_index))
        if variant_config.audit_corruption_mode == "random" and variant_config.corruption_fraction > 0.0:
            num_features = suggestion.role_prior.shape[0]
            num_corrupted_features = max(1, int(num_features * variant_config.corruption_fraction))
            corrupted_indices = generator.choice(num_features, size=num_corrupted_features, replace=False)
            suggestion.role_prior[corrupted_indices] = generator.dirichlet(np.ones(5), size=num_corrupted_features).astype(np.float32)
            suggestion.exclusion_strength[corrupted_indices] = generator.uniform(0.0, 1.0, size=num_corrupted_features).astype(np.float32)
            num_order_pairs = len(suggestion.ordering_pairs)
            if num_order_pairs > 0:
                num_corrupted_pairs = max(1, int(num_order_pairs * variant_config.corruption_fraction))
                suggestion.ordering_pairs = [
                    tuple(generator.choice(num_features, size=2, replace=False).tolist())
                    for _ in range(num_corrupted_pairs)
                ]
        elif variant_config.audit_corruption_mode == "adversarial":
            top_post = [
                feature.feature_index
                for feature in packet.feature_summaries
                if feature.oracle_role == "post_intervention"
            ][: variant_config.adversarial_top_k]
            top_confounding = [
                feature.feature_index
                for feature in packet.feature_summaries
                if feature.oracle_role == "confounding"
            ][: variant_config.adversarial_top_k]
            for feature_index in top_post:
                suggestion.role_prior[feature_index] = np.asarray([0.95, 0.01, 0.01, 0.01, 0.02], dtype=np.float32)
            for feature_index in top_confounding:
                suggestion.role_prior[feature_index] = np.asarray([0.02, 0.48, 0.30, 0.10, 0.10], dtype=np.float32)
            suggestion.ordering_pairs = [(feature_index, feature_index) for feature_index in top_post]
        if variant_config.confidence_override is not None:
            suggestion.confidence = variant_config.confidence_override
        return suggestion

    def fit(
        self,
        train_covariates: np.ndarray,
        train_intervention: np.ndarray,
        train_outcome: np.ndarray,
        metadata: List[FeatureMetadata],
        role_labels: List[str],
        variant_config: OrbitVariantConfig,
        eval_potential_outcomes: Optional[np.ndarray] = None,
    ) -> OrbitTrainingResult:
        seed_everything(self.training_config.seed)
        resolved_device = self._resolve_device()
        device = torch.device(resolved_device)
        metadata_matrix_np = metadata_to_matrix(metadata)
        num_features = train_covariates.shape[1]
        num_intervention_levels = int(train_intervention.max()) + 1
        system = OrbitSystem(
            num_features=num_features,
            metadata_dim=metadata_matrix_np.shape[1],
            num_intervention_levels=num_intervention_levels,
            config=self.training_config,
        ).to(device)
        optimizer = torch.optim.Adam(system.parameters(), lr=self.training_config.learning_rate)
        balance_loss = LocalizedBalanceLoss()
        orthogonality_loss = SubspaceOrthogonalityLoss()
        audit_energy = IterativeStructuredAuditEnergy(margin=0.1)
        bridge_loss = ProxyBridgeBottleneckLoss()
        metadata_matrix = torch.tensor(metadata_matrix_np, dtype=torch.float32, device=device)
        covariates = torch.tensor(train_covariates, dtype=torch.float32, device=device)
        intervention = torch.tensor(train_intervention, dtype=torch.long, device=device)
        outcome = torch.tensor(train_outcome, dtype=torch.float32, device=device)

        role_prior_agent = RolePriorAgent()
        heuristic_auditor = HeuristicAuditor()
        llm_auditor = LLMAuditorAgent() if variant_config.audit_mode == "llm" else None
        oracle_auditor = OracleAuditor() if variant_config.audit_mode == "oracle" else None
        prior_suggestion = role_prior_agent.suggest(metadata, use_llm=variant_config.llm_init_enabled)
        current_role_prior = prior_suggestion.role_prior.astype(np.float32)
        current_ordering_pairs = prior_suggestion.ordering_pairs
        current_exclusion_strength = np.zeros(num_features, dtype=np.float32)
        current_unit_risk = np.zeros(len(train_outcome), dtype=np.float32)
        current_confidence = 1.0

        warmup_schedule = WarmupSchedule(
            total_epochs=self.training_config.total_epochs,
            warmup_fraction=self.training_config.warmup_fraction,
        )
        audit_scheduler = DecayingAuditScheduler(
            base_weight=self.training_config.lambda_audit_base,
            num_features=num_features,
            num_samples=train_covariates.shape[0],
            total_epochs=self.training_config.total_epochs,
        )
        training_history: List[Dict[str, float]] = []
        audit_history: List[Dict[str, float]] = []

        for epoch_index in range(1, self.training_config.total_epochs + 1):
            system.train()
            optimizer.zero_grad(set_to_none=True)
            role_prior_tensor = torch.log(torch.tensor(current_role_prior, dtype=torch.float32, device=device).clamp_min(1e-8))
            outputs = system(covariates, metadata_matrix, variant_config, role_prior=role_prior_tensor)
            factual_outcome = outputs["predicted_potential_outcomes"].gather(1, intervention.unsqueeze(-1)).squeeze(-1)
            sample_weight = 1.0 - current_confidence * torch.tensor(current_unit_risk, dtype=torch.float32, device=device)

            intervention_loss = F.cross_entropy(outputs["intervention_logits"], intervention, reduction="none")
            intervention_loss = (sample_weight * intervention_loss).mean()
            outcome_loss = ((sample_weight * (factual_outcome - outcome).pow(2))).mean()
            confounding_subspace = outputs["confounding_subspace"]
            structured_balance = balance_loss(confounding_subspace, intervention)
            structured_orthogonality = orthogonality_loss(outputs["subspaces"])

            total_loss = intervention_loss + outcome_loss
            total_loss = total_loss + self.training_config.lambda_balance * structured_balance
            total_loss = total_loss + self.training_config.lambda_orthogonality * structured_orthogonality

            if variant_config.proxy_bridge_enabled and variant_config.bridge_bottleneck_enabled and outputs["bridge_output"] is not None:
                proxy_objective = bridge_loss(outputs["bridge_output"], outputs["proxy_subspace"], intervention)
                total_loss = total_loss + self.training_config.lambda_bridge * proxy_objective
            else:
                proxy_objective = torch.tensor(0.0, device=device)

            if self.training_config.lambda_gate > 0.0 and variant_config.gate_source == "learned":
                gate_target = (
                    (metadata_matrix[:, 0] < 0.0).float()
                    * (metadata_matrix[:, 1] < 0.5).float()
                    * (metadata_matrix[:, 3] < 0.5).float()
                    * (metadata_matrix[:, 6] < 0.5).float()
                )
                raw_gate_clamped = outputs["raw_gate"].clamp(1e-6, 1.0 - 1e-6)
                gate_supervision = F.binary_cross_entropy(raw_gate_clamped, gate_target)
                total_loss = total_loss + self.training_config.lambda_gate * gate_supervision
            else:
                gate_supervision = torch.tensor(0.0, device=device)

            if epoch_index > warmup_schedule.num_warmup_epochs and variant_config.audit_mode != "none":
                trim_indicator = (outputs["intervention_probabilities"].min(dim=1).values < self.training_config.trim_threshold).float()
                breakdown = audit_energy(
                    role_probabilities=outputs["role_probabilities"],
                    role_prior=torch.tensor(current_role_prior, dtype=torch.float32, device=device),
                    confounding_gate=outputs["eligibility_gate"],
                    exclusion_strength=torch.tensor(current_exclusion_strength, dtype=torch.float32, device=device),
                    ordering_pairs=current_ordering_pairs,
                    unit_risk=torch.tensor(current_unit_risk, dtype=torch.float32, device=device),
                    trim_indicator=trim_indicator,
                    confidence=torch.tensor(current_confidence, dtype=torch.float32, device=device),
                )
                current_lambda_audit = audit_scheduler.weight(epoch_index, current_confidence)
                total_loss = total_loss + current_lambda_audit * breakdown.total_energy
            else:
                current_lambda_audit = 0.0

            total_loss.backward()
            optimizer.step()

            history_record = {
                "epoch": float(epoch_index),
                "loss_total": float(total_loss.detach().cpu().item()),
                "loss_intervention": float(intervention_loss.detach().cpu().item()),
                "loss_outcome": float(outcome_loss.detach().cpu().item()),
                "loss_balance": float(structured_balance.detach().cpu().item()),
                "loss_orthogonality": float(structured_orthogonality.detach().cpu().item()),
                "loss_bridge": float(proxy_objective.detach().cpu().item()),
                "loss_gate": float(gate_supervision.detach().cpu().item()),
                "lambda_audit": float(current_lambda_audit),
            }
            training_history.append(history_record)

            if epoch_index > warmup_schedule.num_warmup_epochs and variant_config.audit_mode != "none" and epoch_index % self.training_config.audit_interval == 0:
                system.eval()
                with torch.no_grad():
                    refreshed_outputs = system(covariates, metadata_matrix, variant_config, role_prior=role_prior_tensor)
                role_probability_np = refreshed_outputs["role_probabilities"].detach().cpu().numpy()
                propensity_np = refreshed_outputs["intervention_probabilities"].detach().cpu().numpy()
                value_scale = system.encoder.value_scale
                first_assignment_weight = system.intervention_head.network[0].weight
                raw_attribution = compute_assignment_weight_attribution(value_scale, first_assignment_weight)
                attribution_intervention = raw_attribution * np.clip(role_probability_np[:, 1], 0.05, None)
                attribution_outcome = raw_attribution * np.clip(role_probability_np[:, 2], 0.05, None)
                attribution_intervention = attribution_intervention / np.maximum(attribution_intervention.sum(), 1e-8)
                attribution_outcome = attribution_outcome / np.maximum(attribution_outcome.sum(), 1e-8)
                packet = build_audit_packet(
                    covariates=train_covariates,
                    intervention=train_intervention,
                    propensity_scores=propensity_np,
                    role_probabilities=role_probability_np,
                    metadata=metadata,
                    attribution_intervention=attribution_intervention,
                    attribution_outcome=attribution_outcome,
                    cycle_index=epoch_index // self.training_config.audit_interval,
                    trim_threshold=self.training_config.trim_threshold,
                    top_k=self.training_config.audit_top_k,
                )
                unit_risk_template = np.clip(
                    (self.training_config.trim_threshold - propensity_np.min(axis=1)) / max(self.training_config.trim_threshold, 1e-6),
                    0.0,
                    1.0,
                ).astype(np.float32)
                gate_np = refreshed_outputs["eligibility_gate"].detach().cpu().numpy()
                if variant_config.audit_mode == "heuristic":
                    suggestion = heuristic_auditor.suggest(packet, current_role_prior, gate_np, unit_risk_template)
                elif variant_config.audit_mode == "llm":
                    suggestion = llm_auditor.suggest(packet, current_role_prior, gate_np, unit_risk_template, metadata)
                elif variant_config.audit_mode == "oracle":
                    suggestion = oracle_auditor.suggest(packet, metadata, unit_risk_template)
                else:
                    suggestion = None
                if suggestion is not None:
                    suggestion = self._apply_audit_corruption(suggestion, packet, variant_config)
                    current_role_prior = 0.5 * current_role_prior + 0.5 * suggestion.role_prior
                    current_role_prior = current_role_prior / np.maximum(current_role_prior.sum(axis=1, keepdims=True), 1e-8)
                    current_exclusion_strength = clip_exclusion_strength(suggestion.exclusion_strength, gate_np)
                    relative_times = np.asarray([item.relative_time for item in metadata], dtype=np.float32)
                    current_ordering_pairs = filter_temporal_orderings(suggestion.ordering_pairs, relative_times)
                    current_unit_risk = suggestion.unit_risk
                    current_confidence = float(np.clip(suggestion.confidence, 0.0, 1.0))
                    audit_history.append(
                        {
                            "epoch": float(epoch_index),
                            "confidence": current_confidence,
                            "trim_fraction": packet.overlap_summary.trim_fraction,
                            "minimum_propensity": packet.overlap_summary.minimum_propensity,
                            "num_flagged_features": float(len(packet.feature_summaries)),
                            "audit_source": suggestion.source,
                            "updated_feature_count": float(suggestion.updated_feature_count),
                        }
                    )

        system.eval()
        with torch.no_grad():
            final_role_prior = torch.log(torch.tensor(current_role_prior, dtype=torch.float32, device=device).clamp_min(1e-8))
            outputs = system(covariates, metadata_matrix, variant_config, role_prior=final_role_prior)
        predicted_potential_outcomes = outputs["predicted_potential_outcomes"].detach().cpu().numpy()
        predicted_propensity = outputs["intervention_probabilities"].detach().cpu().numpy()
        role_probability_np = outputs["role_probabilities"].detach().cpu().numpy()
        eligibility_gate = outputs["eligibility_gate"].detach().cpu().numpy()
        metrics = compute_misrouting_score(role_probability_np, role_labels)
        if eval_potential_outcomes is not None:
            metrics["caie_rmse"] = compute_caie_rmse(predicted_potential_outcomes, eval_potential_outcomes)
        predictor = OrbitPredictor(
            system=system,
            metadata_matrix=metadata_matrix_np,
            variant_config=variant_config,
            role_prior=current_role_prior,
            device=resolved_device,
        )
        return OrbitTrainingResult(
            potential_outcomes=predicted_potential_outcomes,
            propensity_scores=predicted_propensity,
            role_probabilities=role_probability_np,
            eligibility_gate=eligibility_gate,
            training_history=training_history,
            audit_history=audit_history,
            metrics=metrics,
            predictor=predictor,
        )
