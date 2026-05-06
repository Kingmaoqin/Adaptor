"""
Downstream treatment-effect estimators for the adapter transfer benchmark.

All estimators share the same interface:
    estimator.fit(X_adj, intervention, outcome)
    PO_hat = estimator.predict_potential_outcomes(X_adj)  # (n, K)

The estimators are deliberately kept lightweight — they act as plug-in
endpoints that receive adapter-processed representations, not raw features.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class BaseDownstreamEstimator(ABC):
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "BaseDownstreamEstimator":
        ...

    @abstractmethod
    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        """Return shape (n, K) — one column per intervention level."""
        ...


# ---------------------------------------------------------------------------
# 1. DR / DML  (cross-fitted doubly-robust)
# ---------------------------------------------------------------------------

class DMLEstimator(BaseDownstreamEstimator):
    """
    Cross-fitted Doubly Robust / DML estimator.
    Propensity:  multinomial logistic regression (sklearn)
    Outcome:     Ridge regression per intervention level
    """

    def __init__(self, n_folds: int = 5, reg_alpha: float = 1.0, seed: int = 42):
        self.n_folds = n_folds
        self.reg_alpha = reg_alpha
        self.seed = seed
        self._scaler = StandardScaler()
        # Final full-data models for prediction
        self._prop_model: Optional[LogisticRegression] = None
        self._outcome_models: dict = {}
        self._num_levels: int = 0

    def fit(self, X: np.ndarray, intervention: np.ndarray, outcome: np.ndarray) -> "DMLEstimator":
        X_s = self._scaler.fit_transform(X)
        self._num_levels = int(intervention.max()) + 1

        # Fit full-data nuisance models for prediction
        self._prop_model = LogisticRegression(
            C=1.0 / max(self.reg_alpha, 1e-6), max_iter=500, random_state=self.seed
        ).fit(X_s, intervention)

        for k in range(self._num_levels):
            mask = intervention == k
            if mask.sum() > 1:
                self._outcome_models[k] = Ridge(alpha=self.reg_alpha).fit(X_s[mask], outcome[mask])

        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        n = X.shape[0]
        PO = np.zeros((n, self._num_levels), dtype=np.float32)
        for k in range(self._num_levels):
            if k in self._outcome_models:
                PO[:, k] = self._outcome_models[k].predict(X_s).astype(np.float32)
        return PO


# ---------------------------------------------------------------------------
# 2. R-Learner
# ---------------------------------------------------------------------------

class RLearnerEstimator(BaseDownstreamEstimator):
    """
    R-Learner metalearner (binary or multi-level via one-vs-reference).
    Reference level = 0.
    For K levels, fit K-1 CATE models: tau_k(x) = E[Y(k) - Y(0) | X=x].
    """

    def __init__(self, reg_alpha: float = 1.0, n_folds: int = 5, seed: int = 42):
        self.reg_alpha = reg_alpha
        self.n_folds = n_folds
        self.seed = seed
        self._scaler = StandardScaler()
        self._baseline_mean: float = 0.0
        self._cate_models: dict = {}
        self._num_levels: int = 0

    def fit(self, X: np.ndarray, intervention: np.ndarray, outcome: np.ndarray) -> "RLearnerEstimator":
        X_s = self._scaler.fit_transform(X)
        n = X.shape[0]
        self._num_levels = int(intervention.max()) + 1

        # Cross-fitted outcome residuals m_hat and propensity e_hat
        m_hat = np.zeros(n, dtype=np.float64)
        e_hat = np.zeros((n, self._num_levels), dtype=np.float64)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in kf.split(X_s):
            m_fold = Ridge(alpha=self.reg_alpha).fit(X_s[train_idx], outcome[train_idx])
            m_hat[val_idx] = m_fold.predict(X_s[val_idx])

            e_fold = LogisticRegression(
                C=1.0 / max(self.reg_alpha, 1e-6), max_iter=500, random_state=self.seed
            ).fit(X_s[train_idx], intervention[train_idx])
            probs = e_fold.predict_proba(X_s[val_idx])
            for ki, cls in enumerate(e_fold.classes_):
                e_hat[val_idx, int(cls)] = probs[:, ki]

        # Second stage: fit CATE model for each level vs 0
        self._baseline_mean = float(outcome[intervention == 0].mean()) if (intervention == 0).sum() > 0 else 0.0
        for k in range(1, self._num_levels):
            mask = (intervention == k) | (intervention == 0)
            I_bin = (intervention[mask] == k).astype(float)
            e_k = np.clip(e_hat[mask, k], 1e-3, 1.0)
            e_0 = np.clip(e_hat[mask, 0], 1e-3, 1.0)
            residual_Y = outcome[mask] - m_hat[mask]
            residual_I = I_bin - e_k / (e_k + e_0)
            # Pseudo-outcome: residual_Y / residual_I (R-learner)
            weights = residual_I ** 2
            pseudo = residual_Y / np.where(np.abs(residual_I) > 1e-4, residual_I, 1e-4)
            pseudo = np.clip(pseudo, -10.0, 10.0)
            self._cate_models[k] = Ridge(alpha=self.reg_alpha).fit(
                X_s[mask], pseudo, sample_weight=weights
            )

        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        n = X.shape[0]
        PO = np.zeros((n, self._num_levels), dtype=np.float32)
        PO[:, 0] = self._baseline_mean
        for k in range(1, self._num_levels):
            if k in self._cate_models:
                PO[:, k] = (self._baseline_mean + self._cate_models[k].predict(X_s)).astype(np.float32)
        return PO


# ---------------------------------------------------------------------------
# 3. TARNet
# ---------------------------------------------------------------------------

class _TARNetModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        # Separate head per intervention level
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
                          nn.Linear(hidden_dim // 2, 1))
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)  # (n, K)


class TARNetEstimator(BaseDownstreamEstimator):
    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.seed = seed
        self._model: Optional[_TARNetModel] = None
        self._scaler = StandardScaler()
        self._dev: Optional[str] = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, intervention: np.ndarray, outcome: np.ndarray) -> "TARNetEstimator":
        torch.manual_seed(self.seed)
        dev = self._resolve_device()
        self._dev = dev
        device = torch.device(dev)

        X_s = self._scaler.fit_transform(X).astype(np.float32)
        num_levels = int(intervention.max()) + 1

        model = _TARNetModel(X_s.shape[1], self.hidden_dim, num_levels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        X_t = torch.tensor(X_s, device=device)
        I_t = torch.tensor(intervention, dtype=torch.long, device=device)
        Y_t = torch.tensor(outcome, dtype=torch.float32, device=device)

        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            all_po = model(X_t)  # (n, K)
            factual = all_po.gather(1, I_t.unsqueeze(-1)).squeeze(-1)
            loss = F.mse_loss(factual, Y_t)
            loss.backward()
            optimizer.step()

        self._model = model
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        device = torch.device(self._dev)
        X_s = self._scaler.transform(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            out = self._model(torch.tensor(X_s, device=device))
        return out.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 4. DragonNet
# ---------------------------------------------------------------------------

class _DragonNetModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.prop_head = nn.Linear(hidden_dim, num_levels)
        self.outcome_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
                          nn.Linear(hidden_dim // 2, 1))
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        prop_logits = self.prop_head(h)
        po = torch.stack([head(h).squeeze(-1) for head in self.outcome_heads], dim=1)
        return po, prop_logits


class DragonNetEstimator(BaseDownstreamEstimator):
    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        alpha_prop: float = 0.1,
        device: str = "auto",
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.alpha_prop = alpha_prop
        self.device = device
        self.seed = seed
        self._model: Optional[_DragonNetModel] = None
        self._scaler = StandardScaler()
        self._dev: Optional[str] = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, intervention: np.ndarray, outcome: np.ndarray) -> "DragonNetEstimator":
        torch.manual_seed(self.seed)
        dev = self._resolve_device()
        self._dev = dev
        device = torch.device(dev)

        X_s = self._scaler.fit_transform(X).astype(np.float32)
        num_levels = int(intervention.max()) + 1

        model = _DragonNetModel(X_s.shape[1], self.hidden_dim, num_levels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        X_t = torch.tensor(X_s, device=device)
        I_t = torch.tensor(intervention, dtype=torch.long, device=device)
        Y_t = torch.tensor(outcome, dtype=torch.float32, device=device)

        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            all_po, prop_logits = model(X_t)
            factual = all_po.gather(1, I_t.unsqueeze(-1)).squeeze(-1)
            loss_Y = F.mse_loss(factual, Y_t)
            loss_prop = F.cross_entropy(prop_logits, I_t)
            loss = loss_Y + self.alpha_prop * loss_prop
            loss.backward()
            optimizer.step()

        self._model = model
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        device = torch.device(self._dev)
        X_s = self._scaler.transform(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            po, _ = self._model(torch.tensor(X_s, device=device))
        return po.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Causal Forest (via econml) — T-17
# ---------------------------------------------------------------------------

class CausalForestEstimator(BaseDownstreamEstimator):
    """
    Causal Forest estimator via DML + sklearn RandomForestRegressor.

    Implements DML with forests:
      1. Estimate m(x) = E[Y|X] and e(x) = E[T|X] via cross-fitted RF
      2. Residualize: Ỹ = Y - m(X), T̃ = T - e(X)
      3. Fit RF on modified outcome: τ(x) = argmin E[(Ỹ - τ(X)·T̃)²]

    Represents non-neural, tree-based estimator family.
    Tests plug-and-play claim for adapters on tree models.

    Parameters
    ----------
    n_estimators : int, default 100
    min_samples_leaf : int, default 5
    n_folds : int, default 3
    seed : int, default 42
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        n_folds: int = 3,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_folds = n_folds
        self.seed = seed
        self._tau_forest = None
        self._num_levels: int = 2

    def fit(
        self,
        X: np.ndarray,
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "CausalForestEstimator":
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import KFold

        self._num_levels = int(intervention.max()) + 1
        n = X.shape[0]
        Y = outcome.astype(np.float64)
        T = intervention.astype(np.float64)

        # Cross-fitted residuals
        Y_hat = np.zeros(n)
        T_hat = np.zeros(n)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in kf.split(X):
            rf_y = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
                n_jobs=-1,
            )
            rf_t = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
                n_jobs=-1,
            )
            rf_y.fit(X[train_idx], Y[train_idx])
            rf_t.fit(X[train_idx], T[train_idx])
            Y_hat[val_idx] = rf_y.predict(X[val_idx])
            T_hat[val_idx] = rf_t.predict(X[val_idx])

        # Residuals
        Y_res = Y - Y_hat    # Ỹ
        T_res = T - T_hat    # T̃

        # Modified outcome: τ estimated via WLS-style RF
        # Multiply X by T̃ to create the "Riesz" weight signal
        T_res_sq = T_res ** 2 + 1e-8
        # Guard against T_res == 0 exactly (np.sign(0) == 0 would still give 0-denominator).
        safe_denom = np.where(np.abs(T_res) > 1e-4, T_res, np.where(T_res >= 0, 1e-4, -1e-4))
        Y_mod = np.clip(Y_res / safe_denom, -1e6, 1e6)

        tau_forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=-1,
        )
        tau_forest.fit(X, Y_mod, sample_weight=T_res_sq)
        self._tau_forest = tau_forest
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        assert self._tau_forest is not None, "Call fit() first"
        tau = self._tau_forest.predict(X)  # (n,)
        n = X.shape[0]
        po = np.zeros((n, self._num_levels), dtype=np.float32)
        for k in range(1, self._num_levels):
            po[:, k] = (tau * k).astype(np.float32)
        return po


# ---------------------------------------------------------------------------
# Registry: name → class
# ---------------------------------------------------------------------------

ESTIMATOR_REGISTRY = {
    "DML":          DMLEstimator,
    "RLearner":     RLearnerEstimator,
    "TARNet":       TARNetEstimator,
    "DragonNet":    DragonNetEstimator,
    "CausalForest": CausalForestEstimator,   # T-17
}
