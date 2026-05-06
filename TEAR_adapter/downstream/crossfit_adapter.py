"""
Cross-fitting Boundary Integrity (修改 E)
==========================================
Proper foldwise adapter fit for DML / RLearner.

问题：
  当前 pipeline:
    adapter.fit(X_train)        ← 看到全部训练数据（包括 fold validation 集）
    X_rep = adapter.transform(X_train)
    DML cross-fitting on X_rep  ← fold-level nuisance models 用的表示由看过全数据的 encoder 生成

  这构成 representation-level 轻微泄露：encoder 权重利用了 fold validation 的
  X 信息（虽然没有直接利用该 fold 的 (I, Y) 标签）。

  严格正确的做法：
    每个 fold 单独 fit adapter（仅用 train split），再 transform 该 fold 的 val split。

FoldwiseDMLEstimator / FoldwiseRLearnerEstimator:
  包装 DML / RLearner，在 cross-fitting 内嵌 per-fold adapter refit。

GlobalAdapterDMLEstimator / GlobalAdapterRLearnerEstimator:
  对照版：全局 fit adapter 再 cross-fitting（当前行为）。

用于 run_crossfit_integrity_check.py 实验。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Type

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ..data.schemas import FeatureMetadata
from ..adapters.base import BaseAdapter
from .estimators import BaseDownstreamEstimator


# ---------------------------------------------------------------------------
# Base class for foldwise vs global adapter fitting
# ---------------------------------------------------------------------------

class _AdapterDMLBase(BaseDownstreamEstimator):
    """
    DML estimator that accepts an adapter class + config and handles adapter
    fitting either globally or per-fold.

    Parameters
    ----------
    adapter_class : class
        Adapter class (e.g. SafeTemporalRoleAdapter or LegacyTemporalRoleAdapter).
    adapter_kwargs : dict
        Keyword arguments passed to adapter_class(**adapter_kwargs) per fold.
    output_mode : str
        'confounding' or 'safe_full' or 'full_legacy'.
    n_folds : int
        Number of cross-fitting folds.
    reg_alpha : float
        Ridge / logistic regularization.
    seed : int
    foldwise : bool
        If True, re-fit adapter per fold (strict).
        If False, fit adapter once on full training data (legacy).
    """

    def __init__(
        self,
        adapter_class,
        adapter_kwargs: dict,
        metadata: List[FeatureMetadata],
        output_mode: str = "confounding",
        n_folds: int = 3,
        reg_alpha: float = 1.0,
        seed: int = 42,
        foldwise: bool = True,
    ):
        self.adapter_class = adapter_class
        self.adapter_kwargs = adapter_kwargs
        self.metadata = metadata
        self.output_mode = output_mode
        self.n_folds = n_folds
        self.reg_alpha = reg_alpha
        self.seed = seed
        self.foldwise = foldwise

        self._scaler = StandardScaler()
        self._prop_model: Optional[LogisticRegression] = None
        self._outcome_models: dict = {}
        self._num_levels: int = 0
        # Global adapter (used for transform at prediction time)
        self._global_adapter: Optional[BaseAdapter] = None
        self._global_output_mode: str = output_mode

    def fit(
        self,
        X: np.ndarray,
        intervention: np.ndarray,
        outcome: np.ndarray,
    ) -> "_AdapterDMLBase":
        self._num_levels = int(intervention.max()) + 1
        n = X.shape[0]

        # Always fit a global adapter for prediction-time transform
        global_adapter = self.adapter_class(**self.adapter_kwargs)
        global_adapter.fit(X, self.metadata, intervention, outcome)
        self._global_adapter = global_adapter

        # Get representations for cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        X_rep_crossfit = np.zeros((n, self._rep_dim(global_adapter, X)), dtype=np.float32)

        for train_idx, val_idx in kf.split(X):
            if self.foldwise:
                # Modification E: re-fit adapter per fold (strict)
                fold_adapter = self.adapter_class(**self.adapter_kwargs)
                fold_adapter.fit(
                    X[train_idx], self.metadata,
                    intervention[train_idx], outcome[train_idx]
                )
                fold_out = fold_adapter.transform(X[val_idx], self.metadata)
            else:
                # Legacy: use global adapter (mild leakage)
                fold_out = global_adapter.transform(X[val_idx], self.metadata)

            X_rep_crossfit[val_idx] = fold_out.repr_for_estimator(self.output_mode)

        X_s = self._scaler.fit_transform(X_rep_crossfit)

        # Nuisance models on cross-fitted representations
        self._prop_model = LogisticRegression(
            C=1.0 / max(self.reg_alpha, 1e-6), max_iter=500, random_state=self.seed
        ).fit(X_s, intervention)

        for k in range(self._num_levels):
            mask = intervention == k
            if mask.sum() > 1:
                self._outcome_models[k] = Ridge(alpha=self.reg_alpha).fit(
                    X_s[mask], outcome[mask]
                )
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        assert self._global_adapter is not None
        out = self._global_adapter.transform(X, self.metadata)
        X_rep = out.repr_for_estimator(self.output_mode)
        X_s = self._scaler.transform(X_rep)
        n = X.shape[0]
        PO = np.zeros((n, self._num_levels), dtype=np.float32)
        for k in range(self._num_levels):
            if k in self._outcome_models:
                PO[:, k] = self._outcome_models[k].predict(X_s).astype(np.float32)
        return PO

    def _rep_dim(self, adapter, X: np.ndarray) -> int:
        out = adapter.transform(X[:2], self.metadata)
        return out.repr_for_estimator(self.output_mode).shape[1]


# ---------------------------------------------------------------------------
# Concrete foldwise vs global classes
# ---------------------------------------------------------------------------

class FoldwiseAdapterDML(_AdapterDMLBase):
    """Strict foldwise adapter fit (Modification E — correct version)."""
    def __init__(self, adapter_class, adapter_kwargs, metadata, **kwargs):
        super().__init__(adapter_class, adapter_kwargs, metadata,
                         foldwise=True, **kwargs)


class GlobalAdapterDML(_AdapterDMLBase):
    """Global adapter fit (legacy behavior — potential mild leakage)."""
    def __init__(self, adapter_class, adapter_kwargs, metadata, **kwargs):
        super().__init__(adapter_class, adapter_kwargs, metadata,
                         foldwise=False, **kwargs)
