"""
temporal_role_adapter_legacy.py
================================
Legacy adapter — re-exports from temporal_role_adapter.py unchanged.
Used for old_full / old_confounding ablation conditions.

Do NOT use as the default adapter in new experiments.
The current main version is SafeTemporalRoleAdapter in temporal_role_adapter_safe.py.
"""
from .temporal_role_adapter import (
    TemporalRoleAdapter as LegacyTemporalRoleAdapter,
    RobustTemporalRoleAdapter as LegacyRobustTemporalRoleAdapter,
    AdapterTrainingConfig as LegacyAdapterTrainingConfig,
    _AdapterCore as _LegacyAdapterCore,
)

__all__ = [
    "LegacyTemporalRoleAdapter",
    "LegacyRobustTemporalRoleAdapter",
    "LegacyAdapterTrainingConfig",
]
