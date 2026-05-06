from .identity_adapter import IdentityAdapter
from .simple_prefilter_adapter import SimplePrefilterAdapter
from .heuristic_role_adapter import HeuristicRoleAdapter
from .temporal_role_adapter import TemporalRoleAdapter, RobustTemporalRoleAdapter
from .metadata_filter_adapter import MetadataFilterAdapter
from .shared_repr_adapter import SharedReprAdapter

__all__ = [
    "IdentityAdapter",
    "SimplePrefilterAdapter",
    "HeuristicRoleAdapter",
    "TemporalRoleAdapter",
    "RobustTemporalRoleAdapter",
    "MetadataFilterAdapter",   # T-11: rule-based temporal filter (no learning)
    "SharedReprAdapter",       # T-10: encoder without role routing (ablation)
]
