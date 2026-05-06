# NIPSCODETREAT

Main entry points:
- `newtry_llm_orbit.adapter.ASOrbitAdapter`
- `newtry_llm_orbit.adapter.ASOrbitConfig`
- `newtry_llm_orbit.semantic_engine.SemanticEngineConfig`
- `orbit_adapter.adapters.temporal_role_adapter_safe.SafeTemporalRoleAdapter`
- `orbit_adapter.adapters.temporal_role_adapter_safe.SafeAdapterConfig`

Minimal usage sketch:

```python
from newtry_llm_orbit.adapter import ASOrbitAdapter, ASOrbitConfig
from newtry_llm_orbit.semantic_engine import SemanticEngineConfig

semantic_cfg = SemanticEngineConfig(cache_dir="./semantic_cache", backend="heuristic_self_consistency")
adapter = ASOrbitAdapter(semantic_config=semantic_cfg, config=ASOrbitConfig())
adapter.fit(X_train, metadata, treatment_train, outcome_train, dataset_name="my_dataset")
z_train = adapter.transform(X_train).repr_for_estimator("safe_dense_view")
z_test = adapter.transform(X_test).repr_for_estimator("safe_dense_view")
```

Dependencies are the same as the original algorithm implementation, mainly `numpy`, `torch`, `scikit-learn`, and optional LLM backend dependencies if `backend="local_llm"` is used.
