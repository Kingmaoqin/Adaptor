# NIPSCODETREAT

This folder contains a cleaned algorithm-only code snapshot for the NeurIPS paper.

Included:
- `newtry_llm_orbit/`: Adaptive Semantic ORBIT wrapper, semantic prior engine, cached semantic record schemas, and heuristic/local LLM backends.
- `orbit_adapter/adapters/`: ORBIT adapters, including `SafeTemporalRoleAdapter`, safe temporal routing variants, rule filters, and baseline adapters used by the algorithm.
- `orbit_adapter/models/`: feature encoder, temporal eligibility gate, safe role router, subspace aggregation, role adapters, and outcome heads.
- `orbit_adapter/objectives/`: balance, orthogonality, bridge, disentanglement, audit, and gate-ranking losses.
- `orbit_adapter/training/`: core training loops and cross-fitting utilities.
- `orbit_adapter/downstream/`: downstream estimator wrappers used to consume ORBIT representations.
- `orbit_adapter/data/schemas.py`: feature metadata and dataset schema types required by the algorithm.
- `orbit_adapter/evaluation/`: algorithm diagnostics and routing/gate metrics.

Excluded:
- Experiment runners, paper table/figure scripts, generated results, logs, caches, PDFs, and review/audit writeups.
- Dataset-specific raw/cohort builders and result folders.
- Test and ablation-output folders.

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
