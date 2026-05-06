"""
orbit_adapter — Temporal Role Routing Adapter for Treatment Effect Estimation.

This is an estimator-agnostic plug-in that restructures high-dimensional
mixed-role covariates before any downstream estimator.  The core components
are the temporal eligibility gate and 5-way role router.  LLM auditor,
proxy bridge, and audit energy are intentionally absent in this branch.

orbit_plus is preserved unchanged as the original ORBIT+ estimator branch.
"""

