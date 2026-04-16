# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-16

### Added

- Initial stable release of `desdeo-brb` 1.x.x, a trainable Belief Rule-Based inference
  system implementing the RIMER methodology (Yang et al. 2006; Chen et al. 2011).
- Core `BRBModel` class with scikit-learn-compatible `fit()` and `predict()` API.
- NumPy backend with SLSQP and trust-constr optimizers for standard MSE training.
- JAX backend with L-BFGS-B and automatic differentiation for fast training of
  large models.
- Pyomo/IPOPT backend for use with custom symbolic objectives.
- Differential Evolution (`DE`) and hybrid `DE+SLSQP` training methods for
  non-convex problems.
- Multi-start optimization via `n_restarts` parameter to handle local minima.
- Adaptive referential value training as described in Chen et al. (2011).
- Explainability features: `describe_rule()`, `describe_all_rules()`,
  `InferenceResult.explain()`, and `BRBModel.explain()` for human-readable
  rule descriptions and prediction traces.
- Custom loss function support via `fit_custom()` for domain-specific objectives
  such as INFRINGER-style value function learning.
- Four Jupyter notebooks covering getting started, multi-attribute models,
  expert knowledge integration with pipeline leak detection, and explainability.

### Dependencies

- Core: `numpy>=1.24`, `scipy>=1.10`, `pydantic>=2.0`
- Optional: `jax` (for JAX backend), `pyomo` (for IPOPT backend),
  `jupyter` + `matplotlib` (for running the notebooks)

[1.0.0]: https://github.com/gialmisi/desdeo-brb/releases/tag/v1.0.0
