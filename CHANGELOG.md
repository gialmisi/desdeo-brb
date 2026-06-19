# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-06-19

### Added

- `py.typed` marker so type information ships with the package and is picked up
  by downstream type checkers (PEP 561), matching the `Typing :: Typed`
  classifier.
- Community health files: GitHub issue and pull request templates.

### Fixed

- Corrected the Zenodo DOI in the README badge to match `CITATION.cff`.
- `CONTRIBUTING.md` now references the correct default branch (`master`) and the
  docstring style actually used in the codebase (Google, matching `mkdocs.yml`).

### Dependencies

- Bumped several development, docs, and notebook dependencies via Dependabot
  (e.g. `jupyterlab`, `notebook`, `jupyter-server`, `tornado`, `bleach`,
  `mistune`, `urllib3`, `idna`, `pymdown-extensions`). No changes to the core
  runtime dependencies.

[1.0.1]: https://github.com/gialmisi/desdeo-brb/releases/tag/v1.0.1

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
