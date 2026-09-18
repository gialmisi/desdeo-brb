# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-18

### Added

- **Incomplete rules.** A rule's belief degrees may now sum to less than one,
  the shortfall being its ignorance about that consequent, as RIMER requires
  (Yang et al. 2006, Eq. 3). Validation previously demanded exactly one, so a
  rule base could not express an expert who is sure about one region and vague
  about another.
  - `RuleBase.ignorance` and `RuleBase.is_complete`.
  - `compute_utility_bounds`, the utility interval of Yang and Xu (2002,
    Section II-H). Belief left unassigned could belong to any grade, so an
    incomplete assessment bounds the output rather than fixing it.
  - `InferenceResult.utility_bounds`, `.ignorance` and `.is_complete`, filled
    in by both the NumPy and the JAX prediction paths. `to_dict()` emits the
    first two alongside the rest of the trace.
  - `fit(..., allow_incomplete=...)` chooses between the sum-to-one equality
    and the cap of Yang et al. (2007), constraint 12b. The default, `None`,
    follows the rule base being trained.
- **Several consequent attributes.** A rule base can predict more than one
  output, each with its own grades. Pass `crv` as a list of arrays, one per
  output, and give `fit` a target of shape `(n_samples, n_outputs)`.
  - `RuleBase.consequent_group_sizes` delimits the concatenated grades;
    `n_outputs`, `group_sizes`, `consequent_slices`, `consequent_values()`,
    `beliefs_for()` and `block_sums` address one output at a time.
  - `InferenceResult` gains the matching `consequent_group_sizes`, `n_outputs`
    and `consequent_slices`, and `explain()` prints one belief block per output.
  - `RuleBase.describe_rule()` and `describe_all_rules()` likewise give each
    output its own distribution. `consequent_name` accepts a sequence of names,
    one per output, and refuses a single name for several outputs rather than
    guessing which objective it belongs to.
  - Activation weights depend only on the antecedents, so they are computed
    once and the evidential reasoning combination runs once per output.
    Completeness and ignorance are per rule per output.
  - `fit(..., scale_outputs=...)` divides each output's residual by the span of
    its own grades before squaring, so an objective measured in hundreds does
    not crowd out one measured in tenths. Pass `False` for the raw sum.
  - Supported on all three backends: NumPy, JAX and Pyomo/IPOPT.
- **Extended antecedents.** A rule may carry `antecedent_beliefs`, a belief
  distribution over each attribute's referential values, in place of
  `rule_antecedent_indices`, so that it sits between referential values rather
  than only at them. This is the extended belief rule base of Liu et al. (2008).
  - `compute_extended_activation_weights` matches two distributions by distance
    following Zhuang et al. (2021), Eqs. (7) to (9).
  - `RuleBase.is_extended` and `RuleBase.conventional_indices`.
  - `describe_rule()`, `describe_all_rules()` and `explain()` all work on an
    extended rule base, showing each rule's distribution over the referential
    values in place of the single value it does not have.
  - `fit()` trains an extended rule base on the NumPy backend. Belief degrees
    and weights train as usual; the referential values are pinned, since the
    antecedent distributions were computed against them and would go stale if
    they moved.
  - Integration tests reproduce the Liu-EBRB column of Zhuang et al. (2021),
    Table 2, to within a point on Iris, Ecoli and Glass. Their fourth dataset,
    Pima, is not covered because UCI withdrew it.
- Documentation for all three features: an Incomplete rules, a Several outputs
  and an Extended antecedents section in the training guide, the backend
  restrictions in the backends guide, the new references, and a rewritten
  explainability guide covering several outputs, utility bounds and ignorance,
  and how the description helpers read an extended rule base.
- `notebooks/05_incomplete_and_multi_output.ipynb`, a worked example of
  ignorance, utility bounds and several consequent attributes, and of how the
  two compose: completeness is tracked per rule per output, so a rule base can
  be confident about one objective and vague about another.
- A `network` pytest marker for tests that download a dataset, which skip when
  the fetch fails.
- The notebooks are executed in CI by `nbmake`, which is now part of the `dev`
  extra. Nothing previously checked that the tutorials still ran against the
  API. Run them locally the same way with `pytest --nbmake notebooks/`.

### Changed

- Rows of `RuleBase.belief_degrees` must now be non-negative and sum to at most
  one, where they previously had to sum to exactly one. This widens what
  validation accepts, so existing rule bases keep working.
- The scalar output is the average expected utility of the combined
  distribution. For a complete assessment this is the plain weighted sum as
  before; for an incomplete one it is the midpoint of the utility interval.
- `BRBModel.score()` scales each output's residual by the span of that output's
  grades. A single output is unscaled, as before.

### Fixed

- `InferenceResult.explain()` and `BRBModel.explain()` ignored their
  `consequent_name` argument entirely: it was accepted, documented as passed
  through to `describe_rule`, and then never used, since `explain` builds its
  own rule descriptions. The name now labels the prediction and the combined
  belief distribution. This changes the output of `explain()` for callers who
  were already passing the argument and silently getting nothing for it.

### Known limitations

- Extended antecedents are NumPy only. `backend="jax"` and `method="ipopt"`
  both gather one referential index per attribute, which an extended rule base
  does not have, and refuse it rather than silently running a different model.
  `fix_endpoint_beliefs` likewise raises, since no rule sits on a referential
  value and so none of them is a boundary rule.
- An antecedent belief distribution must sum to one. Distance-based matching
  halves the squared distance so that two disjoint distributions are exactly
  one apart, which holds only when both carry the same mass; a rule with less
  of it would match everything better. Liu et al. (2008) and Zhuang et al.
  (2021) both admit a sum below one, but neither defines the distance for it,
  so it is refused rather than computed wrongly.

### Dependencies

- Added `scikit-learn` and `ucimlrepo` to the `dev` extra, for the benchmark
  datasets used by the published-accuracy tests, and `nbmake`, for executing
  the notebooks. No changes to the core runtime dependencies.

[1.1.0]: https://github.com/gialmisi/desdeo-brb/releases/tag/v1.1.0

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
