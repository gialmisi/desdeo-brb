# desdeo-brb

[![Tests](https://github.com/gialmisi/desdeo-brb/actions/workflows/test.yml/badge.svg)](https://github.com/gialmisi/desdeo-brb/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/desdeo-brb.svg)](https://pypi.org/project/desdeo-brb/)
[![Python versions](https://img.shields.io/pypi/pyversions/desdeo-brb.svg)](https://pypi.org/project/desdeo-brb/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/gialmisi/desdeo-brb/branch/master/graph/badge.svg)](https://codecov.io/gh/gialmisi/desdeo-brb)

A trainable Belief Rule-Based (BRB) inference system with an sklearn-compatible API.

> **Note:** `desdeo` is part of the package name for historical reasons. This
> library is standalone and has no dependency on the [DESDEO
> framework](https://github.com/industrial-optimization-group/DESDEO).

## Overview

Belief Rule-Based (BRB) systems are a generalization of traditional IF-THEN rule
bases where each rule's consequent is expressed as a belief distribution over
possible outcomes rather than a single crisp value. Inference is performed using
the evidential reasoning (ER) algorithm, which analytically combines activated
rules into a final output distribution. Unlike black-box models, every
intermediate quantity: which rules fired, how strongly, and how beliefs were
combined; is directly inspectable and interpretable.

This library provides a BRB implementation that is trainable from data (belief
degrees, rule weights, attribute weights, and referential values are all
optimizable), supports varying-length referential values per attribute, and
offers an optional JAX backend for JIT-compiled inference and exact-gradient
training via autodiff. The `BRBModel` class follows the
[scikit-learn estimator interface](https://scikit-learn.org/stable/developers/develop.html)
(`fit`, `predict`, `score`, `get_params`, `set_params`), making it easy to
integrate into existing ML workflows and pipelines.

## Installation

```bash
pip install desdeo-brb          # Core (NumPy + SciPy)
pip install desdeo-brb[jax]     # + JAX backend
pip install desdeo-brb[pyomo]   # + Pyomo/IPOPT backend
pip install desdeo-brb[all]     # Everything
```

The IPOPT backend requires IPOPT binaries installed separately:

```bash
apt install coinor-libipopt-dev   # Debian/Ubuntu
conda install -c conda-forge ipopt  # conda
```

## Quick start

```python
import numpy as np
from desdeo_brb import BRBModel

# Define the function to model: f(x) = x * sin(x^2) on [0, 3]
f = lambda x: x[0] * np.sin(x[0] ** 2)

# Discretize input and output spaces
prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])

# Construct model (initial beliefs computed from f at referential values)
model = BRBModel(prv, crv, initial_rule_fn=f)

# Generate training data
rng = np.random.default_rng(42)
X_train = rng.uniform(0, 3, size=(1000, 1))
y_train = X_train[:, 0] * np.sin(X_train[:, 0] ** 2)

# Train (multiple restarts for reliable results)
model.fit(X_train, y_train, n_restarts=5)

# Predict
X_test = np.linspace(0, 3, 100).reshape(-1, 1)
result = model.predict(X_test)

print("Outputs:", result.output[:5])
print("Top-3 rules for first sample:", result.dominant_rules(top_k=3)[0])
```

See `notebooks/01_getting_started.ipynb` for a full walkthrough with plots.

## API reference

`BRBModel` follows the [scikit-learn estimator interface](https://scikit-learn.org/stable/developers/develop.html) and can be used anywhere an sklearn estimator is expected (e.g., `cross_val_score`, `GridSearchCV`, pipelines).

| Symbol | Description |
|--------|-------------|
| `BRBModel(prv, crv, ...)` | Constructor. Pass referential values and optionally a `RuleBase`, `initial_rule_fn`, `utility_fn`, or `backend="jax"`. |
| `.fit(X, y, method=..., optimizer_options=..., n_restarts=...)` | Train by minimizing MSE. NumPy backend supports `"SLSQP"` (default), `"trust-constr"`, `"DE"` (differential evolution), `"DE+SLSQP"` (global search then local polish), and `"ipopt"` (requires `desdeo-brb[pyomo]`); JAX backend uses `"L-BFGS-B"` with exact gradients. `"DE+SLSQP"` is recommended for reliable training without needing `n_restarts`. Pass `optimizer_options` to override defaults; for `"DE+SLSQP"` use sub-dicts `{"de": {...}, "slsqp": {...}}`. |
| `.fit_custom(loss_fn)` | Train with a user-supplied loss function. |
| `.predict(X)` | Full inference. Returns an `InferenceResult` with all intermediate quantities. |
| `.predict_values(X)` | Scalar outputs only, shape `(n_samples,)`. |
| `.score(X, y)` | Negative MSE (sklearn convention: higher is better). |
| `.get_params()` / `.set_params()` | Sklearn-compatible parameter access. |
| `.rule_base` | The current `RuleBase` (belief degrees, weights, referential values). |

**`InferenceResult` fields:**

| Field | Description |
|-------|-------------|
| `output` | Scalar predictions, shape `(n_samples,)` |
| `activation_weights` | Per-rule activation, shape `(n_samples, n_rules)` |
| `combined_belief_degrees` | Output belief distribution, shape `(n_samples, n_consequents)` |
| `input_belief_distributions` | Per-attribute input beliefs (list of arrays) |
| `dominant_rules(top_k)` | Indices of the top-k most activated rules per sample |
| `to_dict()` | JSON-serializable summary |

See source docstrings for full details.

## Key concepts

**Referential values** are the discrete anchor points that define the input and output spaces. Each input attribute has its own set of referential values (which may differ in number), and the output has a separate set of consequent referential values. The Cartesian product of all input referential values defines the set of rules.

**Belief degrees** express each rule's consequent as a probability distribution over the consequent referential values. For example, a rule might say "if temperature is High, then risk is {Low: 0.1, Medium: 0.7, High: 0.2}." Each row of the belief degree matrix sums to 1.

**Activation weights** measure how strongly each rule matches a given input. When the input falls exactly on a rule's antecedent referential values, that rule gets full activation. Between referential values, adjacent rules share activation proportionally.

**Combined belief degrees** are computed by the evidential reasoning algorithm, which analytically aggregates the activated rules' belief distributions into a single output distribution. The scalar output is then computed as a weighted average of the consequent values using this distribution.

**Training** optimizes belief degrees, rule weights, attribute weights, and optionally the referential value positions themselves. All parameters are subject to constraints: belief rows sum to 1, rule weights sum to 1, attribute weights are non-negative, and referential values remain sorted.

See the notebooks for worked examples with mathematical context.

## Training methods

| Method | Type | Best for | Requires |
|--------|------|----------|----------|
| `SLSQP` (default) | Local, constrained | Small models with `n_restarts` | NumPy, SciPy |
| `trust-constr` | Local, constrained | Alternative to SLSQP | NumPy, SciPy |
| `DE` | Global, evolutionary | Large models, complex landscapes | NumPy, SciPy |
| `DE+SLSQP` | Global + local polish | Reliable single-run training | NumPy, SciPy |
| `ipopt` | Local, interior-point | Custom Pyomo objectives | `desdeo-brb[pyomo]` + IPOPT |
| JAX backend | Local, autodiff | Fast iteration, large datasets | `desdeo-brb[jax]` |

For most problems, `method="SLSQP"` with `n_restarts=10` provides the best balance of speed and solution quality. BRB training is a non-convex optimization problem with multiple local minima, so multiple restarts are strongly recommended.

## References

The implementation follows the RIMER (Rule-base Inference Methodology using the
Evidential Reasoning approach) framework. Key papers:

1. Yang, J.-B., Liu, J., Wang, J., Sii, H.-S., & Wang, H.-W. (2006). Belief
   rule-base inference methodology using the evidential reasoning approach — RIMER.
   *IEEE Transactions on Systems, Man, and Cybernetics — Part A*, 36(2), 266-285.
2. Yang, J.-B., Liu, J., Xu, D.-L., Wang, J., & Wang, H.-W. (2007). Optimization
   models for training belief-rule-based systems. *IEEE Transactions on Systems,
   Man, and Cybernetics — Part A*, 37(4), 569-585.
3. Chen, Y.-W., Yang, J.-B., Xu, D.-L., Zhou, Z.-J., & Tang, D.-W. (2011).
   Inference analysis and adaptive training for belief rule based systems. *Expert
   Systems with Applications*, 38(10), 12845-12860.
4. Xu, D.-L., Liu, J., Yang, J.-B., Liu, G.-P., Wang, J., Jenkinson, I., & Ren,
   J. (2007). Inference and learning methodology of belief-rule-based expert system
   for pipeline leak detection. *Expert Systems with Applications*, 32(1), 103-113.
5. Misitano, G. (2020). Interactively learning the preferences of a decision
   maker in multi-objective optimization utilizing belief-rules. *IEEE SSCI 2020*,
   133-140.

## Citation

If using this software in academic work, please cite:

```
Misitano, G. (2020). Interactively Learning the Preferences of a Decision Maker
in Multi-objective Optimization Utilizing Belief-rules. IEEE SSCI 2020.

Chen, Y.-W., Yang, J.-B., Xu, D.-L., Zhou, Z.-J., & Tang, D.-W. (2011).
Inference analysis and adaptive training for belief rule based systems.
Expert Systems with Applications, 38(10), 12845-12860.
```

## Maintainer

Giovanni Misitano (@gialmisi)

## License

MIT
