# Training

BRB training minimises mean-squared error subject to several constraints:
each rule's belief degrees sum to 1 for every output, rule weights sum to 1,
attribute weights are non-negative, and referential values stay sorted. The
loss landscape is non-convex with many local minima.

The belief sum is an equality only when the trained rule base is meant to come
out complete. See [Incomplete rules](#incomplete-rules) for relaxing it.

## Training methods

`BRBModel.fit(..., method=...)` accepts the following methods:

| Method | Type | Best for | Requires |
|--------|------|----------|----------|
| `SLSQP` (default) | Local, constrained | Small models with `n_restarts` | NumPy, SciPy |
| `trust-constr` | Local, constrained | Alternative to SLSQP | NumPy, SciPy |
| `DE` | Global, evolutionary | Large models, complex landscapes | NumPy, SciPy |
| `DE+SLSQP` | Global + local polish | Reliable single-run training | NumPy, SciPy |
| `ipopt` | Local, interior-point | Custom Pyomo objectives | `desdeo-brb[pyomo]` + IPOPT |
| JAX backend | Local, autodiff | Fast iteration, large datasets | `desdeo-brb[jax]` |

## Choosing a method

For most problems `method="SLSQP"` with `n_restarts=10` is the best balance of
speed and solution quality. The 10 independent starts almost always find a
near-global optimum for problems with up to ~50 rules.

If SLSQP struggles (slow progress, poor MSE), try:

- `DE+SLSQP` — global exploration followed by local refinement
- `ipopt` — interior-point solver with exact Hessians; often finds
  smoother solutions
- JAX backend — exact gradients via automatic differentiation; much
  faster per iteration on large models

## Multi-start: `n_restarts`

BRB loss landscapes typically have multiple basins, some much worse than
others. A single SLSQP run can converge to a bad basin even from a good
initial guess. Setting `n_restarts > 1` runs the optimiser from several
random perturbations of the initial parameters and keeps the best result.

```python
model.fit(X_train, y_train, n_restarts=10)
```

## Fixing endpoints

Referential values at the domain boundaries are usually fixed (e.g., the
minimum and maximum of the input range). Set `fix_endpoints=True` (the
default) to prevent the optimiser from moving them.

To also pin the belief degrees of boundary rules (useful when initial
boundary beliefs are known to be correct from `initial_rule_fn`), pass
`fix_endpoint_beliefs=True`.

## Custom optimiser options

All methods accept per-optimiser options through `optimizer_options`:

```python
# SLSQP with a tighter tolerance
model.fit(X, y, method="SLSQP",
          optimizer_options={"maxiter": 2000, "ftol": 1e-12})

# Two-phase DE+SLSQP with per-phase options
model.fit(X, y, method="DE+SLSQP",
          optimizer_options={
              "de":    {"maxiter": 300, "seed": 42},
              "slsqp": {"maxiter": 1000, "ftol": 1e-12},
          })

# IPOPT
model.fit(X, y, method="ipopt",
          optimizer_options={"max_iter": 5000, "tol": 1e-9})
```

## Incomplete rules

RIMER requires only that a rule's belief degrees sum to *at most* one
[@YangEtAl2006]. A shortfall is the rule's ignorance about its consequent,
and the evidential reasoning combination carries it through to the result.
Yang et al. impose the sum-to-one equality only when a complete trained rule
base is wanted; otherwise the sum is capped at one [@YangEtAl2007].

`allow_incomplete` controls which of the two applies:

```python
# Rules may leave belief unassigned
model.fit(X, y, allow_incomplete=True)
```

The default, `None`, follows the rule base being trained. A rule base that
arrives incomplete keeps that freedom, so training does not silently discard
ignorance an expert deliberately expressed; one that arrives complete stays
complete, so it gains no ignorance nobody asked for. Pass `True` or `False`
to override.

Two things follow from allowing it:

- The prediction becomes the *average expected utility* of the utility
  interval [@YangXu2002], not a plain weighted sum. `InferenceResult` exposes
  the interval as `utility_bounds` and the unassigned mass as `ignorance`.
- Ignorance becomes a free parameter. Since total ignorance predicts the
  midpoint of the utility range, declining to commit only lowers the loss for
  a model already worse than predicting the mean, so it is a weak attractor
  rather than a shortcut.

A rule base that arrives complete is seeded with a small starting ignorance,
because the gradient of the loss with respect to the underlying parameters
scales with the ignorance itself and a smaller seed would leave the optimiser
with nothing to follow.

## Several outputs

A rule base can predict more than one consequent attribute. Pass a list of
referential value arrays, one per output, and give `fit` a target of shape
`(n_samples, n_outputs)`:

```python
model = BRBModel(prv, [np.array([0.0, 0.5, 1.0]),
                       np.array([100.0, 250.0, 400.0])])
model.fit(X, y)          # y has shape (n_samples, 2)
```

Each output keeps its own grades, because objectives generally have their own
scales and units. The activation weights depend only on the antecedents, so
they are computed once and the evidential reasoning combination runs once per
output: what a rule believes about one objective places no constraint on what
it believes about another. Completeness and ignorance are likewise per rule
per output.

By default each output's residual is divided by the span of that output's own
grades before squaring, so an objective measured in hundreds does not crowd
out one measured in tenths. Pass `scale_outputs=False` to sum the raw squared
errors instead.

This is a fixed weighting taken from the rule base. Yang et al. instead
normalise each objective between its own ideal and its worst value in a
payoff table and minimise the largest scaled error, which yields an efficient
solution at the cost of one extra training run per output
[@YangEtAl2007].

## Custom loss functions

`fit_custom(loss_fn)` lets you optimise any scalar loss of the model. This
is how INFRINGER-style value function learning is implemented [@Misitano2020].

```python
def my_loss(model):
    # model is a BRBModel whose parameters have just been updated
    # by the optimiser. Return a scalar loss.
    y_pred = model.predict_values(X_train)
    return float(np.mean((y_train - y_pred) ** 2))

model.fit_custom(my_loss, fix_endpoints=True, n_restarts=5)
```

`fit_custom` accepts the same `method`, `optimizer_options`, `n_restarts`,
and other parameters as `fit`. The structural BRB constraints are always
enforced.
