# Training

BRB training minimises mean-squared error subject to several equality
constraints: belief-degree rows sum to 1, rule weights sum to 1, attribute
weights are non-negative, and referential values stay sorted. The loss
landscape is non-convex with many local minima.

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
