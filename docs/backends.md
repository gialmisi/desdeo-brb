# Backends

`desdeo-brb` supports three computational backends that differ in how
inference and gradient computations are performed. The backend is chosen
at model construction time:

```python
model = BRBModel(prv, crv, backend="numpy")  # default
model = BRBModel(prv, crv, backend="jax")
```

The Pyomo/IPOPT backend is activated per training call via
`method="ipopt"` and does not require a separate `backend=...` setting.

## NumPy backend (default)

Pure NumPy implementation of the ER inference pipeline. The training methods
`SLSQP`, `trust-constr`, `DE`, and `DE+SLSQP` all use this backend with
finite-difference gradients.

**When to use:** the default choice. Excellent for small to medium models
(up to ~100 rules) and for prototyping.

## JAX backend

JIT-compiled inference and training via L-BFGS-B with exact gradients
computed by `jax.grad`. Uses a differentiable reparameterization
(softmax for belief degrees / rule weights, softplus for attribute weights,
sort for referential values) so box-constrained L-BFGS-B can replace the
equality-constrained SLSQP.

**When to use:** models with many rules or large training sets. JIT
compilation has a one-time cost of a few seconds, but each subsequent
iteration is much faster than NumPy with finite differences. A benchmark
with 64 rules and 500 samples shows ~37x speed-up over SLSQP.

```python
model = BRBModel(prv, crv, backend="jax")
model.fit(X, y, n_restarts=5)
```

## Pyomo / IPOPT backend

Builds a symbolic optimisation model using Pyomo and solves it with IPOPT.
IPOPT uses exact Hessians and is often better behaved than scipy's
optimisers on ill-conditioned problems.

**When to use:** large models where exact Hessians help convergence, or when
you want to customise the objective with additional Pyomo expressions.
Building the full symbolic ER expression scales superlinearly with rule
count and sample count, so it is most practical with
`optimize_referential_values=False` (the default inside `_fit_pyomo`).

```python
model.fit(X, y, method="ipopt", n_restarts=3)
```

For custom Pyomo objectives, build the model yourself:

```python
from desdeo_brb.pyomo_backend import build_pyomo_brb_model
import pyomo.environ as pyo

m = build_pyomo_brb_model(model, X, y, optimize_referential_values=False)

# Replace or augment the objective
m.del_component(m.obj)
m.obj = pyo.Objective(expr=my_custom_expression(m), sense=pyo.minimize)

pyo.SolverFactory("ipopt").solve(m)
model.update_from_pyomo(m)
```

## Choosing a backend

| Scenario | Recommended backend |
|---|---|
| Quick prototyping, small models | NumPy, `method="SLSQP"` with `n_restarts=10` |
| Many local minima | NumPy, `method="DE+SLSQP"` |
| Large models (>50 rules) | JAX |
| Custom symbolic objective | Pyomo with `build_pyomo_brb_model` + `update_from_pyomo` |
