# desdeo-brb

**A trainable Belief Rule-Based (BRB) inference system for Python.**

`desdeo-brb` implements the RIMER methodology for Belief Rule-Based systems
[@YangEtAl2006; @ChenEtAl2011] with a scikit-learn-compatible API, multiple
optimization backends, and first-class explainability.

## Features

- Trainable BRB models with MSE or custom loss objectives
- scikit-learn-style `fit()` / `predict()` API
- Multiple training backends: NumPy (SLSQP, trust-constr, DE), JAX (L-BFGS-B with autodiff), and Pyomo (IPOPT with exact Hessians)
- Multi-start optimization for non-convex problems
- Adaptive referential value training
- Human-readable rule descriptions and prediction explanations
- Extensive test suite reproducing paper benchmarks

## Quick example

```python
import numpy as np
from desdeo_brb import BRBModel

# Define a BRB model for f(x) = x * sin(x^2)
prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
model = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] * np.sin(x[0] ** 2))

# Train with multiple restarts for reliable convergence
X_train = np.linspace(0, 3, 1000).reshape(-1, 1)
y_train = X_train[:, 0] * np.sin(X_train[:, 0] ** 2)
model.fit(X_train, y_train, n_restarts=10)

# Predict
y_pred = model.predict_values(np.array([[1.5]]))

# Explain the prediction
print(model.explain(np.array([[1.5]])))
```

## Next steps

- [Install `desdeo-brb`](installation.md)
- [Quickstart guide](quickstart.md)
- [Training deep-dive](training.md)
- [Explainability guide](explainability.md)
- [API reference](api/brb.md)
