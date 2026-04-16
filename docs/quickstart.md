# Quickstart

This page walks through a minimal BRB workflow on the benchmark function
$f(x) = x \sin(x^2)$ from [@ChenEtAl2011]. For a fuller walkthrough with
plots, see `notebooks/01_getting_started.ipynb` in the repository.

## Define the model

A BRB model is specified by its **referential values**: discrete anchor points
on the input and output spaces.

```python
import numpy as np
from desdeo_brb import BRBModel

# 7 precedent referential values for x in [0, 3]
prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]

# 5 consequent referential values for f(x)
crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])

# Construct the model; initial beliefs computed by evaluating f
# at each precedent referential value
model = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] * np.sin(x[0] ** 2))
```

The Cartesian product of precedent referential values defines 7 rules. Each
rule has a belief distribution over the 5 consequent referential values.

## Inspect the initial rule base

```python
print(model.rule_base.describe_all_rules(
    attribute_names=["x"], consequent_name="f(x)"
))
```

Output:

```
Rule 0: IF x is 0 THEN f(x) = {-1: 0.500, 1: 0.500} [w=0.143]
Rule 1: IF x is 0.5 THEN f(x) = {-1: 0.438, 1: 0.562} [w=0.143]
Rule 2: IF x is 1 THEN f(x) = {-1: 0.079, 1: 0.921} [w=0.143]
...
```

## Train the model

BRB training is non-convex, so multiple random restarts are recommended:

```python
X_train = np.linspace(0, 3, 1000).reshape(-1, 1)
y_train = X_train[:, 0] * np.sin(X_train[:, 0] ** 2)

model.fit(X_train, y_train, fix_endpoints=True, n_restarts=10)
```

See the [training guide](training.md) for details on the available methods.

## Predict and explain

```python
# Predict scalar outputs
y_pred = model.predict_values(np.array([[1.5], [2.5]]))
print(y_pred)  # [1.167..., 0.093...]

# Get the full inference trace
result = model.predict(np.array([[1.5]]))
print(result.activation_weights)    # which rules fired
print(result.combined_belief_degrees)  # output belief distribution

# Human-readable explanation
print(model.explain(np.array([[1.5]]), attribute_names=["x"], consequent_name="f(x)"))
```

Typical `explain()` output:

```
Prediction: 1.167

Top activated rules:
  Rule 3 (w=1.0000, x=1.5): {1: 0.833, 2: 0.167}

Combined belief distribution:
  {1: 0.833, 2: 0.167}
```

## Next steps

- Read the [training guide](training.md) for method-by-method recommendations
- Read the [explainability guide](explainability.md) for inspection tools
- See the [API reference](api/brb.md) for full signatures
