# Explainability

Every intermediate quantity in a BRB inference has semantic meaning: which
rules fired, how strongly, and how their belief distributions were combined.
`desdeo-brb` exposes these with first-class helpers.

For an end-to-end walkthrough with plots, see
`notebooks/04_explainability.ipynb` in the repository.

## Describe a single rule

```python
print(model.rule_base.describe_rule(3, attribute_names=["x"], consequent_name="f(x)"))
# Rule 3: IF x is 1.5 THEN f(x) = {1: 0.833, 2: 0.167} [w=0.143]
```

Options:

- `attribute_names` — list of human-readable names for the attributes.
  Defaults to `x1, x2, ...`.
- `consequent_name` — name for the consequent variable. Defaults to no name.
- `show_zero_beliefs` — if `False` (default), consequent values with
  belief degree < 0.001 are hidden for readability.

## Describe all rules

```python
print(model.rule_base.describe_all_rules(
    attribute_names=["FlowDiff", "PressureDiff"],
    consequent_name="LeakSize",
))
```

## Explain a single prediction

`model.explain(X)` returns a structured explanation:

```python
print(model.explain(
    np.array([[1.5]]),
    top_k=3,
    attribute_names=["x"],
    consequent_name="f(x)",
))
```

Output:

```
Prediction: 1.167

Top activated rules:
  Rule 3 (w=1.0000, x=1.5): {1: 0.833, 2: 0.167}

Combined belief distribution:
  {1: 0.833, 2: 0.167}
```

For multiple samples, use `sample_idx` to select which one to explain:

```python
X = np.array([[1.5], [2.5]])
print(model.explain(X, sample_idx=1))
```

## Accessing raw inference data

`predict()` returns an `InferenceResult` with the full trace as NumPy arrays:

```python
result = model.predict(X)

result.output                    # (n_samples,) scalar predictions
result.activation_weights        # (n_samples, n_rules)
result.combined_belief_degrees   # (n_samples, n_consequents)
result.input_belief_distributions  # list of (n_samples, n_rv_i) arrays
```

Use `result.dominant_rules(top_k=3)` for the indices of the most-activated
rules per sample, and `result.explain(sample_idx, rule_base=model.rule_base)`
to format the explanation for a given sample.

## Why it matters

In decision-support systems, explainability is not a luxury. The
INFRINGER method [@Misitano2020] uses BRB systems to learn a decision
maker's preferences during interactive multi-objective optimisation;
each preference update must be explainable to the decision maker for
them to trust the system.
