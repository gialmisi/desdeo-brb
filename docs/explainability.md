# Explainability

Every intermediate quantity in a BRB inference has semantic meaning: which
rules fired, how strongly, how their belief distributions were combined, and
how much belief the combination left unassigned. `desdeo-brb` exposes these
with first-class helpers.

For an end-to-end walkthrough with plots, see
`notebooks/04_explainability.ipynb` in the repository.

## Describe a single rule

```python
print(model.rule_base.describe_rule(3, attribute_names=["x"], consequent_name="f(x)"))
# Rule 3: IF x is 1.5 THEN f(x) = {1: 0.833, 2: 0.167} [w=0.143]
```

Options:

- `attribute_names`: list of human-readable names for the attributes.
  Defaults to `x1, x2, ...`.
- `consequent_name`: name for the consequent: one name for a single output,
  or a sequence giving one name per output. Defaults to no name for a single
  output and `y1, y2, ...` for several.
- `show_zero_beliefs`: if `False` (default), values with belief degree
  < 0.001 are hidden for readability.

A rule whose belief degrees sum to less than one is incomplete: the missing
mass is its ignorance, and it simply does not appear in the printed
distribution. Read it explicitly with `rule_base.ignorance`, which gives one
number per rule (or one per rule per output), and `rule_base.is_complete` for
the rule base as a whole.

With [several outputs](training.md#several-outputs) each consequent gets its
own distribution, because the grades of one objective say nothing about
another's:

```python
print(rule_base.describe_rule(4, attribute_names=["Temp", "Pressure"],
                              consequent_name=["Yield", "Cost"]))
# Rule 4: IF Temp is 0.5 AND Pressure is 0.5 THEN Yield = {0: 0.100, 0.5: 0.900},
#         Cost = {100: 0.075, 300: 0.925} [w=0.111]
```

Passing a single name when there are several outputs raises, rather than
guessing which objective it belongs to.

With [extended antecedents](training.md#extended-antecedents) a rule sits
between referential values rather than on one, so the IF clause shows its
distribution over them:

```python
print(rule_base.describe_rule(7, attribute_names=["x"]))
# Rule 7: IF x is {1: 0.300, 2: 0.700} THEN {1: 1.000} [w=0.010]
```

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
print(model.explain(np.array([[1.5]]), top_k=3, attribute_names=["x"]))
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

`consequent_name` labels the prediction and the combined distribution, taking
the same forms as it does for `describe_rule`:

```python
print(model.explain(np.array([[1.5]]), attribute_names=["x"], consequent_name="f(x)"))
```

```
Prediction: f(x)=1.167

Top activated rules:
  Rule 3 (w=1.0000, x=1.5): {1: 0.833, 2: 0.167}

Combined belief distribution:
  f(x): {1: 0.833, 2: 0.167}
```

With several outputs, the prediction line carries one value per output and the
belief distributions are printed one block per output, because the grades of
one objective say nothing about another's:

```
Prediction: y1=0.3917, y2=258.5

Top activated rules:
  Rule 3 (w=0.2500, x1=1, x2=1): {0: 0.399, 0.5: 0.563, 1: 0.038} {100: 0.086, 250: 0.465, 400: 0.449}
  Rule 2 (w=0.2500, x1=1, x2=0): {0: 0.529, 0.5: 0.021, 1: 0.450} {100: 0.111, 250: 0.546, 400: 0.343}

Combined belief distribution:
  y1: {0: 0.478, 0.5: 0.261, 1: 0.261}
  y2: {100: 0.259, 250: 0.426, 400: 0.315}
```

## Accessing raw inference data

`predict()` returns an `InferenceResult` with the full trace as NumPy arrays:

```python
result = model.predict(X)

result.output                    # (n_samples,), or (n_samples, n_outputs)
result.activation_weights        # (n_samples, n_rules)
result.combined_belief_degrees   # (n_samples, n_consequents)
result.input_belief_distributions  # list of (n_samples, n_rv_i) arrays
```

Use `result.dominant_rules(top_k=3)` for the indices of the most-activated
rules per sample, and `result.explain(sample_idx, rule_base=model.rule_base)`
to format the explanation for a given sample.

With several outputs, `combined_belief_degrees` holds every output's grades
concatenated along its second axis. `result.n_outputs` says how many there are
and `result.consequent_slices` gives the column range of each, so one output's
distribution is a slice away:

```python
for o, block in enumerate(result.consequent_slices):
    grades = result.consequent_values[block]
    beliefs = result.combined_belief_degrees[:, block]
    print(f"output {o}: grades {grades}, beliefs {beliefs[0]}")
```

The same layout applies to the rule base: `rule_base.consequent_values(o)` and
`rule_base.beliefs_for(o)` return one output's grades and one output's belief
degrees.

## Ignorance and utility bounds

When the rules that fired were themselves incomplete, the combined assessment
is incomplete too, and the prediction is no longer a single defensible number.
Three fields make that explicit:

```python
result.ignorance       # belief the combination left unassigned, shaped like output
result.is_complete     # True when every assessment assigns all of its belief
result.utility_bounds  # (lower, upper), each shaped like output
```

`utility_bounds` is the utility interval of Yang and Xu (2002, Section II-H).
The unassigned belief could belong to any grade, so it bounds the output rather
than fixing it: the lower bound gives all of it to the least preferred grade,
the upper bound to the most preferred one. `result.output` is the midpoint, the
*average expected utility*. The width of the interval is how much the model is
declining to say.

```python
lower, upper = result.utility_bounds
print(f"{result.output[0]:.3f} in [{lower[0]:.3f}, {upper[0]:.3f}]"
      f" (ignorance {result.ignorance[0]:.3f})")
```

For a complete assessment the bounds coincide with the output and `ignorance`
is zero, so this reduces to the ordinary weighted average and existing code
reads the same numbers as before.

`to_dict()` emits `output`, `ignorance` and, when present, `utility_bounds`
alongside the rest of the trace, so an explanation can be serialised and shown
elsewhere.

See [Incomplete rules](training.md#incomplete-rules) for how training interacts
with all this, and `notebooks/05_incomplete_and_multi_output.ipynb` for a worked
example that plots the interval widening exactly where a rule base declines to
commit.

## Extended rule bases

A rule base with [extended antecedents](training.md#extended-antecedents)
matches inputs by distance between belief distributions rather than by gathering
one referential value per attribute. The whole trace reads as usual:
`activation_weights`, `combined_belief_degrees`, `output`, `ignorance` and
`utility_bounds` all mean what they mean anywhere else; and the description
helpers work too, showing each rule's antecedent distribution in place of a
single value:

```python
print(model.explain(np.array([[0.5]]), attribute_names=["x"], consequent_name="y"))
```

```
Prediction: y=0.8672

Top activated rules:
  Rule 1 (w=0.5000, x={0: 0.300, 1: 0.700}): {1: 1.000}
  Rule 0 (w=0.3125, x={0: 1.000}): {0: 1.000}
  Rule 2 (w=0.1875, x={1: 0.200, 2: 0.800}): {2: 1.000}

Combined belief distribution:
  y: {0: 0.270, 1: 0.593, 2: 0.137}
```

Because matching is by distance, a rule is normally activated to some degree by
every input, rather than only by inputs near its own cell. Expect more rules
with non-negligible weight than in a conventional rule base, and use `top_k` to
keep an explanation readable.

An extended rule base also normally has one rule per training sample, so a rule
index points back at the data point the rule was read off. `result.explain()`
called without a `rule_base` identifies rules by that index alone.

## Why it matters

In decision-support systems, explainability is not a luxury. The
INFRINGER method [@Misitano2020] uses BRB systems to learn a decision
maker's preferences during interactive multi-objective optimisation;
each preference update must be explainable to the decision maker for
them to trust the system.
