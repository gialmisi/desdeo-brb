# Notebooks

Worked examples demonstrating `desdeo-brb` features.

| Notebook | Focus |
|---|---|
| `01_getting_started.ipynb` | First BRB model: f(x) = x sin(x^2) |
| `02_multi_attribute.ipynb` | Multi-attribute models: additive, Himmelblau |
| `03_expert_knowledge.ipynb` | Expert rules + training: pipeline leak detection |
| `04_explainability.ipynb` | Interpreting models and predictions |
| `05_incomplete_and_multi_output.ipynb` | Ignorance, utility bounds, and several consequent attributes |

## Running the notebooks

Install with notebook dependencies:

```bash
pip install desdeo-brb[notebooks]
```

Or with all backends:

```bash
pip install desdeo-brb[all]
```

Then launch:

```bash
jupyter notebook notebooks/
```

## Testing the notebooks

The notebooks are executed in CI to catch drift from the API. To run them
locally the same way:

```bash
uv run pytest --nbmake notebooks/
```

They are committed without stored outputs, so run them to see the results.
