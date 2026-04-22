# Notebooks

Worked examples demonstrating `desdeo-brb` features.

| Notebook | Focus |
|---|---|
| `01_getting_started.ipynb` | First BRB model: f(x) = x sin(x^2) |
| `02_multi_attribute.ipynb` | Multi-attribute models: additive, Himmelblau |
| `03_expert_knowledge.ipynb` | Expert rules + training: pipeline leak detection |
| `04_explainability.ipynb` | Interpreting models and predictions |

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
