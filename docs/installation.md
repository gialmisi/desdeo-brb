# Installation

`desdeo-brb` requires Python 3.10 or later. Install from PyPI:

```bash
pip install desdeo-brb          # Core (NumPy + SciPy)
pip install desdeo-brb[jax]     # + JAX backend
pip install desdeo-brb[pyomo]   # + Pyomo/IPOPT backend
pip install desdeo-brb[all]     # Everything
```

## Optional backends

### JAX backend

The JAX backend provides JIT-compiled inference and exact-gradient training
via automatic differentiation. It is useful on larger problems where
finite-difference gradients become a bottleneck.

```bash
pip install desdeo-brb[jax]
```

GPU builds of JAX can be installed separately; see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for details.

### Pyomo / IPOPT backend

The Pyomo backend builds a symbolic optimization model that can be solved
with any AMPL-compatible solver. By default the library uses IPOPT.

```bash
pip install desdeo-brb[pyomo]
```

The IPOPT binary must be installed separately:

=== "Debian / Ubuntu"

    ```bash
    sudo apt install coinor-libipopt-dev
    ```

=== "Conda"

    ```bash
    conda install -c conda-forge ipopt
    ```

=== "macOS (Homebrew)"

    ```bash
    brew install ipopt
    ```

Once the IPOPT binary is on your `PATH`, you can train with
`model.fit(X, y, method="ipopt")`.

## Development installation

To contribute or run the test suite, install the `dev` extras:

```bash
git clone https://github.com/gialmisi/desdeo-brb
cd desdeo-brb
uv sync --all-extras
uv run pytest
```

See [CONTRIBUTING](https://github.com/gialmisi/desdeo-brb/blob/master/CONTRIBUTING.md)
for the full contribution guide.

## Notebooks

The tutorial notebooks in `notebooks/` require `jupyter` and `matplotlib`:

```bash
pip install desdeo-brb[notebooks]
```

Then launch with:

```bash
jupyter notebook notebooks/
```
