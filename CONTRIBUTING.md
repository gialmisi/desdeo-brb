# Contributing to desdeo-brb

Thank you for your interest in contributing! This document outlines how to
contribute code, report bugs, and request features.

## Reporting bugs

Please open an issue at https://github.com/gialmisi/desdeo-brb/issues with:

- A minimal code example that reproduces the bug
- The expected behavior and actual behavior
- Your Python version and `desdeo-brb` version (`pip show desdeo-brb`)
- Full error traceback if applicable

## Requesting features

Feature requests are welcome as issues. Please describe:

- The use case motivating the feature
- A rough API sketch if you have one in mind
- Any references to relevant papers or existing implementations

## Contributing code

1. **Fork** the repository and create a feature branch from `main`.
2. **Install** the development dependencies:
   ```bash
   uv sync --all-extras
   ```
3. **Make your changes** with tests. All new features require tests.
4. **Run the test suite**:
   ```bash
   uv run pytest
   ```
5. **Check code style**:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   ```
6. **Submit a pull request** with a clear description of the change.

## Code style

- Follow PEP 8 (enforced by `ruff`)
- Use type hints for all public functions
- Write docstrings in NumPy style for all public API
- Keep lines under 100 characters

## Testing

- All public functions must have tests
- Use `pytest` fixtures from `tests/conftest.py` where appropriate
- Mark slow tests with `@pytest.mark.slow`
- Aim for meaningful coverage, not just line coverage

## Scientific contributions

If your contribution implements a method from a published paper, please:

- Cite the paper in the relevant docstring
- Add a test that reproduces a result from the paper where feasible
- Update `CHANGELOG.md` with the reference

## License

By contributing, you agree that your contributions will be licensed under the
project's MIT License.
