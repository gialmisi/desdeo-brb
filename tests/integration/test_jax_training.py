"""Integration tests for JAX-based training."""

import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.jax_backend import JAX_AVAILABLE

from desdeo_brb.brb import BRBModel
from desdeo_brb.utils import generate_uniform_referential_values

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


def f_xsinx2(x: np.ndarray) -> float:
    return float(x[0] * np.sin(x[0] ** 2))


def test_jax_fit_xsinx2():
    """Reproduce f(x)=x*sin(x^2) with JAX backend; trained MSE < 0.05."""
    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])

    model = BRBModel(prv, crv, initial_rule_fn=f_xsinx2, backend="jax")

    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 3, size=(1000, 1))
    y_train = X_train[:, 0] * np.sin(X_train[:, 0] ** 2)

    X_eval = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_eval = X_eval[:, 0] * np.sin(X_eval[:, 0] ** 2)

    # Untrained
    mse_untrained = float(np.mean((y_eval - model.predict_values(X_eval)) ** 2))
    assert mse_untrained > 0

    # Train with JAX (L-BFGS-B needs more iterations than SLSQP for this problem)
    model.fit(X_train, y_train, fix_endpoints=True, options={"maxiter": 500})

    y_pred = model.predict_values(X_eval)
    mse_trained = float(np.mean((y_eval - y_pred) ** 2))

    assert mse_trained < mse_untrained
    assert mse_trained < 0.05, f"Trained MSE too high: {mse_trained:.4f}"


def test_jax_fit_not_catastrophically_slower():
    """JAX training is not catastrophically slower than NumPy on a small problem."""
    prv = [generate_uniform_referential_values(0.0, 5.0, 6)]
    crv = generate_uniform_referential_values(0.0, 11.0, 12)

    X_train = np.linspace(0, 5, 50).reshape(-1, 1)
    y_train = 2 * X_train.ravel() + 1

    # NumPy timing
    model_np = BRBModel(
        prv, crv, initial_rule_fn=lambda x: 2 * x[0] + 1, backend="numpy"
    )
    t0 = time.perf_counter()
    model_np.fit(X_train, y_train, fix_endpoints=True)
    np_time = time.perf_counter() - t0

    # JAX timing (includes JIT compilation)
    model_jax = BRBModel(
        prv, crv, initial_rule_fn=lambda x: 2 * x[0] + 1, backend="jax"
    )
    t0 = time.perf_counter()
    model_jax.fit(X_train, y_train, fix_endpoints=True)
    jax_time = time.perf_counter() - t0

    # JAX may be slower on small problems due to JIT overhead.
    # Just check it's not absurdly slower (< 20x).
    assert jax_time < np_time * 20, (
        f"JAX too slow: {jax_time:.2f}s vs NumPy {np_time:.2f}s"
    )


@pytest.mark.slow
def test_jax_faster_on_large_problem():
    """JAX with exact gradients is faster than NumPy/SLSQP on a large problem.

    The advantage comes from gradient computation cost: SLSQP approximates
    gradients via finite differences, costing O(n_params) forward passes per
    iteration. JAX computes exact gradients in ~1 forward+backward pass.

    With 2 attributes x 8 referential values = 64 rules x 6 consequents,
    there are ~500 trainable parameters. Each SLSQP gradient step needs ~500
    forward evaluations (one per parameter), while JAX needs ~2 (one forward
    + one backward). With 500 training samples this adds up quickly.
    """
    # 2 attributes, 8 referential values each -> 8^2 = 64 rules
    prv = [
        generate_uniform_referential_values(0.0, 1.0, 8),
        generate_uniform_referential_values(0.0, 1.0, 8),
    ]
    crv = generate_uniform_referential_values(0.0, 2.0, 6)

    def f_sum(x: np.ndarray) -> float:
        return float(x[0] + x[1])

    rng = np.random.default_rng(99)
    X_train = rng.uniform(0, 1, size=(500, 2))
    y_train = X_train.sum(axis=1)

    # Both backends get the same max iterations so we compare
    # wall-clock cost per iteration, not convergence quality.
    maxiter = 20

    # NumPy/SLSQP timing
    model_np = BRBModel(prv, crv, initial_rule_fn=f_sum, backend="numpy")
    t0 = time.perf_counter()
    model_np.fit(X_train, y_train, fix_endpoints=True, options={"maxiter": maxiter})
    np_time = time.perf_counter() - t0

    # JAX/L-BFGS-B timing — warm up JIT first so compilation cost is excluded
    model_jax_warmup = BRBModel(prv, crv, initial_rule_fn=f_sum, backend="jax")
    model_jax_warmup.fit(X_train, y_train, fix_endpoints=True, options={"maxiter": 1})

    # Now time the real run (JIT caches are warm)
    model_jax = BRBModel(prv, crv, initial_rule_fn=f_sum, backend="jax")
    t0 = time.perf_counter()
    model_jax.fit(X_train, y_train, fix_endpoints=True, options={"maxiter": maxiter})
    jax_time = time.perf_counter() - t0

    # JAX should be faster with warm JIT on this problem
    speedup = np_time / jax_time if jax_time > 0 else float("inf")
    assert jax_time < np_time, (
        f"JAX ({jax_time:.2f}s) was not faster than NumPy ({np_time:.2f}s) "
        f"on a 64-rule, 500-sample problem with {maxiter} iterations"
    )
    # Print for visibility when run with -s
    print(
        f"\n  NumPy/SLSQP: {np_time:.2f}s | JAX/L-BFGS-B: {jax_time:.2f}s | "
        f"Speedup: {speedup:.1f}x"
    )


def test_jax_predict_matches_numpy_predict():
    """After JAX training, predictions match a NumPy model with same parameters."""
    prv = [generate_uniform_referential_values(0.0, 5.0, 6)]
    crv = generate_uniform_referential_values(0.0, 11.0, 12)

    X_train = np.linspace(0, 5, 50).reshape(-1, 1)
    y_train = 2 * X_train.ravel() + 1

    # Train with JAX
    model_jax = BRBModel(
        prv, crv, initial_rule_fn=lambda x: 2 * x[0] + 1, backend="jax"
    )
    model_jax.fit(X_train, y_train, fix_endpoints=True)

    # Create NumPy model with the same trained rule base
    model_np = BRBModel(prv, crv, rule_base=model_jax.rule_base, backend="numpy")

    X_eval = np.linspace(0, 5, 20).reshape(-1, 1)
    y_jax = model_jax.predict_values(X_eval)
    y_np = model_np.predict_values(X_eval)

    assert_allclose(y_jax, y_np, atol=1e-6)
