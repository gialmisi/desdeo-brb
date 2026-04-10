"""Tests for desdeo_brb.jax_backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.jax_backend import JAX_AVAILABLE

from desdeo_brb.inference import (
    compute_activation_weights,
    compute_combined_belief_degrees,
    compute_output,
    input_transform,
)
from desdeo_brb.utils import build_rule_antecedent_indices, pad_referential_values

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from desdeo_brb.jax_backend import (
        compute_activation_weights_jax,
        compute_combined_belief_degrees_jax,
        full_inference_jax,
        input_transform_jax,
    )


def _make_test_data():
    """Create consistent test data for NumPy/JAX comparison."""
    rng = np.random.default_rng(123)
    ref_values = [np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.5, 1.0])]
    X = rng.uniform(-1, 4, size=(7, 2))  # includes out-of-range values
    X[:, 1] = np.clip(X[:, 1], -0.5, 1.5)  # some out-of-range for attr 1 too

    rule_indices = build_rule_antecedent_indices(ref_values)
    n_rules = len(rule_indices)
    n_consequents = 4
    thetas = np.full(n_rules, 1.0 / n_rules)
    deltas = np.ones((n_rules, 2))

    # Random valid belief degrees (rows sum to 1)
    bd_raw = rng.random((n_rules, n_consequents))
    belief_degrees = bd_raw / bd_raw.sum(axis=1, keepdims=True)

    consequent_rv = np.array([0.0, 1.0, 2.0, 3.0])

    return {
        "ref_values": ref_values,
        "X": X,
        "rule_indices": rule_indices,
        "thetas": thetas,
        "deltas": deltas,
        "belief_degrees": belief_degrees,
        "consequent_rv": consequent_rv,
        "n_rules": n_rules,
        "n_consequents": n_consequents,
    }


def test_input_transform_jax_matches_numpy():
    """JAX and NumPy input_transform produce identical results."""
    d = _make_test_data()
    padded_rv, rv_lengths = pad_referential_values(d["ref_values"])

    alphas_np = input_transform(d["X"], d["ref_values"])
    alphas_jax = np.asarray(
        input_transform_jax(
            jnp.asarray(d["X"]),
            jnp.asarray(padded_rv),
            tuple(int(x) for x in rv_lengths),
        )
    )

    for i, rv_len in enumerate(rv_lengths):
        assert_allclose(
            alphas_jax[:, i, :rv_len],
            alphas_np[i],
            atol=1e-6,
            err_msg=f"Mismatch for attribute {i}",
        )


def test_activation_weights_jax_matches_numpy():
    """JAX and NumPy activation weights produce identical results."""
    d = _make_test_data()
    padded_rv, rv_lengths = pad_referential_values(d["ref_values"])

    alphas_np = input_transform(d["X"], d["ref_values"])
    w_np = compute_activation_weights(
        alphas_np, d["rule_indices"], d["thetas"], d["deltas"]
    )

    alphas_jax = input_transform_jax(
        jnp.asarray(d["X"]),
        jnp.asarray(padded_rv),
        tuple(int(x) for x in rv_lengths),
    )
    w_jax = np.asarray(
        compute_activation_weights_jax(
            alphas_jax,
            jnp.asarray(d["rule_indices"]),
            jnp.asarray(d["thetas"]),
            jnp.asarray(d["deltas"]),
        )
    )

    assert_allclose(w_jax, w_np, atol=1e-6)


def test_combined_belief_degrees_jax_matches_numpy():
    """JAX and NumPy combined belief degrees produce identical results."""
    d = _make_test_data()
    padded_rv, rv_lengths = pad_referential_values(d["ref_values"])

    alphas_np = input_transform(d["X"], d["ref_values"])
    w_np = compute_activation_weights(
        alphas_np, d["rule_indices"], d["thetas"], d["deltas"]
    )
    beta_np = compute_combined_belief_degrees(d["belief_degrees"], w_np)

    beta_jax = np.asarray(
        compute_combined_belief_degrees_jax(
            jnp.asarray(d["belief_degrees"]),
            jnp.asarray(w_np),
        )
    )

    assert_allclose(beta_jax, beta_np, atol=1e-6)


def test_full_inference_jax_matches_numpy():
    """End-to-end JAX prediction matches NumPy prediction."""
    d = _make_test_data()
    rv_lengths_tuple = tuple(len(rv) for rv in d["ref_values"])

    flat_params = np.concatenate(
        [
            d["belief_degrees"].ravel(),
            d["thetas"],
            d["deltas"].ravel(),
            *d["ref_values"],
        ]
    )

    output_jax = np.asarray(
        full_inference_jax(
            jnp.asarray(flat_params),
            jnp.asarray(d["X"]),
            jnp.asarray(d["consequent_rv"]),
            jnp.asarray(d["rule_indices"]),
            d["n_rules"],
            d["n_consequents"],
            2,
            rv_lengths_tuple,
        )
    )

    # NumPy path
    alphas = input_transform(d["X"], d["ref_values"])
    weights = compute_activation_weights(
        alphas, d["rule_indices"], d["thetas"], d["deltas"]
    )
    combined = compute_combined_belief_degrees(d["belief_degrees"], weights)
    output_np = compute_output(combined, d["consequent_rv"])

    assert_allclose(output_jax, output_np, atol=1e-5)


def test_full_inference_jax_is_jittable():
    """jax.jit(full_inference_jax) compiles and runs without error."""
    d = _make_test_data()
    rv_lengths_tuple = tuple(len(rv) for rv in d["ref_values"])

    flat_params = jnp.concatenate(
        [
            jnp.array(d["belief_degrees"].ravel()),
            jnp.array(d["thetas"]),
            jnp.array(d["deltas"].ravel()),
            *[jnp.array(rv) for rv in d["ref_values"]],
        ]
    )

    jitted = jax.jit(
        full_inference_jax,
        static_argnames=("n_rules", "n_consequents", "n_attributes", "rv_lengths"),
    )

    output = jitted(
        flat_params,
        jnp.asarray(d["X"]),
        jnp.asarray(d["consequent_rv"]),
        jnp.asarray(d["rule_indices"]),
        d["n_rules"],
        d["n_consequents"],
        2,
        rv_lengths_tuple,
    )
    assert output.shape == (7,)
    assert jnp.all(jnp.isfinite(output))


def test_full_inference_jax_is_differentiable():
    """jax.grad produces finite gradients through the full inference."""
    d = _make_test_data()
    rv_lengths_tuple = tuple(len(rv) for rv in d["ref_values"])

    flat_params = jnp.concatenate(
        [
            jnp.array(d["belief_degrees"].ravel()),
            jnp.array(d["thetas"]),
            jnp.array(d["deltas"].ravel()),
            *[jnp.array(rv) for rv in d["ref_values"]],
        ]
    )

    X_jax = jnp.asarray(d["X"])
    crv_jax = jnp.asarray(d["consequent_rv"])
    rai_jax = jnp.asarray(d["rule_indices"])

    def loss(params):
        y = full_inference_jax(
            params,
            X_jax,
            crv_jax,
            rai_jax,
            d["n_rules"],
            d["n_consequents"],
            2,
            rv_lengths_tuple,
        )
        return jnp.sum(y)

    grad = jax.grad(loss)(flat_params)
    assert grad.shape == flat_params.shape
    assert jnp.all(jnp.isfinite(grad))
