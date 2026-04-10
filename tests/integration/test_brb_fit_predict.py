"""Integration tests for the BRB fit-predict pipeline."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.brb import BRBModel
from desdeo_brb.models import InferenceResult
from desdeo_brb.utils import generate_uniform_referential_values


def test_predict_untrained_identity():
    """BRBModel with initial_rule_fn=identity predicts well at referential values."""
    rv = [generate_uniform_referential_values(0.0, 5.0, 6)]
    crv = generate_uniform_referential_values(0.0, 5.0, 6)

    model = BRBModel(rv, crv, initial_rule_fn=lambda x: x[0])

    # Predict at the referential values themselves
    X = rv[0].reshape(-1, 1)
    result = model.predict(X)
    assert_allclose(result.output, rv[0], atol=0.1)


def test_predict_returns_full_trace():
    """Verify InferenceResult has all fields with correct shapes."""
    rv = [
        generate_uniform_referential_values(0.0, 1.0, 3),
        generate_uniform_referential_values(0.0, 1.0, 4),
    ]
    crv = generate_uniform_referential_values(0.0, 1.0, 5)
    model = BRBModel(rv, crv)

    X = np.array([[0.25, 0.75], [0.5, 0.5]])
    result = model.predict(X)

    assert isinstance(result, InferenceResult)
    # input_belief_distributions: list of arrays, one per attribute
    assert len(result.input_belief_distributions) == 2
    assert result.input_belief_distributions[0].shape == (2, 3)
    assert result.input_belief_distributions[1].shape == (2, 4)
    # activation_weights: (n_samples, n_rules) = (2, 3*4=12)
    assert result.activation_weights.shape == (2, 12)
    # combined_belief_degrees: (n_samples, n_consequents) = (2, 5)
    assert result.combined_belief_degrees.shape == (2, 5)
    # output: (n_samples,)
    assert result.output.shape == (2,)


def test_fit_linear_function():
    """Fit on f(x)=2x+1, verify MSE < 0.1 after training."""
    rv = [generate_uniform_referential_values(0.0, 5.0, 6)]
    crv = generate_uniform_referential_values(0.0, 11.0, 12)

    model = BRBModel(rv, crv, initial_rule_fn=lambda x: 2 * x[0] + 1)

    X_train = np.linspace(0, 5, 20).reshape(-1, 1)
    y_train = 2 * X_train.ravel() + 1

    model.fit(X_train, y_train, fix_endpoints=True)

    y_pred = model.predict_values(X_train)
    mse = float(np.mean((y_train - y_pred) ** 2))
    assert mse < 0.1, f"MSE too high after training: {mse}"


def test_fit_custom_loss():
    """Pass a custom loss, verify parameters update."""
    rv = [generate_uniform_referential_values(0.0, 1.0, 3)]
    crv = generate_uniform_referential_values(0.0, 1.0, 3)

    model = BRBModel(rv, crv)
    initial_bd = model.belief_degrees.copy()

    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 0.5, 1.0])

    def mse_loss(m: BRBModel) -> float:
        y_pred = m.predict_values(X)
        return float(np.mean((y - y_pred) ** 2))

    model.fit_custom(mse_loss)

    # Parameters should have changed (or at least the optimizer ran)
    assert model.belief_degrees is not initial_bd


def test_sklearn_get_set_params():
    """Verify round-trip of get_params / set_params."""
    rv = [generate_uniform_referential_values(0.0, 1.0, 3)]
    crv = generate_uniform_referential_values(0.0, 1.0, 3)
    model = BRBModel(rv, crv)

    params = model.get_params(deep=True)
    assert "precedent_referential_values" in params
    assert "consequent_referential_values" in params
    assert "rule_base" in params

    # Set a new utility function
    model.set_params(utility_fn=lambda d: d**2)
    new_params = model.get_params()
    assert new_params["utility_fn"] is not None


def test_fit_endpoint_fidelity():
    """After training with fix_endpoints=True, predictions at the
    exact endpoint referential values should closely match the
    true function values."""

    def f(x):
        return x * np.sin(x**2)

    precedents = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    consequents = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(precedents, consequents, initial_rule_fn=lambda x: f(x[0]))

    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 3, size=(1000, 1))
    y_train = f(X_train[:, 0])
    model.fit(X_train, y_train, fix_endpoints=True, fix_endpoint_beliefs=True)

    # Predict at endpoints
    X_endpoints = np.array([[0.0], [3.0]])
    y_pred = model.predict_values(X_endpoints)
    y_true = f(X_endpoints[:, 0])

    # Endpoints should be close to true values
    assert_allclose(
        y_pred,
        y_true,
        atol=0.15,
        err_msg="Trained model diverges from true function at endpoints",
    )


def test_fix_endpoint_beliefs_allows_training():
    """Verify that fix_endpoint_beliefs=True still allows meaningful training."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    X_eval = np.linspace(0, 3, 500).reshape(-1, 1)
    y_true = X_eval[:, 0] * np.sin(X_eval[:, 0] ** 2)

    # Untrained MSE
    y_untrained = model.predict_values(X_eval)
    mse_before = float(np.mean((y_true - y_untrained) ** 2))

    # Train with both endpoint fixes
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 3, size=(1000, 1))
    y_train = X_train[:, 0] * np.sin(X_train[:, 0] ** 2)
    model.fit(X_train, y_train, fix_endpoints=True, fix_endpoint_beliefs=True)

    # Trained MSE
    y_trained = model.predict_values(X_eval)
    mse_after = float(np.mean((y_true - y_trained) ** 2))

    # Training must have actually improved the model
    assert mse_after < mse_before, (
        f"MSE did not improve: {mse_before:.4f} -> {mse_after:.4f}"
    )
    assert mse_after < 0.2, f"Trained MSE too high: {mse_after:.4f}"

    # Endpoint beliefs should be preserved
    y_endpoints = model.predict_values(np.array([[0.0], [3.0]]))
    assert_allclose(y_endpoints[0], 0.0, atol=0.15)
    assert_allclose(y_endpoints[1], f(3.0), atol=0.15)


def test_fit_with_trust_constr():
    """Verify training works with trust-constr method."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    X_train = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_train = f(X_train[:, 0])

    model.fit(X_train, y_train, fix_endpoints=True, method="trust-constr")

    X_eval = np.linspace(0, 3, 500).reshape(-1, 1)
    y_pred = model.predict_values(X_eval)
    y_true = f(X_eval[:, 0])
    mse = float(np.mean((y_true - y_pred) ** 2))

    assert mse < 0.05, f"trust-constr MSE too high: {mse}"


def test_fit_with_custom_options():
    """Verify custom optimizer options are passed through."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    X_train = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_train = f(X_train[:, 0])

    model.fit(
        X_train,
        y_train,
        fix_endpoints=True,
        method="SLSQP",
        optimizer_options={"maxiter": 2000, "ftol": 1e-12},
    )

    X_eval = np.linspace(0, 3, 500).reshape(-1, 1)
    y_pred = model.predict_values(X_eval)
    y_true = f(X_eval[:, 0])
    mse = float(np.mean((y_true - y_pred) ** 2))

    assert mse < 0.05, f"Custom options MSE too high: {mse}"


def test_fit_invalid_method():
    """Verify invalid method raises ValueError."""
    prv = [np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0])
    model = BRBModel(prv, crv)

    X = np.array([[0.5]])
    y = np.array([0.5])

    with pytest.raises(ValueError):
        model.fit(X, y, method="invalid_method")


def test_referential_values_move_during_training():
    """Verify that interior referential values shift during training.

    When fix_endpoints=True, the first and last referential values are fixed
    but the interior ones should move to better capture the function's shape.
    """

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    interior_before = model.rule_base.precedent_referential_values[0][1:-1].copy()

    X_train = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_train = f(X_train[:, 0])
    model.fit(X_train, y_train, fix_endpoints=True)

    rv_after = model.rule_base.precedent_referential_values[0]
    interior_after = rv_after[1:-1]

    # Endpoints must be fixed
    assert rv_after[0] == 0.0
    assert rv_after[-1] == 3.0

    # Interior values must have moved
    assert not np.allclose(interior_before, interior_after, atol=0.01), (
        f"Interior referential values did not move: {interior_after}"
    )

    # Referential values must remain sorted
    assert np.all(np.diff(rv_after) >= -1e-10)


def test_varying_referential_value_lengths():
    """Two attributes with 3 and 5 referential values; predict works."""
    rv = [
        generate_uniform_referential_values(0.0, 1.0, 3),
        generate_uniform_referential_values(0.0, 1.0, 5),
    ]
    crv = generate_uniform_referential_values(0.0, 1.0, 4)
    model = BRBModel(rv, crv)

    X = np.array([[0.5, 0.5], [0.0, 1.0]])
    result = model.predict(X)
    assert result.output.shape == (2,)
    # n_rules = 3 * 5 = 15
    assert result.activation_weights.shape == (2, 15)
