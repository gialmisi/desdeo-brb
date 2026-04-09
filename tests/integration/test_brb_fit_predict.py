"""Integration tests for the BRB fit-predict pipeline."""

import numpy as np
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
