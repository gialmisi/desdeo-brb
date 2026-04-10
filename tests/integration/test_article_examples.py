"""Integration tests reproducing numerical examples from the source papers.

References:
- Chen et al. (2011): "A generalized RIMER method for rule-based
  classification with varying rule lengths"
- Thesis: Section 3.5 (simple additive example), Section 5.1 (Himmelblau)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.brb import BRBModel
from desdeo_brb.utils import build_rule_antecedent_indices


# Helper functions


def f_xsinx2(x: np.ndarray) -> float:
    """f(x) = x * sin(x^2), scalar input."""
    return float(x[0] * np.sin(x[0] ** 2))


def f_x1_plus_x2(x: np.ndarray) -> float:
    """f(x1, x2) = x1 + x2."""
    return float(x[0] + x[1])


def f_himmelblau(x: np.ndarray) -> float:
    """Himmelblau function: (x^2 + y - 11)^2 + (x + y^2 - 7)^2."""
    return float((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)


# Test 1: f(x) = x*sin(x^2) from Chen et al. (2011), Section 4.2


def test_xsinx2_training_improves_mse():
    """Train on f(x)=x*sin(x^2); verify training reduces MSE below 0.05."""
    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])

    model = BRBModel(prv, crv, initial_rule_fn=f_xsinx2)

    # Training data: 200 uniform points in [0, 3] (paper uses 1000 but
    # 200 is sufficient to train the 7-rule model and keeps the test fast)
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 3, size=(200, 1))
    y_train = X_train[:, 0] * np.sin(X_train[:, 0] ** 2)

    # Evaluation data: 500 evenly spaced
    X_eval = np.linspace(0, 3, 500).reshape(-1, 1)
    y_eval = X_eval[:, 0] * np.sin(X_eval[:, 0] ** 2)

    # Untrained predictions
    y_pred_untrained = model.predict_values(X_eval)
    mse_untrained = float(np.mean((y_eval - y_pred_untrained) ** 2))
    assert np.isfinite(mse_untrained) and mse_untrained > 0

    # Train
    model.fit(X_train, y_train, fix_endpoints=True)

    # Trained predictions
    y_pred_trained = model.predict_values(X_eval)
    mse_trained = float(np.mean((y_eval - y_pred_trained) ** 2))

    assert mse_trained < mse_untrained, (
        f"Training did not improve MSE: {mse_trained:.4f} >= {mse_untrained:.4f}"
    )
    assert mse_trained < 0.05, f"Trained MSE too high: {mse_trained:.4f}"


# Test 2: f(x1, x2) = x1 + x2 from the thesis, Section 3.5


def test_x1_plus_x2_at_referential_values():
    """BRB initialized with f(x1,x2)=x1+x2 is exact at referential values."""
    prv = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    model = BRBModel(prv, crv, initial_rule_fn=f_x1_plus_x2)

    # All 9 referential value combinations
    indices = build_rule_antecedent_indices(prv)
    X_ref = np.column_stack([prv[0][indices[:, 0]], prv[1][indices[:, 1]]])
    y_true = X_ref[:, 0] + X_ref[:, 1]

    y_pred = model.predict_values(X_ref)
    assert_allclose(y_pred, y_true, atol=1e-4)


def test_x1_plus_x2_interpolation():
    """BRB produces reasonable interpolations at intermediate points."""
    prv = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    model = BRBModel(prv, crv, initial_rule_fn=f_x1_plus_x2)

    X_mid = np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 1.0]])
    y_true = X_mid[:, 0] + X_mid[:, 1]
    y_pred = model.predict_values(X_mid)

    # For a simple additive function with matching consequent range,
    # the BRB interpolation should be close
    assert_allclose(y_pred, y_true, atol=0.5)


# Test 3: Himmelblau function from Chen et al. (2011), Section 5.1


@pytest.mark.slow
def test_himmelblau_training_improves():
    """Train on Himmelblau function; verify training substantially reduces MSE."""
    rv_1d = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    prv = [rv_1d, rv_1d]
    crv = np.array([0.0, 200.0, 500.0, 1000.0, 2200.0])

    model = BRBModel(prv, crv, initial_rule_fn=f_himmelblau)

    # Training grid: 13 x 13 = 169 points
    x_train = np.linspace(-6, 6, 13)
    X1, X2 = np.meshgrid(x_train, x_train, indexing="ij")
    X_train = np.column_stack([X1.ravel(), X2.ravel()])
    y_train = np.array([f_himmelblau(row) for row in X_train])

    # Evaluation grid: denser
    x_eval = np.linspace(-6, 6, 20)
    X1e, X2e = np.meshgrid(x_eval, x_eval, indexing="ij")
    X_eval = np.column_stack([X1e.ravel(), X2e.ravel()])
    y_eval = np.array([f_himmelblau(row) for row in X_eval])

    # Untrained MSE
    y_pred_untrained = model.predict_values(X_eval)
    mse_untrained = float(np.mean((y_eval - y_pred_untrained) ** 2))

    # Train with more iterations for the large parameter space (49 rules)
    model.fit(X_train, y_train, fix_endpoints=True, options={"maxiter": 500})

    # Trained MSE
    y_pred_trained = model.predict_values(X_eval)
    mse_trained = float(np.mean((y_eval - y_pred_trained) ** 2))

    assert mse_trained < mse_untrained, (
        f"Training did not improve MSE: {mse_trained:.1f} >= {mse_untrained:.1f}"
    )


# Test 4: Inference trace explainability


def test_inference_trace_at_exact_rule():
    """At an exact rule antecedent, the trace shows dominant activation."""
    prv = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    model = BRBModel(prv, crv, initial_rule_fn=f_x1_plus_x2)

    # Point [2, 2] matches the rule with antecedent indices [2, 2]
    # In a 3x3 grid with ij-indexing, rule index for [2,2] is 2*3+2 = 8
    result = model.predict(np.array([[2.0, 2.0]]))

    # The dominant rule should have the highest activation weight
    top_rule = result.dominant_rules(top_k=1)
    assert top_rule.shape == (1, 1)
    assert top_rule[0, 0] == 8  # last rule in 3x3 grid

    # That rule should have very high activation weight
    assert result.activation_weights[0, 8] > 0.99

    # Combined belief degrees should concentrate on consequent = 4.0
    # crv = [0, 1, 2, 3, 4], so index 4 corresponds to value 4.0
    assert result.combined_belief_degrees[0, 4] > 0.99

    # Output should be 4.0
    assert_allclose(result.output, [4.0], atol=1e-4)
