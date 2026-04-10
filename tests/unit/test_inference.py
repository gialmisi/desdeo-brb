"""Tests for desdeo_brb.inference."""

import numpy as np
from numpy.testing import assert_allclose

from desdeo_brb.inference import (
    compute_activation_weights,
    compute_combined_belief_degrees,
    compute_output,
    input_transform,
)


# input_transform tests


def test_input_transform_at_referential_value():
    """Input exactly at a referential value gives belief 1.0 there, 0 elsewhere."""
    rv = [np.array([0.0, 0.5, 1.0])]
    X = np.array([[0.5]])  # exactly at the middle referential value
    alphas = input_transform(X, rv)
    assert_allclose(alphas[0][0], [0.0, 1.0, 0.0])


def test_input_transform_between_values():
    """Input midway between two referential values gives 0.5 and 0.5."""
    rv = [np.array([0.0, 1.0])]
    X = np.array([[0.5]])
    alphas = input_transform(X, rv)
    assert_allclose(alphas[0][0], [0.5, 0.5])


def test_input_transform_outside_range():
    """Input outside range is clamped to the nearest boundary (RIMER spec)."""
    rv = [np.array([0.0, 0.5, 1.0])]
    X_below = np.array([[-0.1]])
    X_above = np.array([[1.1]])
    alphas_below = input_transform(X_below, rv)
    alphas_above = input_transform(X_above, rv)
    # Below range -> belief 1.0 at first referential value
    assert_allclose(alphas_below[0][0], [1.0, 0.0, 0.0])
    # Above range -> belief 1.0 at last referential value
    assert_allclose(alphas_above[0][0], [0.0, 0.0, 1.0])


def test_input_transform_boundary_clamping():
    """Verify inputs outside referential value range are clamped to boundaries."""
    rv = [np.array([0.0, 1.0, 2.0, 3.0])]

    # Well below range
    X = np.array([[-5.0]])
    alphas = input_transform(X, rv)
    assert alphas[0][0, 0] == 1.0
    assert np.sum(alphas[0][0, 1:]) == 0.0

    # Well above range
    X = np.array([[10.0]])
    alphas = input_transform(X, rv)
    assert alphas[0][0, -1] == 1.0
    assert np.sum(alphas[0][0, :-1]) == 0.0

    # Slightly below range
    X = np.array([[-0.001]])
    alphas = input_transform(X, rv)
    assert alphas[0][0, 0] == 1.0

    # Slightly above range
    X = np.array([[3.001]])
    alphas = input_transform(X, rv)
    assert alphas[0][0, -1] == 1.0


def test_input_transform_varying_lengths():
    """Two attributes with different numbers of referential values."""
    rv = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0])]
    X = np.array([[0.5, 5.0]])
    alphas = input_transform(X, rv)
    # Attribute 0: 0.5 is midway between 0.0 and 1.0
    assert alphas[0].shape == (1, 3)
    assert_allclose(alphas[0][0], [0.5, 0.5, 0.0])
    # Attribute 1: 5.0 is midway between 0.0 and 10.0
    assert alphas[1].shape == (1, 2)
    assert_allclose(alphas[1][0], [0.5, 0.5])


# compute_activation_weights tests


def _make_simple_brb():
    """Create a simple 2-attribute, 4-rule BRB for testing."""
    rv = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    # 4 rules: all combinations of 2 referential values x 2 attributes
    rule_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    thetas = np.ones(4)
    deltas = np.ones((4, 2))
    return rv, rule_indices, thetas, deltas


def test_activation_weights_sum_to_one():
    """Activation weights across rules sum to 1 for any input."""
    rv, rule_indices, thetas, deltas = _make_simple_brb()
    X = np.array([[0.3, 0.7], [0.0, 1.0], [0.5, 0.5]])
    alphas = input_transform(X, rv)
    w = compute_activation_weights(alphas, rule_indices, thetas, deltas)
    assert_allclose(w.sum(axis=1), np.ones(3), atol=1e-10)


def test_activation_weights_single_rule_fires():
    """When input exactly matches one rule's antecedents, that rule dominates."""
    rv, rule_indices, thetas, deltas = _make_simple_brb()
    # Input at (1.0, 1.0) matches rule index [1, 1] = rule 3
    X = np.array([[1.0, 1.0]])
    alphas = input_transform(X, rv)
    w = compute_activation_weights(alphas, rule_indices, thetas, deltas)
    assert w[0, 3] > 0.99
    assert_allclose(w.sum(axis=1), [1.0], atol=1e-10)


# compute_combined_belief_degrees tests


def test_combined_belief_degrees_sum_to_one():
    """Combined beliefs sum to 1 when all rules are complete (BRE rows sum to 1)."""
    n_rules, n_consequents = 4, 3
    # Each rule distributes belief fully across consequents
    bre = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
            [0.2, 0.2, 0.6],
        ]
    )
    assert_allclose(bre.sum(axis=1), np.ones(n_rules))  # sanity check

    # Some arbitrary weights that sum to 1
    weights = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.2, 0.2, 0.1]])
    beta = compute_combined_belief_degrees(bre, weights)
    assert beta.shape == (2, n_consequents)
    assert_allclose(beta.sum(axis=1), np.ones(2), atol=1e-10)


def test_combined_belief_single_rule():
    """When only one rule is active, combined beliefs equal that rule's beliefs."""
    bre = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ]
    )
    # Only rule 0 fires
    weights = np.array([[1.0, 0.0]])
    beta = compute_combined_belief_degrees(bre, weights)
    assert_allclose(beta[0], [0.6, 0.3, 0.1], atol=1e-10)


# compute_output tests


def test_compute_output_identity():
    """With identity utility and known beliefs, verify scalar output."""
    consequents = np.array([10.0, 20.0, 30.0])
    belief_degrees = np.array([[0.5, 0.3, 0.2]])  # weighted avg = 5+6+6 = 17
    y = compute_output(belief_degrees, consequents)
    assert_allclose(y, [17.0])


def test_compute_output_custom_utility():
    """Custom utility function is applied to consequent values."""
    consequents = np.array([1.0, 2.0, 3.0])
    belief_degrees = np.array([[1.0, 0.0, 0.0]])
    # Utility squares the values
    y = compute_output(belief_degrees, consequents, utility_fn=lambda d: d**2)
    assert_allclose(y, [1.0])
