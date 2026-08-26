"""Tests for desdeo_brb.inference."""

import numpy as np
from numpy.testing import assert_allclose

from desdeo_brb.inference import (
    compute_activation_weights,
    compute_combined_belief_degrees,
    compute_output,
    compute_utility_bounds,
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


# Cross-check against the recursive evidential reasoning algorithm


def _recursive_er(beta, w):
    """Combine belief degrees the long way, one rule at a time.

    This is the recursive form of the ER algorithm as stated in Yang and Xu
    (2013), Eqs. (14) to (17). It is deliberately a separate implementation
    from the analytical formula under test, so that the two agreeing is
    evidence about the formula rather than about shared code.

    Returns the combined belief degrees and the degree of global ignorance.
    """
    n_rules = beta.shape[0]

    # Eq. (14). The residual support of a rule splits in two: the part left
    # over because the rule is incomplete, and the part left over because the
    # rule carries less than all of the weight.
    m = w[0] * beta[0]
    m_theta = w[0] * (1.0 - beta[0].sum())
    m_powerset = 1.0 - w[0]

    for k in range(1, n_rules):
        m_next = w[k] * beta[k]
        m_theta_next = w[k] * (1.0 - beta[k].sum())
        m_powerset_next = 1.0 - w[k]

        # Eq. (16d).
        cross = np.outer(m, m_next)
        normaliser = 1.0 / (1.0 - (cross.sum() - np.trace(cross)))

        # Eqs. (16a) to (16c).
        m, m_theta, m_powerset = (
            normaliser
            * (m * m_next + m * (m_theta_next + m_powerset_next) + (m_theta + m_powerset) * m_next),
            normaliser
            * (m_theta * m_theta_next + m_theta * m_powerset_next + m_powerset * m_theta_next),
            normaliser * (m_powerset * m_powerset_next),
        )

    # Eqs. (17a) and (17b).
    return m / (1.0 - m_powerset), m_theta / (1.0 - m_powerset)


def test_combined_belief_matches_recursive_er_when_complete():
    """The analytical formula agrees with the recursive algorithm."""
    bre = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    weights = np.array([0.5, 0.3, 0.2])

    beta = compute_combined_belief_degrees(bre, weights[np.newaxis, :])[0]
    expected, ignorance = _recursive_er(bre, weights)

    assert_allclose(beta, expected, atol=1e-10)
    assert_allclose(ignorance, 0.0, atol=1e-10)


def test_combined_belief_matches_recursive_er_when_incomplete():
    """Agreement holds once rules stop assigning all of their belief.

    The analytical formula keeps the row sum of the rule base separate from
    one, which is what lets it carry the ignorance of an incomplete rule
    through the combination instead of silently dropping it.
    """
    rng = np.random.default_rng(20260825)

    for _ in range(200):
        n_rules = int(rng.integers(2, 6))
        n_consequents = int(rng.integers(2, 5))

        raw = rng.random((n_rules, n_consequents))
        # Leave each rule a random amount of belief unassigned.
        totals = rng.random(n_rules) * rng.choice([1.0, 0.6, 0.3], size=n_rules)
        bre = raw / raw.sum(axis=1, keepdims=True) * totals[:, np.newaxis]

        weights = rng.random(n_rules)
        weights /= weights.sum()

        beta = compute_combined_belief_degrees(bre, weights[np.newaxis, :])[0]
        expected, ignorance = _recursive_er(bre, weights)

        assert_allclose(beta, expected, atol=1e-9)
        assert_allclose(1.0 - beta.sum(), ignorance, atol=1e-9)


def test_utility_bounds_collapse_when_complete():
    """A complete assessment leaves nothing for the bounds to disagree about."""
    consequents = np.array([0.0, 0.5, 1.0])
    belief_degrees = np.array([[0.2, 0.3, 0.5]])

    lower, upper = compute_utility_bounds(belief_degrees, consequents)
    assert_allclose(lower, upper, atol=1e-12)
    assert_allclose(lower, compute_output(belief_degrees, consequents), atol=1e-12)


def test_utility_bounds_span_the_unassigned_belief():
    """Unassigned belief could have gone to any grade, so it widens the interval."""
    consequents = np.array([0.0, 0.5, 1.0])
    # A quarter of the belief is unassigned.
    belief_degrees = np.array([[0.25, 0.25, 0.25]])

    lower, upper = compute_utility_bounds(belief_degrees, consequents)
    assert_allclose(lower, 0.375, atol=1e-12)  # all of it to the worst grade
    assert_allclose(upper, 0.625, atol=1e-12)  # all of it to the best grade
    assert_allclose(upper - lower, 0.25, atol=1e-12)


def test_utility_bounds_total_ignorance_spans_the_whole_range():
    """A rule base that says nothing bounds the output only by its own scale."""
    consequents = np.array([2.0, 5.0, 9.0])
    belief_degrees = np.zeros((1, 3))

    lower, upper = compute_utility_bounds(belief_degrees, consequents)
    assert_allclose(lower, 2.0, atol=1e-12)
    assert_allclose(upper, 9.0, atol=1e-12)


def test_output_is_the_midpoint_of_the_bounds():
    """The point estimate sits exactly between the two bounds.

    This is the invariant the first version of these bounds violated. With a
    utility whose least preferred grade is not zero, the old output fell below
    its own lower bound.
    """
    consequents = np.array([0.0, 0.5, 1.0])
    belief_degrees = np.array([[0.25, 0.25, 0.25]])

    def utility(d):
        return 10.0 * d + 5.0

    lower, upper = compute_utility_bounds(belief_degrees, consequents, utility)
    output = compute_output(belief_degrees, consequents, utility)

    assert_allclose(output, 0.5 * (lower + upper), atol=1e-12)
    # assigned = 0.25*(5 + 10 + 15) = 7.5, and the unassigned quarter is worth
    # the average of the extreme grades, 0.25 * (5 + 15) / 2 = 2.5.
    assert_allclose(output, 10.0, atol=1e-9)
    assert_allclose(lower, 8.75, atol=1e-9)
    assert_allclose(upper, 11.25, atol=1e-9)


def test_output_is_bracketed_by_the_bounds():
    """The bounds bracket the output whatever the utility and the ignorance."""
    rng = np.random.default_rng(20260826)

    for _ in range(300):
        n_consequents = int(rng.integers(2, 6))
        # An offset utility is what exposes the bug: with the identity on
        # consequents starting at zero, the lower bound and the old output
        # happened to agree.
        consequents = np.sort(rng.random(n_consequents) * 20.0 - 10.0)

        raw = rng.random((1, n_consequents))
        total = rng.random() * rng.choice([1.0, 0.5])
        belief_degrees = raw / raw.sum() * total

        lower, upper = compute_utility_bounds(belief_degrees, consequents)
        output = compute_output(belief_degrees, consequents)

        assert lower[0] <= output[0] + 1e-12
        assert output[0] <= upper[0] + 1e-12


def test_output_unchanged_for_a_complete_assessment():
    """A complete assessment still gives the plain weighted sum."""
    rng = np.random.default_rng(7)

    for _ in range(100):
        n_consequents = int(rng.integers(2, 6))
        consequents = np.sort(rng.random(n_consequents) * 20.0 - 10.0)
        raw = rng.random((1, n_consequents))
        belief_degrees = raw / raw.sum()

        assert_allclose(
            compute_output(belief_degrees, consequents),
            belief_degrees @ consequents,
            atol=1e-12,
        )
