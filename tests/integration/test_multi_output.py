"""Rule bases with more than one consequent attribute.

RIMER (Yang et al. 2006, Eq. 3) writes a rule's consequent as a distribution
over grades of one attribute. Several attributes are several such
distributions sharing the same antecedents, so the activation weights are
computed once and the evidential reasoning combination runs once per output.

Outputs keep their own grades, because objectives generally have their own
scales and units.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.brb import BRBModel
from desdeo_brb.inference import compute_combined_belief_degrees, compute_output
from desdeo_brb.jax_backend import JAX_AVAILABLE
from desdeo_brb.models import RuleBase

PRECEDENT_RV = [np.linspace(0.0, 1.0, 4)]
FIRST = np.array([0.0, 0.5, 1.0])
SECOND = np.array([100.0, 250.0, 400.0])
BOTH = [FIRST, SECOND]


def _data(n=80, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 1))
    y = np.column_stack([(np.sin(6.0 * X[:, 0]) + 1.0) / 2.0, 100.0 + 300.0 * X[:, 0] ** 2])
    return X, y


def test_a_single_output_rule_base_is_unchanged():
    """Passing one array behaves exactly as before, with no output axis."""
    model = BRBModel(PRECEDENT_RV, FIRST)

    assert model.rule_base.n_outputs == 1
    assert model.rule_base.consequent_group_sizes is None

    X, _ = _data()
    result = model.predict(X)
    assert result.output.shape == (len(X),)
    assert result.ignorance.shape == (len(X),)


def test_outputs_keep_their_own_grades():
    model = BRBModel(PRECEDENT_RV, BOTH)
    rule_base = model.rule_base

    assert rule_base.n_outputs == 2
    assert rule_base.group_sizes == (3, 3)
    assert_allclose(rule_base.consequent_values(0), FIRST)
    assert_allclose(rule_base.consequent_values(1), SECOND)
    # The default rule base is uniform within each block, not across the row.
    assert_allclose(rule_base.block_sums, 1.0, atol=1e-12)


def test_prediction_gains_an_output_axis():
    model = BRBModel(PRECEDENT_RV, BOTH)
    X, _ = _data(n=7)
    result = model.predict(X)

    assert result.output.shape == (7, 2)
    assert result.ignorance.shape == (7, 2)
    for bound in result.utility_bounds:
        assert bound.shape == (7, 2)
    # Each output lands inside its own range of grades.
    assert np.all((result.output[:, 0] >= FIRST.min()) & (result.output[:, 0] <= FIRST.max()))
    assert np.all((result.output[:, 1] >= SECOND.min()) & (result.output[:, 1] <= SECOND.max()))


def test_outputs_are_independent_of_each_other():
    """Combining two outputs together equals combining each on its own.

    The activation weights depend only on the antecedents, so an output cannot
    be influenced by what the rules say about a different objective.
    """
    model_both = BRBModel(PRECEDENT_RV, BOTH)
    X, _ = _data(n=12)

    joint = model_both.predict(X)

    for output, consequents in enumerate(BOTH):
        alone = BRBModel(
            PRECEDENT_RV,
            consequents,
            rule_base=RuleBase(
                precedent_referential_values=model_both.rule_base.precedent_referential_values,
                consequent_referential_values=consequents,
                belief_degrees=model_both.rule_base.beliefs_for(output),
                rule_weights=model_both.rule_base.rule_weights,
                attribute_weights=model_both.rule_base.attribute_weights,
                rule_antecedent_indices=model_both.rule_base.rule_antecedent_indices,
            ),
        ).predict(X)

        assert_allclose(joint.output[:, output], alone.output, atol=1e-12)


def test_a_block_may_be_incomplete_while_another_is_not():
    """Ignorance is per rule per output, not a property of the whole rule."""
    rule_base = RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0])],
        consequent_referential_values=np.concatenate(BOTH),
        consequent_group_sizes=(3, 3),
        belief_degrees=np.array(
            [
                [0.6, 0.0, 0.0, 0.5, 0.5, 0.0],  # vague about the first, sure about the second
                [0.0, 0.0, 1.0, 0.0, 0.3, 0.0],  # the other way round
            ]
        ),
        rule_weights=np.array([0.5, 0.5]),
        attribute_weights=np.ones((2, 1)),
        rule_antecedent_indices=np.array([[0], [1]]),
    )

    assert_allclose(rule_base.ignorance, [[0.4, 0.0], [0.0, 0.7]], atol=1e-12)
    assert not rule_base.is_complete


def test_over_assignment_is_rejected_per_output():
    """A block summing above one is invalid even if another block is empty."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="belief_degrees"):
        RuleBase(
            precedent_referential_values=[np.array([0.0, 1.0])],
            consequent_referential_values=np.concatenate(BOTH),
            consequent_group_sizes=(3, 3),
            belief_degrees=np.array([[0.6, 0.6, 0.0, 0.0, 0.0, 0.0]]),
            rule_weights=np.array([1.0]),
            attribute_weights=np.ones((1, 1)),
            rule_antecedent_indices=np.array([[0]]),
        )


def test_group_sizes_must_describe_the_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="consequent_group_sizes"):
        RuleBase(
            precedent_referential_values=[np.array([0.0, 1.0])],
            consequent_referential_values=np.concatenate(BOTH),
            consequent_group_sizes=(3, 2),
            belief_degrees=np.zeros((1, 6)),
            rule_weights=np.array([1.0]),
            attribute_weights=np.ones((1, 1)),
            rule_antecedent_indices=np.array([[0]]),
        )


def test_grades_need_only_be_sorted_within_an_output():
    """The concatenation is not sorted, and must not be required to be.

    The second output here starts below where the first ended.
    """
    RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0])],
        consequent_referential_values=np.array([10.0, 20.0, 1.0, 2.0]),
        consequent_group_sizes=(2, 2),
        belief_degrees=np.array([[0.5, 0.5, 0.5, 0.5]]),
        rule_weights=np.array([1.0]),
        attribute_weights=np.ones((1, 1)),
        rule_antecedent_indices=np.array([[0]]),
    )


def test_kernels_accept_group_sizes_directly():
    """The inference functions are usable without a model wrapped round them."""
    bre = np.array([[0.6, 0.4, 0.2, 0.8], [0.1, 0.9, 0.7, 0.3]])
    weights = np.array([[0.5, 0.5]])

    combined = compute_combined_belief_degrees(bre, weights, group_sizes=(2, 2))
    assert combined.shape == (1, 4)

    consequents = np.array([0.0, 1.0, 100.0, 200.0])
    output = compute_output(combined, consequents, group_sizes=(2, 2))
    assert output.shape == (1, 2)

    # Same answer as running each block through the single-output path.
    for o, block in enumerate((slice(0, 2), slice(2, 4))):
        alone = compute_combined_belief_degrees(bre[:, block], weights)
        assert_allclose(combined[:, block], alone, atol=1e-12)
        assert_allclose(output[:, o], compute_output(alone, consequents[block]), atol=1e-12)


def test_a_utility_function_per_output():
    """A sequence of utilities applies one to each output; None leaves it be."""
    model = BRBModel(PRECEDENT_RV, BOTH, utility_fn=[lambda d: d * 10.0, None])
    X, _ = _data(n=5)
    result = model.predict(X)

    plain = BRBModel(PRECEDENT_RV, BOTH).predict(X)
    assert_allclose(result.output[:, 0], plain.output[:, 0] * 10.0, atol=1e-9)
    assert_allclose(result.output[:, 1], plain.output[:, 1], atol=1e-9)


@pytest.mark.parametrize("method", ["SLSQP", "trust-constr", "DE"])
@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_training_respects_every_block(method, allow_incomplete):
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, BOTH)
    model.fit(
        X, y, method=method, allow_incomplete=allow_incomplete, optimizer_options={"maxiter": 40}
    )

    block_sums = model.rule_base.block_sums
    assert np.all(block_sums <= 1.0 + 1e-6)
    if not allow_incomplete:
        assert_allclose(block_sums, 1.0, atol=1e-6)


def test_a_mismatched_target_is_rejected():
    X, y = _data()
    with pytest.raises(ValueError, match="n_samples, 2"):
        BRBModel(PRECEDENT_RV, BOTH).fit(X, y[:, 0], method="SLSQP")

    with pytest.raises(ValueError, match="n_samples,"):
        BRBModel(PRECEDENT_RV, FIRST).fit(X, y, method="SLSQP")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_backends_agree_on_multi_output():
    X, _ = _data(n=15)
    outputs = {}
    for backend in ("numpy", "jax"):
        model = BRBModel(PRECEDENT_RV, BOTH, backend=backend, utility_fn=lambda d: 2.0 * d + 7.0)
        result = model.predict(X)
        outputs[backend] = result.output
        lower, upper = result.utility_bounds
        assert np.all(lower <= result.output + 1e-9)
        assert np.all(result.output <= upper + 1e-9)

    assert_allclose(outputs["numpy"], outputs["jax"], atol=1e-8)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_jax_training_respects_every_block(allow_incomplete):
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, BOTH, backend="jax")
    model.fit(X, y, allow_incomplete=allow_incomplete, optimizer_options={"maxiter": 40})

    block_sums = model.rule_base.block_sums
    assert np.all(block_sums <= 1.0 + 1e-6)
    if not allow_incomplete:
        assert_allclose(block_sums, 1.0, atol=1e-6)


def test_initial_rule_fn_gives_one_value_per_output():
    """Seeding from a function interpolates within each output's own grades.

    Interpolating across the concatenation instead would put a rule's whole
    belief in one output and leave the other empty.
    """
    model = BRBModel(
        PRECEDENT_RV,
        BOTH,
        initial_rule_fn=lambda x: np.array([x[0], 100.0 + 300.0 * x[0]]),
    )

    assert_allclose(model.rule_base.block_sums, 1.0, atol=1e-12)

    # The seeded rule base reproduces the function at the referential points.
    X = np.array([[0.0], [0.5], [1.0]])
    assert_allclose(model.predict_values(X), [[0.0, 100.0], [0.5, 250.0], [1.0, 400.0]], atol=1e-9)


def test_initial_rule_fn_arity_is_checked():
    with pytest.raises(ValueError, match="one per output"):
        BRBModel(PRECEDENT_RV, BOTH, initial_rule_fn=lambda x: x[0])


def test_initial_rule_fn_still_takes_a_scalar_for_one_output():
    model = BRBModel(PRECEDENT_RV, FIRST, initial_rule_fn=lambda x: x[0])
    assert_allclose(model.rule_base.belief_degrees.sum(axis=1), 1.0, atol=1e-12)


def _relative_rmse(model, X, y):
    """Each output's RMSE as a fraction of its own grade span."""
    spans = np.array([FIRST.max() - FIRST.min(), SECOND.max() - SECOND.min()])
    return np.sqrt(np.mean(((y - model.predict_values(X)) / spans) ** 2, axis=0))


def _same_shape_different_units(n=150, seed=11):
    """Two objectives with identical shape, one on a scale 300 times wider.

    A balanced fit should reach the same relative error on both. Anything else
    is the objective with wider units crowding out the other.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 1))
    shape = (np.sin(6.0 * X[:, 0]) + 1.0) / 2.0
    return X, np.column_stack([shape, 100.0 + 300.0 * shape])


def test_scaling_balances_outputs_with_different_units():
    X, y = _same_shape_different_units()

    model = BRBModel([np.linspace(0.0, 1.0, 5)], BOTH)
    model.fit(X, y, method="SLSQP", optimizer_options={"maxiter": 250})

    first, second = _relative_rmse(model, X, y)
    # Same underlying shape, so a balanced fit reaches the same relative error.
    assert first == pytest.approx(second, rel=0.25)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_scaling_is_what_balances_them():
    """Turning the scaling off lets the wider objective dominate.

    This is the behaviour the default exists to prevent, so it is worth
    pinning: without scaling the narrow objective is fitted far worse relative
    to its own span.
    """
    X, y = _same_shape_different_units()

    unscaled = BRBModel([np.linspace(0.0, 1.0, 5)], BOTH, backend="jax")
    unscaled.fit(X, y, scale_outputs=False, optimizer_options={"maxiter": 250})
    first_unscaled, second_unscaled = _relative_rmse(unscaled, X, y)

    scaled = BRBModel([np.linspace(0.0, 1.0, 5)], BOTH, backend="jax")
    scaled.fit(X, y, optimizer_options={"maxiter": 250})
    first_scaled, second_scaled = _relative_rmse(scaled, X, y)

    # Unscaled, the objective with the narrower units comes off far worse.
    assert first_unscaled > 5.0 * second_unscaled
    # Scaled, the two are fitted to comparable relative accuracy, and the
    # neglected objective improves.
    assert first_scaled == pytest.approx(second_scaled, rel=0.25)
    assert first_scaled < first_unscaled


def test_a_single_output_loss_is_left_alone():
    """Scaling must not change a single-output fit or its reported score."""
    X, y = _data()
    y = y[:, 0]

    scores = []
    for scale_outputs in (False, True):
        model = BRBModel(PRECEDENT_RV, FIRST)
        model.fit(
            X, y, method="SLSQP", scale_outputs=scale_outputs, optimizer_options={"maxiter": 40}
        )
        scores.append(model.score(X, y))

    assert scores[0] == pytest.approx(scores[1], rel=1e-12)


def test_a_degenerate_output_is_not_divided_by_zero():
    """An output whose grades all coincide has no span to scale by."""
    model = BRBModel(PRECEDENT_RV, [FIRST, np.array([5.0, 5.0, 5.0])])
    assert_allclose(model._output_scales(), [1.0, 1.0], atol=1e-12)
