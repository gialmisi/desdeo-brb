"""Training with belief rows that may sum to less than one.

Yang et al. (2007), constraint 12b, requires the sum-to-one equality only when
a complete trained rule base is wanted. Otherwise the sum is capped at one and
the shortfall is the rule's ignorance. These tests cover that mode across every
optimizer path, and check that the default mode is unchanged.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.brb import BRBModel
from desdeo_brb.jax_backend import JAX_AVAILABLE
from desdeo_brb.models import RuleBase

PRECEDENT_RV = [np.linspace(0.0, 1.0, 4)]
CONSEQUENT_RV = np.array([0.0, 0.5, 1.0])


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 1))
    y = (np.sin(6.0 * X[:, 0]) + 1.0) / 2.0
    return X, y


def _incomplete_rule_base():
    """A rule base whose middle rules decline to assign all of their belief."""
    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV)
    bd = model.rule_base.belief_degrees.copy()
    bd[1:3] *= 0.5
    return RuleBase(
        precedent_referential_values=model.rule_base.precedent_referential_values,
        consequent_referential_values=model.rule_base.consequent_referential_values,
        belief_degrees=bd,
        rule_weights=model.rule_base.rule_weights,
        attribute_weights=model.rule_base.attribute_weights,
        rule_antecedent_indices=model.rule_base.rule_antecedent_indices,
    )


NUMPY_METHODS = ["SLSQP", "trust-constr", "DE"]


@pytest.mark.parametrize("method", NUMPY_METHODS)
def test_complete_by_default_on_every_numpy_method(method):
    """Without asking for it, training still returns a complete rule base."""
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV)
    model.fit(X, y, method=method, optimizer_options={"maxiter": 40})

    assert_allclose(model.rule_base.belief_degrees.sum(axis=1), 1.0, atol=1e-6)
    assert model.rule_base.is_complete


@pytest.mark.parametrize("method", NUMPY_METHODS)
def test_rows_never_exceed_one_when_incomplete_is_allowed(method):
    """The cap still holds. Only the equality is gone."""
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV)
    model.fit(X, y, method=method, allow_incomplete=True, optimizer_options={"maxiter": 40})

    row_sums = model.rule_base.belief_degrees.sum(axis=1)
    assert np.all(row_sums <= 1.0 + 1e-6)
    assert np.all(model.rule_base.belief_degrees >= -1e-9)


def test_an_incomplete_rule_base_stays_incomplete_by_default():
    """The default follows the rule base rather than silently completing it.

    Forcing completeness here would discard ignorance the caller deliberately
    expressed, which is the whole point of being able to express it.
    """
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV, rule_base=_incomplete_rule_base())
    assert not model.rule_base.is_complete

    model.fit(X, y, method="SLSQP", optimizer_options={"maxiter": 40})

    assert not model.rule_base.is_complete
    assert np.all(model.rule_base.belief_degrees.sum(axis=1) <= 1.0 + 1e-6)


def test_an_incomplete_rule_base_can_be_completed_on_request():
    """Passing the flag explicitly still overrides what the rule base implies."""
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV, rule_base=_incomplete_rule_base())

    model.fit(X, y, method="SLSQP", allow_incomplete=False, optimizer_options={"maxiter": 40})

    assert model.rule_base.is_complete


def test_ignorance_can_grow_when_the_data_prefers_a_vague_rule():
    """Pure noise about the midpoint is a target that rewards not committing.

    Under the average expected utility, total ignorance predicts the midpoint of
    the utility range. A target sitting there is the case where declining to
    commit is the honest answer, so ignorance should be able to reach it from a
    complete starting rule base.
    """
    rng = np.random.default_rng(1)
    X = rng.random((200, 1))
    y = 0.5 + rng.normal(scale=0.30, size=200)

    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV)
    model.fit(X, y, method="SLSQP", allow_incomplete=True, optimizer_options={"maxiter": 300})

    assert model.rule_base.ignorance.max() > 0.05


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("allow_incomplete", [False, True])
@pytest.mark.parametrize(
    "consequents",
    [CONSEQUENT_RV, [CONSEQUENT_RV, np.array([100.0, 250.0, 400.0])]],
    ids=["single_output", "two_outputs"],
)
def test_jax_reparameterisation_matches_the_numpy_decoder(allow_incomplete, consequents):
    """The optimizer minimises the traced function; we read back with the other.

    If the two reparameterisations disagree, the fitted parameters do not mean
    what the resulting rule base says they mean.
    """
    import jax.numpy as jnp

    from desdeo_brb.jax_backend import full_inference_jax_unconstrained

    rng = np.random.default_rng(3)
    X = rng.random((25, 1))

    model = BRBModel(PRECEDENT_RV, consequents, backend="jax")
    model._allow_incomplete = allow_incomplete
    rule_base = model.rule_base

    flat = model._flatten_params_unconstrained(True)
    flat = flat + rng.normal(scale=2.0, size=flat.shape)

    y_traced = np.asarray(
        full_inference_jax_unconstrained(
            jnp.asarray(flat),
            jnp.asarray(X),
            jnp.asarray(model._consequent_utilities()),
            jnp.asarray(rule_base.rule_antecedent_indices),
            rule_base.n_rules,
            rule_base.n_consequents,
            rule_base.n_attributes,
            tuple(model._ref_value_lengths),
            normalize_rule_weights=True,
            allow_incomplete=allow_incomplete,
            group_sizes=rule_base.consequent_group_sizes,
        )
    )

    decoded = model._unflatten_from_unconstrained(flat, True)
    y_decoded = BRBModel(
        PRECEDENT_RV, consequents, rule_base=decoded, backend="numpy"
    ).predict_values(X)

    assert_allclose(y_traced, y_decoded, atol=1e-9)

    # The decoder normalises each output on its own. Doing it across the whole
    # concatenated row instead leaves every block summing to well under one
    # while the row sums to one, which is what this catches.
    if not allow_incomplete:
        assert_allclose(decoded.block_sums, 1.0, atol=1e-9)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_jax_training_respects_the_cap():
    X, y = _data()
    model = BRBModel(PRECEDENT_RV, CONSEQUENT_RV, backend="jax")
    model.fit(X, y, allow_incomplete=True, optimizer_options={"maxiter": 40})

    assert np.all(model.rule_base.belief_degrees.sum(axis=1) <= 1.0 + 1e-6)
