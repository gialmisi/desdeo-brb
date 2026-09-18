"""Tests for desdeo_brb.models."""

import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from desdeo_brb.models import InferenceResult, RuleBase


def _make_valid_rule_base() -> RuleBase:
    """Create a minimal valid RuleBase for testing."""
    return RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])],
        consequent_referential_values=np.array([0.0, 1.0]),
        belief_degrees=np.array(
            [
                [0.5, 0.5],
                [0.3, 0.7],
                [0.8, 0.2],
                [0.4, 0.6],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        rule_weights=np.full(6, 1.0 / 6),
        attribute_weights=np.ones((6, 2)),
        rule_antecedent_indices=np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
            ]
        ),
    )


def test_rule_base_valid_construction():
    """Create a valid RuleBase and verify properties."""
    rb = _make_valid_rule_base()
    assert rb.n_rules == 6
    assert rb.n_attributes == 2
    assert rb.n_consequents == 2


def _rule_base(belief_degrees):
    """Build a one-attribute rule base around the given belief degrees."""
    belief_degrees = np.asarray(belief_degrees)
    n_rules = belief_degrees.shape[0]
    return RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0])],
        consequent_referential_values=np.array([0.0, 1.0]),
        belief_degrees=belief_degrees,
        rule_weights=np.full(n_rules, 1.0 / n_rules),
        attribute_weights=np.ones((n_rules, 1)),
        rule_antecedent_indices=np.zeros((n_rules, 1), dtype=int),
    )


def test_rule_base_accepts_incomplete_belief_degrees():
    """A row summing to less than 1 is an incomplete rule, not an invalid one.

    RIMER (Yang et al. 2006, Eq. 3) requires only that a rule's belief degrees
    sum to at most one. The shortfall is the degree of ignorance about that
    rule's consequent, and a row of zeros is total ignorance.
    """
    rule_base = _rule_base([[0.3, 0.3], [0.5, 0.5], [0.0, 0.0]])

    np.testing.assert_allclose(rule_base.ignorance, [0.4, 0.0, 1.0])
    assert not rule_base.is_complete


def test_rule_base_reports_a_complete_rule_base_as_complete():
    assert _rule_base([[0.5, 0.5], [1.0, 0.0]]).is_complete


def test_rule_base_belief_degree_validation():
    """Rows summing to more than 1 raise ValidationError.

    Belief may be withheld but not over-assigned: more than a rule's worth of
    belief has no reading in the evidential reasoning combination.
    """
    with pytest.raises(ValidationError, match="belief_degrees"):
        _rule_base([[0.7, 0.6]])  # sums to 1.3


def test_rule_base_negative_belief_degree():
    """Negative belief degrees raise ValidationError."""
    with pytest.raises(ValidationError, match="belief_degrees"):
        _rule_base([[-0.1, 0.5]])


def test_rule_base_rule_weight_validation():
    """Weights not summing to 1 raises ValidationError."""
    with pytest.raises(ValidationError, match="rule_weights"):
        RuleBase(
            precedent_referential_values=[np.array([0.0, 1.0])],
            consequent_referential_values=np.array([0.0, 1.0]),
            belief_degrees=np.array([[0.5, 0.5], [0.5, 0.5]]),
            rule_weights=np.array([0.3, 0.3]),  # sums to 0.6
            attribute_weights=np.ones((2, 1)),
            rule_antecedent_indices=np.array([[0], [1]]),
        )


def test_rule_base_negative_attribute_weight():
    """Negative attribute weight raises ValidationError."""
    with pytest.raises(ValidationError, match="attribute_weights"):
        RuleBase(
            precedent_referential_values=[np.array([0.0, 1.0])],
            consequent_referential_values=np.array([0.0, 1.0]),
            belief_degrees=np.array([[0.5, 0.5]]),
            rule_weights=np.array([1.0]),
            attribute_weights=np.array([[-1.0]]),
            rule_antecedent_indices=np.array([[0]]),
        )


def test_rule_base_unsorted_referential_values():
    """Unsorted precedent referential values raises ValidationError."""
    with pytest.raises(ValidationError, match="sorted ascending"):
        RuleBase(
            precedent_referential_values=[np.array([1.0, 0.0])],  # unsorted
            consequent_referential_values=np.array([0.0, 1.0]),
            belief_degrees=np.array([[0.5, 0.5]]),
            rule_weights=np.array([1.0]),
            attribute_weights=np.ones((1, 1)),
            rule_antecedent_indices=np.array([[0]]),
        )


def test_rule_base_shape_mismatch():
    """Inconsistent shapes raise ValidationError."""
    with pytest.raises(ValidationError, match="shape"):
        RuleBase(
            precedent_referential_values=[np.array([0.0, 1.0])],
            consequent_referential_values=np.array([0.0, 1.0]),
            belief_degrees=np.array([[0.5, 0.5]]),
            rule_weights=np.array([1.0]),
            attribute_weights=np.ones((1, 3)),  # wrong: 3 attrs but only 1
            rule_antecedent_indices=np.array([[0]]),
        )


def test_describe_rule_single_attribute():
    """describe_rule formats correctly for a single attribute."""
    rb = _make_valid_rule_base()
    desc = rb.describe_rule(0)
    assert "Rule 0" in desc
    assert "IF" in desc
    assert "THEN" in desc
    assert "x1" in desc


def test_describe_rule_multi_attribute():
    """describe_rule with multiple attributes uses AND."""
    rb = _make_valid_rule_base()  # has 2 attributes
    desc = rb.describe_rule(0)
    assert "AND" in desc
    assert "x1" in desc
    assert "x2" in desc


def test_describe_rule_with_names():
    """Custom attribute and consequent names appear in output."""
    rb = _make_valid_rule_base()
    desc = rb.describe_rule(
        0,
        attribute_names=["Temperature", "Pressure"],
        consequent_name="Risk",
    )
    assert "Temperature" in desc
    assert "Pressure" in desc
    assert "Risk" in desc


def test_describe_rule_hides_zero_beliefs():
    """Zero beliefs are hidden by default."""
    rb = RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0])],
        consequent_referential_values=np.array([0.0, 1.0, 2.0]),
        belief_degrees=np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]),
        rule_weights=np.array([0.5, 0.5]),
        attribute_weights=np.ones((2, 1)),
        rule_antecedent_indices=np.array([[0], [1]]),
    )
    desc = rb.describe_rule(0, show_zero_beliefs=False)
    # Only the nonzero entry (1.0: 1.000) should appear
    assert "1: 1.000" in desc
    assert "0: 0.000" not in desc


def test_describe_all_rules():
    """describe_all_rules returns one line per rule."""
    rb = _make_valid_rule_base()
    text = rb.describe_all_rules()
    lines = text.strip().split("\n")
    assert len(lines) == rb.n_rules


def test_explain_basic():
    """explain produces readable output with expected sections."""
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[0.6, 0.3, 0.1]]),
        combined_belief_degrees=np.array([[0.4, 0.6]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.6]),
    )
    text = result.explain()
    assert "Prediction:" in text
    assert "Top activated rules:" in text
    assert "Combined belief" in text


def test_explain_with_rule_base():
    """explain with rule_base shows antecedent values."""
    rb = RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0])],
        consequent_referential_values=np.array([0.0, 1.0]),
        belief_degrees=np.array([[0.8, 0.2], [0.3, 0.7]]),
        rule_weights=np.array([0.5, 0.5]),
        attribute_weights=np.ones((2, 1)),
        rule_antecedent_indices=np.array([[0], [1]]),
    )
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[0.6, 0.4]]),
        combined_belief_degrees=np.array([[0.5, 0.5]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.5]),
    )
    text = result.explain(rule_base=rb, attribute_names=["Temp"])
    assert "Temp=" in text


def test_explain_without_rule_base():
    """explain without rule_base still works (indices only)."""
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[0.6, 0.4]]),
        combined_belief_degrees=np.array([[0.5, 0.5]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.5]),
    )
    text = result.explain(rule_base=None)
    assert "Rule 0" in text
    assert "w=" in text


def test_model_explain_convenience():
    """BRBModel.explain() convenience method works."""
    from desdeo_brb import BRBModel

    prv = [np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0, 2.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0])

    text = model.explain(np.array([[0.5]]), attribute_names=["x"])
    assert "Prediction:" in text
    assert "x=" in text
    assert "Combined belief" in text


def test_inference_result_dominant_rules():
    """Verify correct top-k indices."""
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[0.1, 0.5, 0.3, 0.05, 0.05]]),
        combined_belief_degrees=np.array([[0.6, 0.4]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.4]),
    )
    top = result.dominant_rules(top_k=3)
    assert top.shape == (1, 3)
    assert_array_equal(top[0], [1, 2, 0])


def test_inference_result_to_dict():
    """Verify dict output is JSON-serializable."""
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[0.6, 0.4]]),
        combined_belief_degrees=np.array([[0.7, 0.3]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.3]),
    )
    d = result.to_dict()
    # Must be JSON-serializable
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    assert "activation_weights" in d
    assert "output" in d


# Extended antecedents


def _extended_kwargs(**overrides):
    kwargs = {
        "precedent_referential_values": [np.array([0.0, 1.0, 2.0])],
        "consequent_referential_values": np.array([0.0, 1.0]),
        "belief_degrees": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "rule_weights": np.array([0.5, 0.5]),
        "attribute_weights": np.ones((2, 1)),
        "antecedent_beliefs": [np.array([[1.0, 0.0, 0.0], [0.0, 0.3, 0.7]])],
    }
    kwargs.update(overrides)
    return kwargs


def test_extended_rule_base_builds():
    rb = RuleBase(**_extended_kwargs())
    assert rb.rule_antecedent_indices is None
    assert rb.antecedent_beliefs is not None


def test_exactly_one_antecedent_form_is_required():
    """Neither form, or both at once, is a specification error."""
    with pytest.raises(ValueError, match="exactly one"):
        RuleBase(**_extended_kwargs(antecedent_beliefs=None))
    with pytest.raises(ValueError, match="exactly one"):
        RuleBase(**_extended_kwargs(rule_antecedent_indices=np.zeros((2, 1), dtype=int)))


def test_extended_antecedent_shape_is_checked():
    with pytest.raises(ValueError, match=r"antecedent_beliefs\[0\] shape"):
        RuleBase(**_extended_kwargs(antecedent_beliefs=[np.ones((2, 2))]))


def test_extended_antecedent_needs_one_array_per_attribute():
    with pytest.raises(ValueError, match="one per attribute"):
        RuleBase(**_extended_kwargs(antecedent_beliefs=[np.eye(3)[:2], np.eye(3)[:2]]))


def test_extended_antecedent_beliefs_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        RuleBase(
            **_extended_kwargs(antecedent_beliefs=[np.array([[1.0, 0.0, 0.0], [-0.1, 0.5, 0.6]])])
        )


def test_an_extended_antecedent_must_be_complete():
    """Unlike a consequent, an antecedent may not carry ignorance.

    Distance-based matching assumes both distributions have the same mass, and
    a rule holding less of it matches everything better. See
    ``test_incomplete_antecedents_would_match_everything_better`` for the
    behaviour this refusal prevents.
    """
    for antecedent in ([[0.4, 0.0, 0.0], [0.0, 0.3, 0.7]], [[0.0, 0.0, 0.0], [0.0, 0.3, 0.7]]):
        with pytest.raises(ValueError, match="must sum to 1"):
            RuleBase(**_extended_kwargs(antecedent_beliefs=[np.array(antecedent)]))


def test_an_extended_antecedent_may_not_exceed_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        RuleBase(
            **_extended_kwargs(antecedent_beliefs=[np.array([[1.0, 0.5, 0.0], [0.0, 0.3, 0.7]])])
        )


def _multi_output_rule_base() -> RuleBase:
    """A two-output rule base: three grades for y1, three for y2."""
    return RuleBase(
        precedent_referential_values=[np.array([0.0, 1.0])],
        consequent_referential_values=np.array([0.0, 0.5, 1.0, 100.0, 250.0, 400.0]),
        consequent_group_sizes=(3, 3),
        belief_degrees=np.array(
            [
                [0.2, 0.8, 0.0, 0.0, 0.4, 0.6],
                [0.0, 0.5, 0.5, 1.0, 0.0, 0.0],
            ]
        ),
        rule_weights=np.array([0.5, 0.5]),
        attribute_weights=np.ones((2, 1)),
        rule_antecedent_indices=np.array([[0], [1]]),
    )


def test_describe_rule_separates_outputs():
    """Each output gets its own distribution: its grades say nothing about another's."""
    rb = _multi_output_rule_base()
    desc = rb.describe_rule(0)
    # Default labels, one distribution per output, not one merged over all six grades.
    assert "y1 = {0: 0.200, 0.5: 0.800}" in desc
    assert "y2 = {250: 0.400, 400: 0.600}" in desc


def test_describe_rule_names_each_output():
    rb = _multi_output_rule_base()
    desc = rb.describe_rule(0, consequent_name=["Cost", "Yield"])
    assert "Cost = {0: 0.200, 0.5: 0.800}" in desc
    assert "Yield = {250: 0.400, 400: 0.600}" in desc
    assert "y1" not in desc


def test_describe_rule_refuses_one_name_for_several_outputs():
    """A single name cannot be spread over several outputs, so say so."""
    rb = _multi_output_rule_base()
    with pytest.raises(ValueError, match="pass a sequence of names"):
        rb.describe_rule(0, consequent_name="Everything")


def test_describe_rule_refuses_the_wrong_number_of_names():
    rb = _multi_output_rule_base()
    with pytest.raises(ValueError, match="expected 2 consequent names"):
        rb.describe_rule(0, consequent_name=["Cost", "Yield", "Extra"])


def test_describe_rule_single_output_is_unlabelled_by_default():
    """A single output keeps the bare distribution it has always printed."""
    rb = _make_valid_rule_base()
    assert (
        rb.describe_rule(0) == "Rule 0: IF x1 is 0 AND x2 is 0 THEN {0: 0.500, 1: 0.500} [w=0.167]"
    )


def test_describe_rule_shows_an_extended_antecedent_as_a_distribution():
    """An extended rule sits between referential values, so it prints them all."""
    rb = RuleBase(**_extended_kwargs())
    desc = rb.describe_rule(1)
    assert "x1 is {1: 0.300, 2: 0.700}" in desc
    assert "THEN {1: 1.000}" in desc


def test_describe_all_rules_covers_an_extended_rule_base():
    rb = RuleBase(**_extended_kwargs())
    lines = rb.describe_all_rules().strip().split("\n")
    assert len(lines) == rb.n_rules


def test_explain_uses_the_consequent_name():
    """The name labels the prediction and the combined distribution."""
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[1.0]]),
        combined_belief_degrees=np.array([[0.4, 0.6]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.6]),
    )
    text = result.explain(consequent_name="f(x)")
    assert "Prediction: f(x)=0.6" in text
    assert "  f(x): {0: 0.400, 1: 0.600}" in text


def test_explain_without_a_name_is_unlabelled():
    """Omitting the name leaves the output exactly as it was before."""
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[1.0]]),
        combined_belief_degrees=np.array([[0.4, 0.6]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.6]),
    )
    text = result.explain()
    assert "Prediction: 0.6" in text
    assert "\n  {0: 0.400, 1: 0.600}" in text


def test_explain_names_several_outputs():
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.5, 0.5]])],
        activation_weights=np.array([[1.0]]),
        combined_belief_degrees=np.array([[0.4, 0.6, 0.3, 0.7]]),
        consequent_values=np.array([0.0, 1.0, 10.0, 20.0]),
        consequent_group_sizes=(2, 2),
        output=np.array([[0.6, 17.0]]),
    )
    text = result.explain(consequent_name=["Cost", "Yield"])
    assert "Prediction: Cost=0.6, Yield=17" in text
    assert "  Cost: {0: 0.400, 1: 0.600}" in text
    assert "  Yield: {10: 0.300, 20: 0.700}" in text


def test_explain_describes_extended_rules_by_their_antecedents():
    """An extended rule base no longer falls back to bare rule indices."""
    rb = RuleBase(**_extended_kwargs())
    result = InferenceResult(
        input_belief_distributions=[np.array([[0.0, 0.5, 0.5]])],
        activation_weights=np.array([[0.4, 0.6]]),
        combined_belief_degrees=np.array([[0.3, 0.7]]),
        consequent_values=np.array([0.0, 1.0]),
        output=np.array([0.7]),
    )
    text = result.explain(rule_base=rb)
    assert "x1={1: 0.300, 2: 0.700}" in text
    assert "x1={0: 1.000}" in text
