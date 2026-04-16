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
        belief_degrees=np.array([
            [0.5, 0.5],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.4, 0.6],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        rule_weights=np.full(6, 1.0 / 6),
        attribute_weights=np.ones((6, 2)),
        rule_antecedent_indices=np.array([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
        ]),
    )


def test_rule_base_valid_construction():
    """Create a valid RuleBase and verify properties."""
    rb = _make_valid_rule_base()
    assert rb.n_rules == 6
    assert rb.n_attributes == 2
    assert rb.n_consequents == 2


def test_rule_base_belief_degree_validation():
    """Rows not summing to 1 raises ValidationError."""
    with pytest.raises(ValidationError, match="belief_degrees"):
        RuleBase(
            precedent_referential_values=[np.array([0.0, 1.0])],
            consequent_referential_values=np.array([0.0, 1.0]),
            belief_degrees=np.array([[0.3, 0.3]]),  # sums to 0.6
            rule_weights=np.array([1.0]),
            attribute_weights=np.ones((1, 1)),
            rule_antecedent_indices=np.array([[0]]),
        )


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
