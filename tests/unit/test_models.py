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
