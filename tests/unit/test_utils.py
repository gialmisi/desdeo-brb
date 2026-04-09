"""Tests for desdeo_brb.utils."""

import numpy as np
from numpy.testing import assert_array_equal

from desdeo_brb.utils import (
    build_rule_antecedent_indices,
    cartesian_product,
    generate_uniform_referential_values,
)


def test_cartesian_product_basic():
    """Two arrays of length 2 produce 4 rows."""
    result = cartesian_product([np.array([1, 2]), np.array([3, 4])])
    assert result.shape == (4, 2)
    expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]])
    assert_array_equal(result, expected)


def test_cartesian_product_varying_lengths():
    """Arrays of length 2 and 3 produce 6 rows."""
    result = cartesian_product([np.array([1, 2]), np.array([3, 4, 5])])
    assert result.shape == (6, 2)
    # First column cycles slowly, second column cycles fast
    assert_array_equal(result[:, 0], [1, 1, 1, 2, 2, 2])
    assert_array_equal(result[:, 1], [3, 4, 5, 3, 4, 5])


def test_generate_uniform_referential_values():
    """Generates evenly spaced values including endpoints."""
    rv = generate_uniform_referential_values(0.0, 1.0, 5)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(rv, expected)


def test_build_rule_antecedent_indices():
    """Verify shape and content for known inputs."""
    rv = [np.array([10, 20, 30]), np.array([1, 2, 3, 4])]
    indices = build_rule_antecedent_indices(rv)
    assert indices.shape == (12, 2)
    # First column: indices 0,1,2 into rv[0] (length 3), cycling slowly
    assert_array_equal(indices[:4, 0], [0, 0, 0, 0])
    assert_array_equal(indices[4:8, 0], [1, 1, 1, 1])
    # Second column: indices 0,1,2,3 into rv[1] (length 4), cycling fast
    assert_array_equal(indices[:4, 1], [0, 1, 2, 3])
    assert indices.dtype == int
