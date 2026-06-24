"""Tests for desdeo_brb.utils."""

import numpy as np
from numpy.testing import assert_array_equal

from desdeo_brb.utils import (
    build_rule_antecedent_indices,
    cartesian_product,
    generate_uniform_referential_values,
    pad_referential_values,
    unpad_referential_values,
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


# ── pad / unpad referential values roundtrip tests ───────────────────────


class TestPadReferentialValues:
    """Tests for ``pad_referential_values`` and ``unpad_referential_values``."""

    def test_roundtrip_ragged_lengths(self):
        """unpad(pad(x)) == x for ragged arrays."""
        x = [np.array([0., 1.]), np.array([0., .5, 1.]),
             np.array([0., .25, .5, .75, 1.])]
        padded, lengths = pad_referential_values(x)
        unpad = unpad_referential_values(padded, lengths)
        assert len(unpad) == len(x)
        for a, b in zip(x, unpad):
            assert_array_equal(a, b)

    def test_padding_shape(self):
        """padded has shape (n_attributes, max_len)."""
        x = [np.array([1., 2., 3.]), np.array([4.])]
        padded, lengths = pad_referential_values(x)
        assert padded.shape == (2, 3)
        assert_array_equal(lengths, [3, 1])

    def test_padding_values_are_inf(self):
        """Tail padding entries are np.inf; real values are not."""
        x = [np.array([1., 2.]), np.array([3., 4., 5.])]
        padded, _ = pad_referential_values(x)
        assert padded[0, 2] == np.inf
        # slice is empty (no padding) since length matches max_len
        assert padded[1, 3:].size == 0
        assert not np.isinf(padded[0, :2]).any()
        assert not np.isinf(padded[1, :3]).any()

    def test_uniform_lengths_no_padding(self):
        """When all arrays share a length, no inf appears in padded."""
        x = [np.array([1., 2., 3.]), np.array([4., 5., 6.])]
        padded, lengths = pad_referential_values(x)
        assert padded.shape == (2, 3)
        assert not np.isinf(padded).any()
        assert_array_equal(lengths, [3, 3])

    def test_single_attribute(self):
        """A one-element list roundtrips correctly."""
        x = [np.array([10., 20., 30.])]
        padded, lengths = pad_referential_values(x)
        assert padded.shape == (1, 3)
        unpad = unpad_referential_values(padded, lengths)
        assert_array_equal(x[0], unpad[0])

    def test_unpad_returns_independent_copies(self):
        """Mutating unpad result does not affect padded."""
        x = [np.array([1., 2., 3.])]
        padded, lengths = pad_referential_values(x)
        unpad = unpad_referential_values(padded, lengths)
        unpad[0][0] = 999.
        assert padded[0, 0] == 1.0
