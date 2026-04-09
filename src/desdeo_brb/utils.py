"""Utility and helper functions for the BRB system.

Provides common operations such as generating referential value grids
and computing Cartesian products of arrays with varying lengths.
"""

import numpy as np


def cartesian_product(list_of_arrays: list[np.ndarray]) -> np.ndarray:
    """Compute the Cartesian product of a list of 1-D arrays.

    Supports arrays of varying lengths.

    Args:
        list_of_arrays: List of 1-D numpy arrays.

    Returns:
        2-D array of shape ``(N, len(list_of_arrays))`` where *N* is the
        product of the lengths of all input arrays. Each row is one
        combination drawn from the input arrays.
    """
    grids = np.meshgrid(*list_of_arrays, indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def generate_uniform_referential_values(
    low: float, high: float, n: int
) -> np.ndarray:
    """Generate *n* uniformly spaced referential values between *low* and *high*.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        n: Number of referential values to generate.

    Returns:
        1-D array of *n* evenly spaced values from *low* to *high*.
    """
    return np.linspace(low, high, n)


def build_rule_antecedent_indices(
    referential_values: list[np.ndarray],
) -> np.ndarray:
    """Generate all possible rule antecedent index combinations.

    Produces the full Cartesian product of indices into each attribute's
    referential values array. If attribute 0 has 3 values and attribute 1
    has 4 values, the result has shape ``(12, 2)``.

    Args:
        referential_values: List of 1-D arrays, one per attribute.

    Returns:
        2-D integer array of shape ``(n_rules, n_attributes)`` containing
        all index combinations.
    """
    index_arrays = [np.arange(len(rv)) for rv in referential_values]
    return cartesian_product(index_arrays).astype(int)
