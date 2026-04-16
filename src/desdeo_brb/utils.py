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


def generate_uniform_referential_values(low: float, high: float, n: int) -> np.ndarray:
    """Generate *n* uniformly spaced referential values between *low* and *high*.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        n: Number of referential values to generate.

    Returns:
        1-D array of *n* evenly spaced values from *low* to *high*.
    """
    return np.linspace(low, high, n)


def pad_referential_values(
    ref_values: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad varying-length referential value arrays into a single 2D array.

    Required for JAX JIT compilation, which needs static array shapes.
    Unused entries are padded with ``np.inf``.

    Args:
        ref_values: List of 1D arrays, one per attribute.

    Returns:
        Tuple of ``(padded, lengths)`` where *padded* has shape
        ``(n_attributes, max_len)`` and *lengths* is an integer array
        of shape ``(n_attributes,)`` with the original lengths.
    """
    lengths = np.array([len(rv) for rv in ref_values], dtype=int)
    max_len = int(lengths.max())
    n_attributes = len(ref_values)
    padded = np.full((n_attributes, max_len), np.inf)
    for i, rv in enumerate(ref_values):
        padded[i, : len(rv)] = rv
    return padded, lengths


def unpad_referential_values(padded: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Inverse of :func:`pad_referential_values`.

    Args:
        padded: 2D array of shape ``(n_attributes, max_len)``.
        lengths: Integer array of original lengths per attribute.

    Returns:
        List of 1D arrays, one per attribute, with padding removed.
    """
    return [padded[i, : int(lengths[i])].copy() for i in range(len(lengths))]


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
        2D integer array of shape ``(n_rules, n_attributes)`` containing
        all index combinations.
    """
    index_arrays = [np.arange(len(rv)) for rv in referential_values]
    return cartesian_product(index_arrays).astype(int)
