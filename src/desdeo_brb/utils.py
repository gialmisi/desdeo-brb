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
        2-D array of shape (N, len(list_of_arrays)) where N is the product
        of the lengths of all input arrays.
    """
    pass


def generate_uniform_referential_values(
    low: float, high: float, n: int
) -> np.ndarray:
    """Generate n uniformly spaced referential values between low and high.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        n: Number of referential values to generate.

    Returns:
        1-D array of n evenly spaced values from low to high.
    """
    pass
