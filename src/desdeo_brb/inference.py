"""Pure functions implementing the BRB inference pipeline.

Each function corresponds to one step in the BRB evidential reasoning process:
input transformation, activation weight computation, belief combination, and
output aggregation.
"""

from typing import Callable

import numpy as np


def input_transform(x: np.ndarray, referential_values: list[np.ndarray]) -> np.ndarray:
    """Transform a raw input vector into belief distributions over referential values.

    Computes matching degrees (alphas) describing how strongly each input
    component relates to each referential value of the corresponding attribute.

    Args:
        x: Input vector of shape (n_features,).
        referential_values: List of arrays, one per attribute, each containing
            the referential values for that attribute.

    Returns:
        Array of belief distributions (alphas).
    """
    pass


def compute_activation_weights(
    alphas: np.ndarray,
    rule_conditions: np.ndarray,
    thetas: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """Compute the activation weight of each rule given input belief distributions.

    Args:
        alphas: Input belief distributions from `input_transform`.
        rule_conditions: Matrix encoding which referential values each rule
            refers to for each attribute.
        thetas: Rule weights.
        deltas: Attribute weights.

    Returns:
        Array of activation weights, one per rule.
    """
    pass


def compute_combined_belief_degrees(
    bre_matrix: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Combine activated belief degrees using the evidential reasoning algorithm.

    Args:
        bre_matrix: Belief rule expression matrix of shape (n_rules, n_consequents).
        weights: Activation weights of shape (n_rules,).

    Returns:
        Combined belief degree array of shape (n_consequents,).
    """
    pass


def compute_output(
    belief_degrees: np.ndarray,
    consequents: np.ndarray,
    utility_fn: Callable[[np.ndarray], float] | None = None,
) -> float:
    """Compute a scalar output from combined belief degrees and consequent values.

    Args:
        belief_degrees: Combined belief degrees of shape (n_consequents,).
        consequents: Consequent referential values of shape (n_consequents,).
        utility_fn: Optional utility function. If None, a weighted average is used.

    Returns:
        Scalar output value.
    """
    pass
