"""Pydantic models for BRB data structures.

Defines the core data containers used throughout the BRB system:
rule bases, inference results, and trainable parameter containers.
"""

from typing import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict


class RuleBase(BaseModel):
    """Holds the complete specification of a Belief Rule Base.

    Attributes:
        precedent_referential_values: List of arrays defining referential values
            for each antecedent attribute. Arrays may have varying lengths.
        consequent_referential_values: Array of referential values for the consequent.
        belief_rule_expression_matrix: Matrix encoding the belief degrees assigned
            to each consequent referential value for every rule.
        rule_weights: Weight associated with each rule.
        attribute_weights: Weight (importance) of each antecedent attribute.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    precedent_referential_values: list[np.ndarray]
    consequent_referential_values: np.ndarray
    belief_rule_expression_matrix: np.ndarray
    rule_weights: np.ndarray
    attribute_weights: np.ndarray


class InferenceResult(BaseModel):
    """Container for the full output of a BRB inference pass.

    Attributes:
        input_belief_distributions: Matching degrees (alphas) for each input
            against the precedent referential values.
        activation_weights: Activation weight of each rule for the given input.
        combined_belief_degrees: Combined belief degrees over the consequent
            referential values after the evidential reasoning step.
        consequent_values: The consequent referential values used.
        scalar_output: A single scalar output value derived from the combined
            belief degrees and consequent values.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_belief_distributions: np.ndarray
    activation_weights: np.ndarray
    combined_belief_degrees: np.ndarray
    consequent_values: np.ndarray
    scalar_output: float

    def dominant_rules(self, top_k: int) -> list:
        """Return the indices of the top-k most activated rules.

        Args:
            top_k: Number of dominant rules to return.

        Returns:
            List of rule indices sorted by activation weight descending.
        """
        pass


class Trainables(BaseModel):
    """Internal container for flattened parameters used during optimization.

    Packs and unpacks the trainable parameters of a BRB model into a flat
    vector suitable for use with scipy optimizers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pass
