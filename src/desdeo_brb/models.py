"""Pydantic models for BRB data structures.

Defines the core data containers used throughout the BRB system:
rule bases, inference results, and trainable parameter containers.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class RuleBase(BaseModel):
    """Holds the complete specification of a Belief Rule Base.

    Attributes:
        precedent_referential_values: List of 1D sorted arrays, one per
            attribute. Arrays may have varying lengths.
        consequent_referential_values: 1D sorted array of consequent values.
        belief_degrees: Shape ``(n_rules, n_consequents)``, rows sum to 1.
        rule_weights: Shape ``(n_rules,)``, sums to 1, values in [0, 1].
        attribute_weights: Shape ``(n_rules, n_attributes)``, values >= 0.
        rule_antecedent_indices: Shape ``(n_rules, n_attributes)``, integer
            indices into the precedent referential value arrays.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    precedent_referential_values: list[np.ndarray]
    consequent_referential_values: np.ndarray
    belief_degrees: np.ndarray
    rule_weights: np.ndarray
    attribute_weights: np.ndarray
    rule_antecedent_indices: np.ndarray

    @model_validator(mode="after")
    def _validate_all(self) -> "RuleBase":
        n_rules = len(self.rule_weights)
        n_attributes = len(self.precedent_referential_values)
        n_consequents = len(self.consequent_referential_values)

        for i, rv in enumerate(self.precedent_referential_values):
            if len(rv) > 1 and not np.all(rv[:-1] <= rv[1:]):
                raise ValueError(
                    f"precedent_referential_values[{i}] must be sorted ascending"
                )
        if len(self.consequent_referential_values) > 1 and not np.all(
            self.consequent_referential_values[:-1]
            <= self.consequent_referential_values[1:]
        ):
            raise ValueError("consequent_referential_values must be sorted ascending")

        if self.belief_degrees.shape != (n_rules, n_consequents):
            raise ValueError(
                f"belief_degrees shape {self.belief_degrees.shape} does not match "
                f"expected ({n_rules}, {n_consequents})"
            )
        if self.attribute_weights.shape != (n_rules, n_attributes):
            raise ValueError(
                f"attribute_weights shape {self.attribute_weights.shape} does not "
                f"match expected ({n_rules}, {n_attributes})"
            )
        if self.rule_antecedent_indices.shape != (n_rules, n_attributes):
            raise ValueError(
                f"rule_antecedent_indices shape {self.rule_antecedent_indices.shape} "
                f"does not match expected ({n_rules}, {n_attributes})"
            )

        row_sums = self.belief_degrees.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Each row of belief_degrees must sum to 1 (got row sums: {row_sums})"
            )

        if not np.allclose(self.rule_weights.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"rule_weights must sum to 1 (got {self.rule_weights.sum()})"
            )

        if np.any(self.attribute_weights < 0):
            raise ValueError("attribute_weights must be non-negative")

        return self

    @property
    def n_rules(self) -> int:
        return len(self.rule_weights)

    @property
    def n_attributes(self) -> int:
        return len(self.precedent_referential_values)

    @property
    def n_consequents(self) -> int:
        return len(self.consequent_referential_values)


class InferenceResult(BaseModel):
    """Container for the full trace of a BRB inference call.

    Attributes:
        input_belief_distributions: List of arrays, one per attribute, each
            of shape ``(n_samples, n_ref_values_i)``.
        activation_weights: Shape ``(n_samples, n_rules)``.
        combined_belief_degrees: Shape ``(n_samples, n_consequents)``.
        consequent_values: 1-D array of consequent referential values.
        output: Shape ``(n_samples,)``, scalar numerical outputs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_belief_distributions: list[np.ndarray]
    activation_weights: np.ndarray
    combined_belief_degrees: np.ndarray
    consequent_values: np.ndarray
    output: np.ndarray

    def dominant_rules(self, top_k: int = 3) -> np.ndarray:
        """Return the indices of the top-k most activated rules per sample.

        Args:
            top_k: Number of dominant rules to return.

        Returns:
            Integer array of shape ``(n_samples, top_k)`` with rule indices
            sorted by activation weight descending.
        """
        # argsort in descending order and take top_k
        sorted_indices = np.argsort(-self.activation_weights, axis=1)
        return sorted_indices[:, :top_k]

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary of the inference result.

        Numpy arrays are converted to nested Python lists.
        """
        return {
            "input_belief_distributions": [
                a.tolist() for a in self.input_belief_distributions
            ],
            "activation_weights": self.activation_weights.tolist(),
            "combined_belief_degrees": self.combined_belief_degrees.tolist(),
            "consequent_values": self.consequent_values.tolist(),
            "output": self.output.tolist(),
        }
