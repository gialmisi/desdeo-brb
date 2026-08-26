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
        consequent_referential_values: 1D array of consequent values, sorted
            ascending within each output. With several outputs their grades are
            concatenated here and delimited by ``consequent_group_sizes``.
        consequent_group_sizes: Number of grades belonging to each output, or
            ``None`` for a single output. Outputs keep their own grades because
            objectives generally have their own scales and units.
        belief_degrees: Shape ``(n_rules, n_consequents)``, non-negative, with
            each rule's block for each output summing to at most 1. A block
            summing to less than 1 is an incomplete rule: the shortfall is the
            degree of ignorance about that rule's consequent for that output,
            and a block of zeros is total ignorance.
        rule_weights: Shape ``(n_rules,)``, sums to 1, values in [0, 1].
        attribute_weights: Shape ``(n_rules, n_attributes)``, values >= 0.
        rule_antecedent_indices: Shape ``(n_rules, n_attributes)``, integer
            indices into the precedent referential value arrays.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    precedent_referential_values: list[np.ndarray]
    consequent_referential_values: np.ndarray
    consequent_group_sizes: tuple[int, ...] | None = None
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
                raise ValueError(f"precedent_referential_values[{i}] must be sorted ascending")
        if self.consequent_group_sizes is not None:
            if any(size < 1 for size in self.consequent_group_sizes):
                raise ValueError(
                    f"consequent_group_sizes must all be >= 1, got {self.consequent_group_sizes}"
                )
            if sum(self.consequent_group_sizes) != n_consequents:
                raise ValueError(
                    f"consequent_group_sizes {self.consequent_group_sizes} sum to "
                    f"{sum(self.consequent_group_sizes)}, but there are {n_consequents} "
                    "consequent referential values"
                )

        # Sorting is required within an output, not across the concatenation:
        # separate objectives have unrelated scales.
        for o, block in enumerate(self.consequent_slices):
            values = self.consequent_referential_values[block]
            if len(values) > 1 and not np.all(values[:-1] <= values[1:]):
                raise ValueError(
                    f"consequent_referential_values for output {o} must be sorted ascending"
                )

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

        if np.any(self.belief_degrees < 0):
            raise ValueError("belief_degrees must be non-negative")

        # RIMER (Yang et al. 2006, Eq. 3) requires only that a rule's belief
        # degrees sum to at most one. A shortfall is ignorance about that rule's
        # consequent, which the evidential reasoning combination carries through
        # to the result rather than discarding.
        block_sums = self.block_sums
        if np.any(block_sums > 1.0 + 1e-6):
            raise ValueError(
                "Each rule's belief_degrees must sum to at most 1 for every output "
                f"(got sums: {block_sums})"
            )

        if not np.allclose(self.rule_weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"rule_weights must sum to 1 (got {self.rule_weights.sum()})")

        if np.any(self.attribute_weights < 0):
            raise ValueError("attribute_weights must be non-negative")

        return self

    @property
    def n_outputs(self) -> int:
        """Return how many consequent attributes this rule base predicts."""
        if self.consequent_group_sizes is None:
            return 1
        return len(self.consequent_group_sizes)

    @property
    def group_sizes(self) -> tuple[int, ...]:
        """Return the grade count of each output, single output included."""
        if self.consequent_group_sizes is None:
            return (len(self.consequent_referential_values),)
        return self.consequent_group_sizes

    @property
    def consequent_slices(self) -> list[slice]:
        """Return the column range of each output within the concatenation."""
        slices = []
        start = 0
        for size in self.group_sizes:
            slices.append(slice(start, start + size))
            start += size
        return slices

    def consequent_values(self, output: int = 0) -> np.ndarray:
        """Return the referential values belonging to one output."""
        return self.consequent_referential_values[self.consequent_slices[output]]

    def beliefs_for(self, output: int = 0) -> np.ndarray:
        """Return the belief degrees belonging to one output.

        Shape ``(n_rules, n_grades_of_that_output)``.
        """
        return self.belief_degrees[:, self.consequent_slices[output]]

    @property
    def block_sums(self) -> np.ndarray:
        """Return each rule's assigned belief per output, shape ``(n_rules, n_outputs)``."""
        return np.stack(
            [self.belief_degrees[:, block].sum(axis=1) for block in self.consequent_slices],
            axis=1,
        )

    @property
    def n_rules(self) -> int:
        return len(self.rule_weights)

    @property
    def ignorance(self) -> np.ndarray:
        """Return the belief mass each rule leaves unassigned.

        Zero for a complete rule, one for a rule that says nothing about its
        consequent.

        Returns:
            Shape ``(n_rules,)`` for a single output, ``(n_rules, n_outputs)``
            otherwise.
        """
        ignorance = np.clip(1.0 - self.block_sums, 0.0, 1.0)
        return ignorance[:, 0] if self.n_outputs == 1 else ignorance

    @property
    def is_complete(self) -> bool:
        """Return whether every rule assigns all of its belief."""
        return bool(np.allclose(self.ignorance, 0.0, atol=1e-6))

    @property
    def n_attributes(self) -> int:
        return len(self.precedent_referential_values)

    @property
    def n_consequents(self) -> int:
        return len(self.consequent_referential_values)

    def describe_rule(
        self,
        k: int,
        attribute_names: list[str] | None = None,
        consequent_name: str | None = None,
        show_zero_beliefs: bool = False,
    ) -> str:
        """Return a human-readable description of rule *k*.

        Args:
            k: Rule index (0-based).
            attribute_names: Display names for each attribute. If ``None``,
                uses ``x1, x2, ...``.
            consequent_name: Display name for the consequent. If ``None``,
                omitted from the output.
            show_zero_beliefs: If ``False`` (default), skip consequent
                values whose belief degree is below 0.001.
        """
        if attribute_names is None:
            attribute_names = [f"x{i + 1}" for i in range(self.n_attributes)]

        # Build the IF clause
        conditions = []
        for i in range(self.n_attributes):
            idx = int(self.rule_antecedent_indices[k, i])
            val = float(self.precedent_referential_values[i][idx])
            conditions.append(f"{attribute_names[i]} is {val:.4g}")
        if_clause = " AND ".join(conditions)

        # Build the THEN clause
        crv = self.consequent_referential_values
        bd = self.belief_degrees[k]
        belief_parts = []
        for n in range(self.n_consequents):
            if not show_zero_beliefs and bd[n] < 0.001:
                continue
            belief_parts.append(f"{float(crv[n]):.4g}: {bd[n]:.3f}")
        then_clause = "{" + ", ".join(belief_parts) + "}"
        if consequent_name is not None:
            then_clause = f"{consequent_name} = {then_clause}"

        theta = float(self.rule_weights[k])
        return f"Rule {k}: IF {if_clause} THEN {then_clause} [w={theta:.3f}]"

    def describe_all_rules(
        self,
        attribute_names: list[str] | None = None,
        consequent_name: str | None = None,
        show_zero_beliefs: bool = False,
    ) -> str:
        """Return a multi-line string describing every rule."""
        return "\n".join(
            self.describe_rule(k, attribute_names, consequent_name, show_zero_beliefs)
            for k in range(self.n_rules)
        )


class InferenceResult(BaseModel):
    """Container for the full trace of a BRB inference call.

    Attributes:
        input_belief_distributions: List of arrays, one per attribute, each
            of shape ``(n_samples, n_ref_values_i)``.
        activation_weights: Shape ``(n_samples, n_rules)``.
        combined_belief_degrees: Shape ``(n_samples, n_consequents)``.
        consequent_values: 1-D array of consequent referential values.
        output: Shape ``(n_samples,)``, scalar numerical outputs.
        consequent_group_sizes: Number of grades belonging to each output, or
            ``None`` for a single output.
        utility_bounds: Optional pair of arrays of shape ``(n_samples,)``
            bounding the output when the assessment is incomplete. Equal to each
            other, and to ``output``, when it is complete.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_belief_distributions: list[np.ndarray]
    activation_weights: np.ndarray
    combined_belief_degrees: np.ndarray
    consequent_values: np.ndarray
    output: np.ndarray
    utility_bounds: tuple[np.ndarray, np.ndarray] | None = None
    consequent_group_sizes: tuple[int, ...] | None = None

    @property
    def n_outputs(self) -> int:
        """Return how many consequent attributes were predicted."""
        if self.consequent_group_sizes is None:
            return 1
        return len(self.consequent_group_sizes)

    @property
    def consequent_slices(self) -> list[slice]:
        """Return the column range of each output within the concatenation."""
        sizes = self.consequent_group_sizes or (self.combined_belief_degrees.shape[1],)
        slices, start = [], 0
        for size in sizes:
            slices.append(slice(start, start + size))
            start += size
        return slices

    @property
    def ignorance(self) -> np.ndarray:
        """Return the belief each combined assessment leaves unassigned.

        The ``beta_H`` of Yang and Xu (2002), Eq. 25b: the extent to which the
        result is incomplete, which arises when the rules that fired were
        themselves incomplete about their consequents.

        Returns:
            Shape ``(n_samples,)`` for a single output, ``(n_samples,
            n_outputs)`` otherwise. Zero when every assessment is complete.
        """
        per_output = np.stack(
            [
                np.clip(1.0 - self.combined_belief_degrees[:, b].sum(axis=1), 0.0, 1.0)
                for b in self.consequent_slices
            ],
            axis=1,
        )
        return per_output[:, 0] if self.n_outputs == 1 else per_output

    @property
    def is_complete(self) -> bool:
        """Return whether every combined assessment assigns all of its belief."""
        return bool(np.allclose(self.ignorance, 0.0, atol=1e-6))

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

    def explain(
        self,
        sample_idx: int = 0,
        top_k: int = 3,
        rule_base: "RuleBase | None" = None,
        attribute_names: list[str] | None = None,
        consequent_name: str | None = None,
        threshold: float = 0.01,
    ) -> str:
        """Return a human-readable explanation of a single prediction.

        Args:
            sample_idx: Which sample in the batch to explain.
            top_k: Number of top-activated rules to show.
            rule_base: The ``RuleBase`` used for the prediction. If
                provided, rule descriptions include antecedent values.
                If ``None``, rules are identified by index only.
            attribute_names: Passed through to ``RuleBase.describe_rule``.
            consequent_name: Passed through to ``RuleBase.describe_rule``.
            threshold: Minimum activation weight or belief degree to
                display; values below this are omitted for clarity.
        """
        lines: list[str] = []
        s = sample_idx

        # Scalar output, one line per output attribute
        if self.n_outputs == 1:
            lines.append(f"Prediction: {float(self.output[s]):.4g}")
        else:
            values = ", ".join(f"y{o + 1}={float(v):.4g}" for o, v in enumerate(self.output[s]))
            lines.append(f"Prediction: {values}")
        lines.append("")

        # Top activated rules
        w = self.activation_weights[s]
        top_indices = np.argsort(-w)[:top_k]
        lines.append("Top activated rules:")
        for k_idx in top_indices:
            wk = float(w[k_idx])
            if wk < threshold:
                continue
            if rule_base is not None:
                # Build a compact rule description (beliefs only, no weight)
                bd = rule_base.belief_degrees[int(k_idx)]
                crv = rule_base.consequent_referential_values
                # With several outputs each block is shown on its own, because
                # the grades of one objective say nothing about another's.
                blocks = rule_base.consequent_slices
                block_strs = []
                for block in blocks:
                    entries = ", ".join(
                        f"{float(crv[n]):.4g}: {bd[n]:.3f}"
                        for n in range(block.start, block.stop)
                        if bd[n] >= threshold
                    )
                    block_strs.append("{" + entries + "}")
                belief_str = " ".join(block_strs) if len(blocks) > 1 else block_strs[0][1:-1]

                if attribute_names is None:
                    attr_names = [f"x{i + 1}" for i in range(rule_base.n_attributes)]
                else:
                    attr_names = attribute_names
                ante_parts = []
                for i in range(rule_base.n_attributes):
                    idx = int(rule_base.rule_antecedent_indices[k_idx, i])
                    val = float(rule_base.precedent_referential_values[i][idx])
                    ante_parts.append(f"{attr_names[i]}={val:.4g}")
                ante_str = ", ".join(ante_parts)
                if rule_base.n_outputs > 1:
                    lines.append(f"  Rule {int(k_idx)} (w={wk:.4f}, {ante_str}): {belief_str}")
                else:
                    lines.append(f"  Rule {int(k_idx)} (w={wk:.4f}, {ante_str}): {{{belief_str}}}")
            else:
                lines.append(f"  Rule {int(k_idx)} (w={wk:.4f})")

        # Combined belief distribution
        lines.append("")
        lines.append("Combined belief distribution:")
        crv = self.consequent_values
        beta = self.combined_belief_degrees[s]
        blocks = self.consequent_slices
        for o, block in enumerate(blocks):
            parts = []
            for n in range(block.start, block.stop):
                if beta[n] < threshold:
                    continue
                parts.append(f"{float(crv[n]):.4g}: {beta[n]:.3f}")
            prefix = "  " if len(blocks) == 1 else f"  y{o + 1}: "
            lines.append(f"{prefix}{{{', '.join(parts)}}}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary of the inference result.

        Numpy arrays are converted to nested Python lists.
        """
        record = {
            "input_belief_distributions": [a.tolist() for a in self.input_belief_distributions],
            "activation_weights": self.activation_weights.tolist(),
            "combined_belief_degrees": self.combined_belief_degrees.tolist(),
            "consequent_values": self.consequent_values.tolist(),
            "output": self.output.tolist(),
            "ignorance": self.ignorance.tolist(),
        }
        if self.utility_bounds is not None:
            record["utility_bounds"] = [bound.tolist() for bound in self.utility_bounds]
        return record
