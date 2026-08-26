"""Main BRB model class with an sklearn-compatible interface.

Provides the ``BRBModel`` class which supports fitting, predicting, and
inspecting a Belief Rule-Based inference system.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution, minimize

from desdeo_brb.inference import (
    _utilities_per_block,
    compute_activation_weights,
    compute_combined_belief_degrees,
    compute_output,
    compute_utility_bounds,
    input_transform,
)
from desdeo_brb.models import InferenceResult, RuleBase
from desdeo_brb.utils import build_rule_antecedent_indices, pad_referential_values

# Ignorance given to a complete rule base when it is trained with incomplete
# rows allowed. Small enough to leave predictions alone, large enough that the
# optimizer has a gradient to follow if the data prefers a vaguer rule.
_INCOMPLETE_SEED = 1e-2


class BRBModel:
    """A trainable Belief Rule-Based inference model.

    Implements an sklearn-compatible interface (``fit``, ``predict``, ``score``,
    ``get_params``, ``set_params``) for building and using BRB systems.

    Args:
        precedent_referential_values: List of 1D sorted arrays, one per
            attribute.
        consequent_referential_values: 1D sorted array of consequent values.
        rule_base: Optional pre-configured RuleBase. If ``None``, a default
            one is constructed from the referential values.
        utility_fn: Optional utility function applied to consequent values
            before computing the scalar output.
        initial_rule_fn: Optional callable mapping a 1D array of antecedent
            values to a scalar. Used to compute initial belief degrees when
            ``rule_base`` is ``None``.
    """

    def __init__(
        self,
        precedent_referential_values: list[np.ndarray],
        consequent_referential_values: np.ndarray | list[np.ndarray],
        rule_base: RuleBase | None = None,
        utility_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        initial_rule_fn: Callable[[np.ndarray], float] | None = None,
        backend: str = "numpy",
    ) -> None:
        if backend not in ("numpy", "jax"):
            raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}")
        if backend == "jax":
            from desdeo_brb.jax_backend import JAX_AVAILABLE

            if not JAX_AVAILABLE:
                raise ImportError("Install JAX: pip install desdeo-brb[jax]")
        self._backend = backend

        self._precedent_referential_values = [
            np.asarray(rv, dtype=float) for rv in precedent_referential_values
        ]
        # A list of arrays declares one output per entry, each with its own
        # grades; a plain array is the single-output case. Both end up stored
        # concatenated, with the group sizes saying where each output begins.
        if isinstance(consequent_referential_values, (list, tuple)):
            groups = [np.asarray(rv, dtype=float) for rv in consequent_referential_values]
            self._consequent_group_sizes: tuple[int, ...] | None = tuple(len(rv) for rv in groups)
            self._consequent_referential_values = np.concatenate(groups)
        else:
            self._consequent_group_sizes = None
            self._consequent_referential_values = np.asarray(
                consequent_referential_values, dtype=float
            )
        self._utility_fn = utility_fn
        self._allow_incomplete = False
        self._scale_outputs = True
        self._ref_value_lengths = [len(rv) for rv in self._precedent_referential_values]

        if rule_base is not None:
            self.rule_base = rule_base
        else:
            self.rule_base = self._build_default_rule_base(initial_rule_fn)

    def _build_default_rule_base(
        self, initial_rule_fn: Callable[[np.ndarray], float] | None = None
    ) -> RuleBase:
        """Construct a default RuleBase from the referential values."""
        rule_antecedent_indices = build_rule_antecedent_indices(self._precedent_referential_values)
        n_rules = len(rule_antecedent_indices)
        n_consequents = len(self._consequent_referential_values)
        n_attributes = len(self._precedent_referential_values)

        rule_weights = np.full(n_rules, 1.0 / n_rules)
        attribute_weights = np.ones((n_rules, n_attributes))

        if initial_rule_fn is not None:
            belief_degrees = self._beliefs_from_fn(initial_rule_fn, rule_antecedent_indices)
        else:
            # Uniform within each output's own block, so every block sums to one.
            sizes = self._consequent_group_sizes or (n_consequents,)
            belief_degrees = np.concatenate(
                [np.full((n_rules, size), 1.0 / size) for size in sizes], axis=1
            )

        return RuleBase(
            precedent_referential_values=self._precedent_referential_values,
            consequent_referential_values=self._consequent_referential_values,
            consequent_group_sizes=self._consequent_group_sizes,
            belief_degrees=belief_degrees,
            rule_weights=rule_weights,
            attribute_weights=attribute_weights,
            rule_antecedent_indices=rule_antecedent_indices,
        )

    def _beliefs_from_fn(
        self,
        fn: Callable[[np.ndarray], float | np.ndarray],
        rule_antecedent_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute initial belief degrees from a function over antecedent values.

        With several outputs the function returns one value per output and each
        is interpolated within its own grades, because the grades of one
        objective say nothing about another's.
        """
        crv = self._consequent_referential_values
        sizes = self._consequent_group_sizes or (len(crv),)
        n_outputs = len(sizes)
        n_rules = len(rule_antecedent_indices)
        belief_degrees = np.zeros((n_rules, len(crv)))

        blocks, start = [], 0
        for size in sizes:
            blocks.append(slice(start, start + size))
            start += size

        for k in range(n_rules):
            # Build the antecedent value vector for rule k
            x_k = np.array(
                [
                    self._precedent_referential_values[i][rule_antecedent_indices[k, i]]
                    for i in range(len(self._precedent_referential_values))
                ]
            )
            y_k = np.atleast_1d(np.asarray(fn(x_k), dtype=float))
            if y_k.size != n_outputs:
                raise ValueError(
                    f"initial_rule_fn must return {n_outputs} value(s), one per output, "
                    f"got {y_k.size}"
                )

            for o, block in enumerate(blocks):
                values = crv[block]
                offset = block.start
                y = float(y_k[o])

                # Distribute belief via linear interpolation on consequent values
                if y <= values[0]:
                    belief_degrees[k, offset] = 1.0
                elif y >= values[-1]:
                    belief_degrees[k, offset + len(values) - 1] = 1.0
                else:
                    j = np.searchsorted(values, y, side="right") - 1
                    j = min(j, len(values) - 2)
                    frac = (y - values[j]) / (values[j + 1] - values[j])
                    belief_degrees[k, offset + j] = 1.0 - frac
                    belief_degrees[k, offset + j + 1] = frac

        return belief_degrees

    def _consequent_utilities(self) -> np.ndarray:
        """Return the consequent values with the utility function applied.

        The JAX and Pyomo paths cannot call an arbitrary Python callable, so
        they consume utilities computed here instead of raw referential values.
        """
        rb = self.rule_base
        crv = rb.consequent_referential_values
        if self._utility_fn is None:
            return crv
        return np.concatenate(_utilities_per_block(crv, rb.consequent_slices, self._utility_fn))

    def _output_scales(self) -> np.ndarray:
        """Return the divisor applied to each output's residual before squaring.

        Objectives carry their own units, so an output whose grades span 100 to
        400 contributes roughly ninety thousand times the squared error of one
        spanning 0 to 1, and a plain sum would fit the first and ignore the
        second. Dividing each residual by the span of that output's own grades
        makes the terms dimensionless and comparable.

        This is a fixed weighting chosen from the rule base rather than from the
        data. Yang et al. (2007), Eqs. (19c) to (22), instead normalise by the
        ideal and the payoff-table worst of each objective and minimise the
        largest scaled error, which yields an efficient solution but costs one
        extra training run per output and needs a constrained optimizer.

        A single-output rule base gets a divisor of one, so its loss and its
        score are exactly what they were before.
        """
        rule_base = self.rule_base
        if rule_base.n_outputs == 1 or not self._scale_outputs:
            return np.ones(rule_base.n_outputs)

        utilities = self._consequent_utilities()
        spans = []
        for block in rule_base.consequent_slices:
            values = utilities[block]
            span = float(values.max() - values.min())
            # A degenerate output with one grade, or all grades equal, cannot be
            # scaled by its own span; leave it alone rather than divide by zero.
            spans.append(span if span > 0 else 1.0)
        return np.asarray(spans, dtype=float)

    def _mean_squared_error(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error over samples and outputs, each output scaled."""
        residual = (np.asarray(y) - y_pred) / self._output_scales()
        return float(np.mean(residual**2))

    def _resolve_allow_incomplete(self, allow_incomplete: bool | None) -> bool:
        """Decide whether training may leave belief rows summing to under one.

        ``None`` means follow the rule base being trained. Forcing completeness
        on a rule base that arrived incomplete would silently discard the
        ignorance an expert had deliberately expressed, and forcing it on one
        that arrived complete would invent ignorance nobody asked for.
        """
        if allow_incomplete is None:
            return not self.rule_base.is_complete
        return bool(allow_incomplete)

    def _check_target_shape(self, y: np.ndarray) -> None:
        """Reject a target array whose shape does not match the outputs.

        Without this a single-column target against a multi-output rule base
        would broadcast into something silently meaningless rather than fail.
        """
        n_outputs = self.rule_base.n_outputs
        y = np.asarray(y)
        if n_outputs == 1:
            if y.ndim != 1 and not (y.ndim == 2 and y.shape[1] == 1):
                raise ValueError(
                    f"y must have shape (n_samples,) for a single output, got {y.shape}"
                )
        elif y.ndim != 2 or y.shape[1] != n_outputs:
            raise ValueError(
                f"y must have shape (n_samples, {n_outputs}) for a rule base with "
                f"{n_outputs} outputs, got {y.shape}"
            )

    def _append_belief_constraint(self, constraints: list[dict], start: int, end: int) -> None:
        """Constrain one rule's belief block for one output.

        An equality when the trained rule base must come out complete, and a
        cap otherwise: Yang et al. (2007) impose the sum-to-one equality only
        when completeness is wanted, and their constraint 12b caps the sum at
        one in every case.
        """
        if self._allow_incomplete:

            def bd_slack(flat: np.ndarray, s: int = start, e: int = end) -> float:
                return float(1.0 - flat[s:e].sum())

            constraints.append({"type": "ineq", "fun": bd_slack})
        else:

            def bd_constraint(flat: np.ndarray, s: int = start, e: int = end) -> float:
                return float(flat[s:e].sum() - 1.0)

            constraints.append({"type": "eq", "fun": bd_constraint})

    def _project_beliefs(self, bd: np.ndarray) -> np.ndarray:
        """Clip belief degrees and bring each output's block within its budget.

        Each output is normalised on its own: what a rule believes about one
        objective places no constraint on what it believes about another.
        """
        bd = np.clip(bd, 0.0, 1.0)
        for block in self.rule_base.consequent_slices:
            sums = bd[:, block].sum(axis=1, keepdims=True)
            if self._allow_incomplete:
                # Only over-assignment violates the constraint. A block summing
                # to less than one is a legitimate incomplete rule.
                bd[:, block] /= np.where(sums > 1.0, sums, 1.0)
            else:
                bd[:, block] /= np.where(sums > 0, sums, 1.0)
        return bd

    def predict(self, X: np.ndarray) -> InferenceResult:
        """Run the full inference pipeline on input data.

        Args:
            X: Input array of shape ``(n_samples, n_attributes)``.

        Returns:
            An :class:`InferenceResult` with all intermediate and final values.
        """
        if self._backend == "jax":
            return self._predict_jax(X)
        return self._predict_numpy(X)

    def _predict_numpy(self, X: np.ndarray) -> InferenceResult:
        """NumPy inference path."""
        rb = self.rule_base

        alphas = input_transform(X, rb.precedent_referential_values)
        weights = compute_activation_weights(
            alphas, rb.rule_antecedent_indices, rb.rule_weights, rb.attribute_weights
        )
        combined = compute_combined_belief_degrees(
            rb.belief_degrees, weights, rb.consequent_group_sizes
        )
        output = compute_output(
            combined,
            rb.consequent_referential_values,
            self._utility_fn,
            rb.consequent_group_sizes,
        )

        return InferenceResult(
            input_belief_distributions=alphas,
            activation_weights=weights,
            combined_belief_degrees=combined,
            consequent_values=rb.consequent_referential_values,
            output=output,
            utility_bounds=compute_utility_bounds(
                combined,
                rb.consequent_referential_values,
                self._utility_fn,
                rb.consequent_group_sizes,
            ),
            consequent_group_sizes=rb.consequent_group_sizes,
        )

    def _predict_jax(self, X: np.ndarray) -> InferenceResult:
        """JAX inference path. Converts results back to NumPy for the public API."""
        import jax.numpy as jnp

        from desdeo_brb.jax_backend import (
            compute_activation_weights_jax,
            compute_combined_belief_degrees_jax,
            compute_output_jax,
            input_transform_jax,
        )

        rb = self.rule_base
        padded_rv, rv_lengths = pad_referential_values(rb.precedent_referential_values)
        rv_lengths_tuple = tuple(int(x) for x in rv_lengths)

        X_jax = jnp.asarray(X)
        padded_rv_jax = jnp.asarray(padded_rv)

        alphas_3d = input_transform_jax(X_jax, padded_rv_jax, rv_lengths_tuple)
        weights = compute_activation_weights_jax(
            alphas_3d,
            jnp.asarray(rb.rule_antecedent_indices),
            jnp.asarray(rb.rule_weights),
            jnp.asarray(rb.attribute_weights),
        )
        combined = compute_combined_belief_degrees_jax(
            jnp.asarray(rb.belief_degrees), weights, rb.consequent_group_sizes
        )
        # The utility is applied here rather than inside the traced function, so
        # that an arbitrary Python callable never has to be JAX traceable.
        output = compute_output_jax(
            combined, jnp.asarray(self._consequent_utilities()), rb.consequent_group_sizes
        )

        # Convert back to numpy; split padded alphas into list
        alphas_list = [
            np.asarray(alphas_3d[:, i, : int(rv_lengths[i])]) for i in range(len(rv_lengths))
        ]

        return InferenceResult(
            input_belief_distributions=alphas_list,
            activation_weights=np.asarray(weights),
            combined_belief_degrees=np.asarray(combined),
            consequent_values=rb.consequent_referential_values,
            output=np.asarray(output),
            utility_bounds=compute_utility_bounds(
                np.asarray(combined),
                rb.consequent_referential_values,
                self._utility_fn,
                rb.consequent_group_sizes,
            ),
            consequent_group_sizes=rb.consequent_group_sizes,
        )

    def predict_values(self, X: np.ndarray) -> np.ndarray:
        """Convenience method returning only the scalar outputs.

        Args:
            X: Input array of shape ``(n_samples, n_attributes)``.

        Returns:
            1-D array of shape ``(n_samples,)``.
        """
        return self.predict(X).output

    def explain(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        top_k: int = 3,
        attribute_names: list[str] | None = None,
        consequent_name: str | None = None,
        threshold: float = 0.01,
    ) -> str:
        """Predict on *X* and return a human-readable explanation.

        Convenience wrapper that calls ``predict(X)`` and then
        ``InferenceResult.explain()`` with this model's rule base.

        Args:
            X: Input array of shape ``(n_samples, n_attributes)``.
            sample_idx: Which sample in the batch to explain.
            top_k: Number of top-activated rules to show.
            attribute_names: Display names for each attribute.
            consequent_name: Display name for the consequent.
            threshold: Minimum weight/belief to display.
        """
        result = self.predict(X)
        return result.explain(
            sample_idx=sample_idx,
            top_k=top_k,
            rule_base=self.rule_base,
            attribute_names=attribute_names,
            consequent_name=consequent_name,
            threshold=threshold,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool = True,
        fix_endpoint_beliefs: bool = False,
        normalize_rule_weights: bool = True,
        allow_incomplete: bool | None = None,
        scale_outputs: bool = True,
        method: str | None = None,
        optimizer_options: dict | None = None,
        n_restarts: int = 1,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """Train the model by minimizing MSE.

        For the NumPy backend, supported methods are ``"SLSQP"`` (default)
        and ``"trust-constr"``. For the JAX backend, the only supported
        method is ``"L-BFGS-B"`` (default), which uses exact ``jax.grad``
        gradients.

        Args:
            X: Training inputs, shape ``(n_samples, n_attributes)``.
            y: Target values, shape ``(n_samples,)``.
            fix_endpoints: If ``True``, fix the first and last precedent
                referential values (endpoints of each attribute's range).
            fix_endpoint_beliefs: If ``True``, also fix the belief degrees
                for rules at the boundary referential values during training.
                Use this when the initial beliefs at the domain boundaries
                are known to be correct (e.g., from ``initial_rule_fn`` or
                verified expert knowledge) to prevent the optimizer from
                distorting endpoint predictions.
            allow_incomplete: Whether belief rows may sum to less than one
                during training, the shortfall being the rule's ignorance.
                ``None`` (default) follows the rule base: a rule base that
                arrives incomplete keeps that freedom, one that arrives
                complete stays complete. Yang et al. (2007) impose the
                sum-to-one equality only when a complete trained rule base is
                wanted; their constraint 12b otherwise caps the sum at one.
            scale_outputs: If ``True`` (default), divide each output's
                residual by the span of that output's grades before squaring,
                so objectives on wider scales do not dominate the fit. Has no
                effect on a single-output rule base. Set ``False`` to sum the
                raw squared errors instead.
            normalize_rule_weights: If ``True`` (default), constrain rule
                weights to sum to 1 during optimization. If ``False``, only
                bound each rule weight individually to [0, 1]; the optimizer
                may pick any scaling and the final stored weights are
                renormalized. Removing the sum constraint can give SLSQP /
                trust-constr a less coupled search landscape.
            method: scipy.optimize.minimize method. NumPy backend supports
                ``"SLSQP"`` and ``"trust-constr"``; JAX backend supports
                ``"L-BFGS-B"``. If ``None``, uses the backend default.
            optimizer_options: Options dict passed to ``scipy.optimize.minimize``
                as the ``options`` argument. Merged with sensible per-method
                defaults; user values override the defaults.
            n_restarts: Number of optimization runs (default 1). When > 1,
                the first run uses the unperturbed initial parameters and
                subsequent runs perturb the initial parameters with seeded
                random noise. The final model is the best of all runs as
                measured by training MSE. Multi-start is critical for
                escaping bad local minima.
            verbose: If ``True``, print optimizer progress.
            **minimize_kwargs: Extra keyword arguments forwarded to
                ``scipy.optimize.minimize``.

        Returns:
            self
        """
        # Validate method against backend
        if self._backend == "numpy":
            if method is None:
                method = "SLSQP"
            valid_numpy = ("SLSQP", "trust-constr", "ipopt", "DE", "DE+SLSQP")
            if method not in valid_numpy:
                raise ValueError(f"NumPy backend supports methods {valid_numpy}, got {method!r}")
            if method == "ipopt":
                try:
                    from desdeo_brb.pyomo_backend import PYOMO_AVAILABLE
                except ImportError:
                    PYOMO_AVAILABLE = False
                if not PYOMO_AVAILABLE:
                    raise ImportError(
                        "Install Pyomo for IPOPT support: pip install desdeo-brb[pyomo]"
                    )
        else:  # jax
            if method is None:
                method = "L-BFGS-B"
            if method != "L-BFGS-B":
                raise ValueError(f"JAX backend supports method='L-BFGS-B' only, got {method!r}")

        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

        self._check_target_shape(y)

        # Resolved once here rather than threaded through the ten helpers that
        # need it, all of which read self._allow_incomplete.
        self._allow_incomplete = self._resolve_allow_incomplete(allow_incomplete)
        self._scale_outputs = scale_outputs

        def _run_one(verbose_inner: bool = False) -> None:
            if self._backend == "jax":
                self._fit_jax(
                    X,
                    y,
                    fix_endpoints,
                    fix_endpoint_beliefs,
                    method,
                    optimizer_options,
                    verbose=verbose_inner,
                    normalize_rule_weights=normalize_rule_weights,
                    **minimize_kwargs,
                )
            elif method == "ipopt":
                self._fit_pyomo(
                    X,
                    y,
                    fix_endpoints,
                    fix_endpoint_beliefs,
                    normalize_rule_weights,
                    optimizer_options,
                    verbose_inner,
                )
            elif method == "DE":
                self._fit_de(
                    X,
                    y,
                    fix_endpoints,
                    fix_endpoint_beliefs,
                    normalize_rule_weights,
                    optimizer_options,
                    verbose_inner,
                )
            elif method == "DE+SLSQP":
                self._fit_de_slsqp(
                    X,
                    y,
                    fix_endpoints,
                    fix_endpoint_beliefs,
                    normalize_rule_weights,
                    optimizer_options,
                    verbose_inner,
                )
            else:
                self._fit_numpy(
                    X,
                    y,
                    fix_endpoints,
                    fix_endpoint_beliefs,
                    method,
                    optimizer_options,
                    verbose=verbose_inner,
                    normalize_rule_weights=normalize_rule_weights,
                    **minimize_kwargs,
                )

        if n_restarts == 1:
            # Single run, surface optimizer verbosity through the inner call
            _run_one(verbose_inner=verbose)
            return self

        # Multi-start: snapshot the initial parameters, run multiple times,
        # keep the result with the lowest training MSE.
        initial_flat = self._flatten_params()

        best_mse = float("inf")
        best_rule_base: RuleBase | None = None
        best_restart_index = 0

        for restart in range(n_restarts):
            if restart == 0:
                self.rule_base = self._unflatten_params(initial_flat)
            else:
                rng = np.random.default_rng(restart)
                perturbed_flat = self._perturb_params(
                    initial_flat, rng, fix_endpoints, normalize_rule_weights
                )
                self.rule_base = self._unflatten_params(perturbed_flat)

            _run_one(verbose_inner=verbose)

            y_pred = self.predict_values(X)
            mse = self._mean_squared_error(y, y_pred)
            improved = mse < best_mse
            if improved:
                best_mse = mse
                best_rule_base = self.rule_base
                best_restart_index = restart + 1

            if verbose:
                marker = " (best)" if improved else ""
                print(f"Restart {restart + 1}/{n_restarts}: MSE = {mse:.5f}{marker}")

        if best_rule_base is not None:
            self.rule_base = best_rule_base

        if verbose:
            print(f"Best result: restart {best_restart_index}, MSE = {best_mse:.5f}")

        return self

    def _fit_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool = True,
        fix_endpoint_beliefs: bool = True,
        method: str = "SLSQP",
        optimizer_options: dict | None = None,
        verbose: bool = False,
        normalize_rule_weights: bool = True,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """NumPy training path using SLSQP or trust-constr."""
        x0 = self._flatten_params()
        bounds = self._build_bounds(fix_endpoints, fix_endpoint_beliefs)

        if method == "SLSQP":
            constraints = self._build_constraints(fix_endpoint_beliefs, normalize_rule_weights)
            default_options = {
                "maxiter": 1000,
                "ftol": 1e-9,
                "disp": verbose,
            }
        else:  # trust-constr
            constraints = self._build_trust_constr_constraints(
                len(x0), fix_endpoint_beliefs, normalize_rule_weights
            )
            default_options = {
                "maxiter": 2000,
                "gtol": 1e-9,
                "verbose": 2 if verbose else 0,
            }

        # Merge user options with defaults (user wins).
        # Also accept legacy ``options`` kwarg via minimize_kwargs.
        legacy_options = minimize_kwargs.pop("options", {}) or {}
        options = {**default_options, **legacy_options, **(optimizer_options or {})}

        def objective(flat: np.ndarray) -> float:
            return self._mse_objective(flat, X, y)

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options,
            **minimize_kwargs,
        )

        self.rule_base = self._unflatten_params(self._normalize_flat(result.x))
        return self

    def _fit_jax(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool = True,
        fix_endpoint_beliefs: bool = True,
        method: str = "L-BFGS-B",
        optimizer_options: dict | None = None,
        verbose: bool = False,
        normalize_rule_weights: bool = True,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """JAX training path using L-BFGS-B with exact gradients."""
        import jax
        import jax.numpy as jnp

        from desdeo_brb.jax_backend import full_inference_jax_unconstrained

        rb = self.rule_base
        X_jax = jnp.asarray(X)
        y_jax = jnp.asarray(y)
        crv_jax = jnp.asarray(self._consequent_utilities())
        rai_jax = jnp.asarray(rb.rule_antecedent_indices)
        rv_lengths_tuple = tuple(self._ref_value_lengths)

        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes
        norm_rw = normalize_rule_weights  # capture as local for closure
        allow_inc = self._allow_incomplete  # ditto
        group_sizes = rb.consequent_group_sizes
        scales_jax = jnp.asarray(self._output_scales())

        @jax.jit
        def mse_loss(flat_params):
            y_pred = full_inference_jax_unconstrained(
                flat_params,
                X_jax,
                crv_jax,
                rai_jax,
                n_rules,
                n_consequents,
                n_attributes,
                rv_lengths_tuple,
                normalize_rule_weights=norm_rw,
                allow_incomplete=allow_inc,
                group_sizes=group_sizes,
            )
            return jnp.mean(((y_jax - y_pred) / scales_jax) ** 2)

        grad_fn = jax.jit(jax.grad(mse_loss))

        def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
            flat_jax = jnp.asarray(flat)
            loss = float(mse_loss(flat_jax))
            grad = np.asarray(grad_fn(flat_jax))
            return loss, grad

        # Transform initial params to unconstrained space matching the
        # softmax/softplus reparameterization in full_inference_jax.
        x0 = self._flatten_params_unconstrained(normalize_rule_weights)
        bounds = self._build_bounds_jax(fix_endpoints, fix_endpoint_beliefs)

        default_options: dict = {
            "maxiter": 2000,
            "ftol": 1e-12,
        }
        legacy_options = minimize_kwargs.pop("options", {}) or {}
        options = {**default_options, **legacy_options, **(optimizer_options or {})}

        # L-BFGS-B's built-in disp/iprint is deprecated in scipy 1.17+.
        # Use a callback for per-iteration output instead.
        callback = None
        if verbose:
            _iter_count = [0]

            def _verbose_callback(xk):
                _iter_count[0] += 1
                loss = float(mse_loss(jnp.asarray(xk)))
                print(f"  JAX L-BFGS-B iter {_iter_count[0]}: loss={loss:.6f}")

            callback = _verbose_callback

        result = minimize(
            objective_and_grad,
            x0,
            method=method,
            jac=True,
            bounds=bounds,
            options=options,
            callback=callback,
            **minimize_kwargs,
        )

        self.rule_base = self._unflatten_from_unconstrained(
            result.x, normalize_rule_weights=normalize_rule_weights
        )
        return self

    def _unflatten_from_unconstrained(
        self, flat: np.ndarray, normalize_rule_weights: bool = True
    ) -> RuleBase:
        """Convert unconstrained JAX optimizer output to a validated RuleBase.

        Applies softmax to belief degree rows, softmax or sigmoid to rule
        weights, softplus to attribute weights, and sorts referential values.
        Rule weights are renormalized to sum to 1 in the final stored
        RuleBase regardless of the optimization parameterization (the
        activation weight formula is scale-invariant in rule weights).
        """
        from scipy.special import expit as sp_sigmoid
        from scipy.special import softmax as sp_softmax

        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes
        idx = 0

        bd_size = n_rules * n_consequents
        bd_raw = flat[idx : idx + bd_size].reshape(n_rules, n_consequents)
        # Must match full_inference_jax_unconstrained exactly, block by block:
        # each output is normalised on its own, and when incomplete rows are
        # allowed the softmax runs over that block's grades plus one implicit
        # grade at logit zero standing for ignorance, which is then dropped.
        bd_parts = []
        for block in rb.consequent_slices:
            raw = bd_raw[:, block]
            if self._allow_incomplete:
                raw = np.concatenate([raw, np.zeros((n_rules, 1))], axis=1)
                bd_parts.append(sp_softmax(raw, axis=1)[:, : block.stop - block.start])
            else:
                bd_parts.append(sp_softmax(raw, axis=1))
        belief_degrees = np.concatenate(bd_parts, axis=1)
        idx += bd_size

        rw_raw = flat[idx : idx + n_rules]
        if normalize_rule_weights:
            rule_weights = sp_softmax(rw_raw)
        else:
            rw_sigmoid = sp_sigmoid(rw_raw)
            rw_sum = rw_sigmoid.sum()
            rule_weights = rw_sigmoid / rw_sum if rw_sum > 0 else np.full(n_rules, 1.0 / n_rules)
        idx += n_rules

        aw_size = n_rules * n_attributes
        aw_raw = flat[idx : idx + aw_size].reshape(n_rules, n_attributes)
        attribute_weights = np.log1p(np.exp(aw_raw))  # softplus
        idx += aw_size

        precedent_referential_values = []
        for length in self._ref_value_lengths:
            rv = np.sort(flat[idx : idx + length].copy())
            precedent_referential_values.append(rv)
            idx += length

        return RuleBase(
            precedent_referential_values=precedent_referential_values,
            consequent_referential_values=rb.consequent_referential_values,
            consequent_group_sizes=rb.consequent_group_sizes,
            belief_degrees=belief_degrees,
            rule_weights=rule_weights,
            attribute_weights=attribute_weights,
            rule_antecedent_indices=rb.rule_antecedent_indices,
        )

    def _fit_pyomo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool,
        fix_endpoint_beliefs: bool,
        normalize_rule_weights: bool,
        optimizer_options: dict | None,
        verbose: bool,
    ) -> None:
        """Train via the Pyomo backend with the IPOPT interior-point solver.

        Builds a Pyomo ConcreteModel, solves it with IPOPT, and pulls the
        solved parameter values back into ``self.rule_base``.

        The Pyomo model uses ``optimize_referential_values=False`` so the
        referential values are fixed during this training run; the
        symbolic input-transform (Path B in pyomo_backend) creates an
        expression tree large enough that IPOPT becomes impractically
        slow on non-trivial problems. Users who want to optimize
        referential values can call ``build_pyomo_brb_model(...,
        optimize_referential_values=True)`` directly and solve it
        themselves.
        """
        try:
            import pyomo.environ as pyo

            from desdeo_brb.pyomo_backend import build_pyomo_brb_model
        except ImportError as exc:  # pragma: no cover - guarded above
            raise ImportError(
                "Install Pyomo for IPOPT support: pip install desdeo-brb[pyomo]"
            ) from exc

        pyomo_model = build_pyomo_brb_model(
            self,
            X,
            y,
            fix_endpoints=fix_endpoints,
            fix_endpoint_beliefs=fix_endpoint_beliefs,
            normalize_rule_weights=normalize_rule_weights,
            optimize_referential_values=False,
        )

        solver = pyo.SolverFactory("ipopt")
        if not solver.available():
            raise RuntimeError(
                "IPOPT solver binary not found. Install IPOPT (e.g. "
                "'apt install coinor-libipopt-dev' or "
                "'conda install -c conda-forge ipopt')."
            )

        default_options = {
            "max_iter": 3000,
            "tol": 1e-8,
            "print_level": 5 if verbose else 0,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
            # Accept "good enough" before numerical breakdown
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
            "acceptable_dual_inf_tol": 1e10,  # don't reject on dual infeasibility
            "acceptable_compl_inf_tol": 1e-2,
            "acceptable_obj_change_tol": 1e-3,  # stop if objective barely changing
        }
        for k, v in {**default_options, **(optimizer_options or {})}.items():
            solver.options[k] = v

        # load_solutions=False so Pyomo doesn't crash when IPOPT errors
        # but still found an intermediate solution.
        result = solver.solve(pyomo_model, tee=verbose, load_solutions=False)

        status = str(result.solver.status).lower()
        term = str(result.solver.termination_condition).lower()

        # Try to load the solution. Pyomo's load_from refuses "error"
        # status, but IPOPT writes its best iterate to the .sol file
        # even on crash. Override the status to "warning" temporarily
        # so load_from accepts it.
        solution_loaded = False
        if len(result.solution) > 0:
            import pyomo.opt as pyopt

            original_status = result.solver.status
            try:
                result.solver.status = pyopt.SolverStatus.warning
                pyomo_model.solutions.load_from(result)
                solution_loaded = True
            except Exception:
                pass
            finally:
                result.solver.status = original_status

        if solution_loaded:
            if verbose and term not in ("optimal", "locallyoptimal", "feasible"):
                print(
                    f"IPOPT terminated with {term} "
                    f"(objective={pyo.value(pyomo_model.obj):.4f}). "
                    f"Extracted partial solution."
                )
            self.update_from_pyomo(pyomo_model)
        else:
            import warnings

            warnings.warn(
                f"IPOPT failed (status={status}, condition={term}). "
                f"No solution found. Model parameters unchanged.",
                stacklevel=3,
            )

    def _fit_de(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool,
        fix_endpoint_beliefs: bool,
        normalize_rule_weights: bool,
        optimizer_options: dict | None,
        verbose: bool,
    ) -> None:
        """Train via Differential Evolution (global optimizer).

        DE runs with box bounds only — no equality constraints during the
        search, because the BRB equality constraints (belief row sums,
        rule weight sums) define a thin manifold that DE's stochastic
        mutations can never hit. The ER inference pipeline accepts
        non-normalized parameters, so DE can explore freely. After DE
        converges, ``_normalize_flat`` projects the solution back onto
        the exact constraint surface.
        """
        bounds_list = self._build_bounds(fix_endpoints, fix_endpoint_beliefs)

        def objective(flat: np.ndarray) -> float:
            normalized = self._normalize_flat(flat)
            self.rule_base = self._unflatten_params(normalized, validate=False)
            y_pred = self.predict_values(X)
            return self._mean_squared_error(y, y_pred)

        default_options: dict[str, Any] = {
            "maxiter": 1000,
            "tol": 1e-8,
            "seed": 42,
            "polish": False,
            "disp": verbose,
            "strategy": "best1bin",
            "popsize": 15,
            "mutation": (0.5, 1.0),
            "recombination": 0.7,
        }
        opts = {**default_options, **(optimizer_options or {})}

        x0 = self._flatten_params()
        result = differential_evolution(objective, bounds=bounds_list, x0=x0, **opts)

        self.rule_base = self._unflatten_params(self._normalize_flat(result.x))

    def _fit_de_slsqp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool,
        fix_endpoint_beliefs: bool,
        normalize_rule_weights: bool,
        optimizer_options: dict | None,
        verbose: bool,
    ) -> None:
        """Two-phase training: DE for global search, SLSQP for local polish.

        ``optimizer_options`` may contain ``"de"`` and ``"slsqp"`` sub-dicts
        to configure each phase independently::

            model.fit(X, y, method="DE+SLSQP", optimizer_options={
                "de": {"maxiter": 300, "seed": 42},
                "slsqp": {"maxiter": 1000, "ftol": 1e-12},
            })

        If no sub-dicts are present, sensible defaults are used for both.
        """
        opts = optimizer_options or {}
        de_opts = opts.get("de", opts if "maxiter" in opts else {})
        slsqp_opts = opts.get("slsqp", {})

        # Phase 1: DE global search
        if verbose:
            print("DE+SLSQP phase 1: Differential Evolution")
        self._fit_de(
            X,
            y,
            fix_endpoints,
            fix_endpoint_beliefs,
            normalize_rule_weights,
            de_opts,
            verbose,
        )

        # Phase 2: SLSQP local polish from DE's solution
        if verbose:
            print("DE+SLSQP phase 2: SLSQP local polish")
        self._fit_numpy(
            X,
            y,
            fix_endpoints,
            fix_endpoint_beliefs,
            method="SLSQP",
            optimizer_options=slsqp_opts or None,
            verbose=verbose,
            normalize_rule_weights=normalize_rule_weights,
        )

    def update_from_pyomo(self, pyomo_model) -> None:
        """Extract solved parameter values from a Pyomo model and update the rule base.

        Reads the variable values for belief degrees, rule weights,
        attribute weights, and referential values from the Pyomo model
        and assembles them into a fresh ``RuleBase`` (with validation).
        Solver-tolerance violations are projected back onto the constraint
        surface (blocks scaled down if they over-assign, and renormalized to
        sum to 1 unless incomplete rules are allowed, attribute weights clipped
        to be non-negative, referential values sorted).

        This method is the inverse of :func:`build_pyomo_brb_model` for
        the parameter-extraction direction. Users who want to optimize a
        custom Pyomo objective on top of the BRB structure can call::

            from desdeo_brb.pyomo_backend import build_pyomo_brb_model
            import pyomo.environ as pyo

            m = build_pyomo_brb_model(brb, X, y)
            m.del_component(m.obj)
            m.obj = pyo.Objective(expr=my_custom_loss(m), sense=pyo.minimize)
            pyo.SolverFactory("ipopt").solve(m)
            brb.update_from_pyomo(m)
        """
        try:
            import pyomo.environ as pyo
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install Pyomo: pip install desdeo-brb[pyomo]") from exc

        n_rules = pyomo_model._brb_n_rules
        n_consequents = pyomo_model._brb_n_consequents
        n_attributes = pyomo_model._brb_n_attributes
        ref_value_lengths = pyomo_model._brb_ref_value_lengths
        rule_antecedent_indices = pyomo_model._brb_rule_antecedent_indices
        consequent_rv = pyomo_model._brb_consequent_referential_values

        # Extract belief degrees and clamp + renormalize per row
        belief_degrees = np.zeros((n_rules, n_consequents))
        for k in range(n_rules):
            for n in range(n_consequents):
                belief_degrees[k, n] = float(pyo.value(pyomo_model.beta[k, n]))
        belief_degrees = self._project_beliefs(belief_degrees)

        # Extract rule weights and renormalize
        rule_weights = np.array([float(pyo.value(pyomo_model.theta[k])) for k in range(n_rules)])
        rule_weights = np.clip(rule_weights, 0.0, 1.0)
        rw_sum = rule_weights.sum()
        if rw_sum > 0:
            rule_weights = rule_weights / rw_sum
        else:
            rule_weights = np.full(n_rules, 1.0 / n_rules)

        # Extract attribute weights and clip to non-negative
        attribute_weights = np.zeros((n_rules, n_attributes))
        for k in range(n_rules):
            for i in range(n_attributes):
                attribute_weights[k, i] = float(pyo.value(pyomo_model.delta[k, i]))
        attribute_weights = np.clip(attribute_weights, 0.0, None)

        # Extract referential values and sort each attribute's values
        precedent_referential_values: list[np.ndarray] = []
        for i in range(n_attributes):
            length = int(ref_value_lengths[i])
            rv = np.array([float(pyo.value(pyomo_model.A[i, j])) for j in range(length)])
            rv = np.sort(rv)
            precedent_referential_values.append(rv)

        new_rule_base = RuleBase(
            precedent_referential_values=precedent_referential_values,
            consequent_referential_values=np.asarray(consequent_rv),
            consequent_group_sizes=getattr(pyomo_model, "_brb_consequent_group_sizes", None),
            belief_degrees=belief_degrees,
            rule_weights=rule_weights,
            attribute_weights=attribute_weights,
            rule_antecedent_indices=np.asarray(rule_antecedent_indices),
        )

        self.rule_base = new_rule_base
        # Keep the model's cached referential value lengths in sync
        self._precedent_referential_values = precedent_referential_values
        self._ref_value_lengths = [len(rv) for rv in precedent_referential_values]

    def fit_custom(
        self,
        loss_fn: Callable[["BRBModel"], float],
        fix_endpoints: bool = True,
        fix_endpoint_beliefs: bool = False,
        normalize_rule_weights: bool = True,
        method: str = "SLSQP",
        optimizer_options: dict | None = None,
        n_restarts: int = 1,
        constraints: list[dict] | None = None,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """Train using a user-supplied loss function.

        The loss function receives the model instance (with updated
        parameters) and must return a scalar loss value. The model's
        parameters are updated internally before each call so the user
        can simply call ``model.predict_values()`` inside the loss.

        Optimization uses scipy with finite differences regardless of the
        model's backend, since the user's loss function is opaque to JAX.
        The structural BRB constraints (belief degree row sums, rule weight
        sum, attribute weight bounds, referential value ordering) are
        always enforced; users may pass additional ``constraints``.

        Args:
            loss_fn: Callable ``(model) -> float`` returning the scalar loss.
            fix_endpoints: If ``True``, fix the first and last precedent
                referential values for each attribute.
            fix_endpoint_beliefs: If ``True``, fix the belief degrees of
                rules at the boundary referential values.
            normalize_rule_weights: If ``True``, constrain rule weights to
                sum to 1 during optimization.
            method: scipy optimizer to use. Supported: ``"SLSQP"`` (default)
                and ``"trust-constr"``.
            optimizer_options: Options dict passed to ``scipy.optimize.minimize``,
                merged with sensible per-method defaults.
            n_restarts: Number of optimization runs from perturbed initial
                points. The best result by ``loss_fn`` value is kept.
            constraints: Additional constraints to add on top of the BRB
                structural constraints. For SLSQP, list of dicts with
                ``"type"`` / ``"fun"`` keys; for trust-constr, list of
                ``LinearConstraint`` / ``NonlinearConstraint`` objects.
            verbose: If ``True``, print per-restart loss.
            **minimize_kwargs: Extra keyword arguments forwarded to
                ``scipy.optimize.minimize``.

        Returns:
            self
        """
        if method not in ("SLSQP", "trust-constr"):
            raise ValueError(
                f"fit_custom supports method='SLSQP' or 'trust-constr', got {method!r}"
            )
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

        def _run_one() -> None:
            self._fit_custom_inner(
                loss_fn,
                fix_endpoints,
                fix_endpoint_beliefs,
                normalize_rule_weights,
                method,
                optimizer_options,
                constraints,
                **minimize_kwargs,
            )

        if n_restarts == 1:
            _run_one()
            return self

        # Multi-start: snapshot initial parameters, run multiple times,
        # keep the rule base with the lowest loss_fn value.
        initial_flat = self._flatten_params()
        best_loss = float("inf")
        best_rule_base: RuleBase | None = None
        best_restart_index = 0

        for restart in range(n_restarts):
            if restart == 0:
                self.rule_base = self._unflatten_params(initial_flat)
            else:
                rng = np.random.default_rng(restart)
                perturbed_flat = self._perturb_params(
                    initial_flat, rng, fix_endpoints, normalize_rule_weights
                )
                self.rule_base = self._unflatten_params(perturbed_flat)

            _run_one()

            loss = float(loss_fn(self))
            improved = loss < best_loss
            if improved:
                best_loss = loss
                best_rule_base = self.rule_base
                best_restart_index = restart + 1

            if verbose:
                marker = " (best)" if improved else ""
                print(f"Restart {restart + 1}/{n_restarts}: loss = {loss:.5f}{marker}")

        if best_rule_base is not None:
            self.rule_base = best_rule_base

        if verbose:
            print(f"Best result: restart {best_restart_index}, loss = {best_loss:.5f}")

        return self

    def _fit_custom_inner(
        self,
        loss_fn: Callable[["BRBModel"], float],
        fix_endpoints: bool,
        fix_endpoint_beliefs: bool,
        normalize_rule_weights: bool,
        method: str,
        optimizer_options: dict | None,
        constraints: list | None,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """Run a single fit_custom optimization (no multistart)."""
        x0 = self._flatten_params()
        bounds = self._build_bounds(fix_endpoints, fix_endpoint_beliefs)

        if method == "SLSQP":
            all_constraints = self._build_constraints(fix_endpoint_beliefs, normalize_rule_weights)
            default_options = {
                "maxiter": 1000,
                "ftol": 1e-9,
                "disp": False,
            }
        else:  # trust-constr
            all_constraints = self._build_trust_constr_constraints(
                len(x0), fix_endpoint_beliefs, normalize_rule_weights
            )
            default_options = {
                "maxiter": 2000,
                "gtol": 1e-9,
                "verbose": 0,
            }

        if constraints is not None:
            all_constraints = list(all_constraints) + list(constraints)

        legacy_options = minimize_kwargs.pop("options", {}) or {}
        options = {**default_options, **legacy_options, **(optimizer_options or {})}

        def objective(flat: np.ndarray) -> float:
            self.rule_base = self._unflatten_params(flat, validate=False)
            return float(loss_fn(self))

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=all_constraints,
            options=options,
            **minimize_kwargs,
        )

        self.rule_base = self._unflatten_params(self._normalize_flat(result.x))
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters (sklearn-compatible).

        Args:
            deep: If ``True``, return nested parameters.

        Returns:
            Dictionary of parameter names to values.
        """
        params: dict[str, Any] = {
            "precedent_referential_values": self._precedent_referential_values,
            "consequent_referential_values": self._consequent_referential_values,
            "utility_fn": self._utility_fn,
            "backend": self._backend,
        }
        if deep:
            params["rule_base"] = self.rule_base
        return params

    def set_params(self, **params: Any) -> "BRBModel":
        """Set model parameters (sklearn-compatible).

        Args:
            **params: Parameter names and values.

        Returns:
            self
        """
        if "precedent_referential_values" in params:
            self._precedent_referential_values = [
                np.asarray(rv, dtype=float) for rv in params["precedent_referential_values"]
            ]
            self._ref_value_lengths = [len(rv) for rv in self._precedent_referential_values]
        if "consequent_referential_values" in params:
            self._consequent_referential_values = np.asarray(
                params["consequent_referential_values"], dtype=float
            )
        if "utility_fn" in params:
            self._utility_fn = params["utility_fn"]
        if "rule_base" in params:
            self.rule_base = params["rule_base"]
        if "backend" in params:
            backend = params["backend"]
            if backend not in ("numpy", "jax"):
                raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}")
            self._backend = backend
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return negative MSE (sklearn convention: higher is better).

        With several outputs each residual is divided by the span of that
        output's grades before squaring, so the score is not dominated by
        whichever objective happens to have the widest units. See
        ``_output_scales``. A single output is unscaled.

        Args:
            X: Input array, shape ``(n_samples, n_attributes)``.
            y: True target values, shape ``(n_samples,)`` for a single output
                and ``(n_samples, n_outputs)`` otherwise.

        Returns:
            Negative mean squared error.
        """
        y_pred = self.predict_values(X)
        return -self._mean_squared_error(y, y_pred)

    @property
    def belief_degrees(self) -> np.ndarray:
        return self.rule_base.belief_degrees

    @property
    def rule_weights(self) -> np.ndarray:
        return self.rule_base.rule_weights

    @property
    def attribute_weights(self) -> np.ndarray:
        return self.rule_base.attribute_weights

    @property
    def precedent_referential_values(self) -> list[np.ndarray]:
        return self.rule_base.precedent_referential_values

    @property
    def consequent_referential_values(self) -> np.ndarray:
        return self.rule_base.consequent_referential_values

    def _flatten_params(self) -> np.ndarray:
        """Concatenate trainable parameters into a single 1-D vector.

        Layout: belief_degrees (flat) | rule_weights | attribute_weights (flat)
                | precedent referential values (concatenated)
        """
        rb = self.rule_base
        parts = [
            rb.belief_degrees.ravel(),
            rb.rule_weights,
            rb.attribute_weights.ravel(),
        ]
        for rv in rb.precedent_referential_values:
            parts.append(rv)
        return np.concatenate(parts)

    def _flatten_params_unconstrained(self, normalize_rule_weights: bool = True) -> np.ndarray:
        """Flatten parameters into unconstrained space for JAX optimization.

        Applies inverse softmax (log) to belief degrees, inverse softmax or
        inverse sigmoid (logit) to rule weights, and inverse softplus to
        attribute weights. Referential values are left as-is (ordering is
        enforced by jnp.sort in the JAX path).
        """
        rb = self.rule_base
        eps = 1e-12

        # Inverse softmax: log(x) (softmax is shift-invariant, so log works)
        if self._allow_incomplete:
            # Invert the augmented softmax by measuring each belief against its
            # own block's ignorance column, whose logit is pinned at zero. A
            # block that arrives complete has no ignorance to measure against,
            # so seed it with a small one. The seed cannot be arbitrarily
            # small: the gradient of the loss with respect to these logits
            # scales with the ignorance itself, so a seed of 1e-6 leaves the
            # optimizer stranded thirteen logits away with nothing to follow.
            parts = []
            for block in rb.consequent_slices:
                beliefs = rb.belief_degrees[:, block]
                ignorance = np.clip(
                    1.0 - beliefs.sum(axis=1, keepdims=True), _INCOMPLETE_SEED, None
                )
                parts.append(np.log(np.clip(beliefs, eps, None) / ignorance))
            bd_log = np.concatenate(parts, axis=1)
        else:
            bd_log = np.log(np.clip(rb.belief_degrees, eps, None))

        if normalize_rule_weights:
            rw_inv = np.log(np.clip(rb.rule_weights, eps, None))
        else:
            # Inverse sigmoid (logit): log(p / (1 - p))
            rw_clipped = np.clip(rb.rule_weights, eps, 1.0 - eps)
            rw_inv = np.log(rw_clipped / (1.0 - rw_clipped))

        # Inverse softplus: log(exp(x) - 1); for x > ~20 this is just x
        aw = rb.attribute_weights
        aw_inv = np.where(aw > 20, aw, np.log(np.expm1(np.clip(aw, eps, None))))

        parts = [bd_log.ravel(), rw_inv, aw_inv.ravel()]
        for rv in rb.precedent_referential_values:
            parts.append(rv)
        return np.concatenate(parts)

    def _build_bounds_jax(
        self, fix_endpoints: bool, fix_endpoint_beliefs: bool = False
    ) -> list[tuple[float | None, float | None]]:
        """Construct bounds for the JAX/L-BFGS-B optimizer.

        Since belief degrees and rule weights use softmax reparameterization
        and attribute weights use softplus, those parameters are unconstrained.
        Only referential values have meaningful bounds. When
        ``fix_endpoint_beliefs`` is True, boundary rule belief logits are
        also fixed.
        """
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes
        bounds: list[tuple[float | None, float | None]] = []

        # Belief degrees (unconstrained logits)
        if fix_endpoint_beliefs:
            boundary = self._boundary_rule_mask()
            eps = 1e-12
            bd_log = np.log(np.clip(rb.belief_degrees, eps, None))
            for k in range(n_rules):
                for c in range(n_consequents):
                    if boundary[k]:
                        val = float(bd_log[k, c])
                        bounds.append((val, val))
                    else:
                        bounds.append((None, None))
        else:
            bounds.extend([(None, None)] * (n_rules * n_consequents))

        # Rule weights (unconstrained logits): no bounds
        bounds.extend([(None, None)] * n_rules)

        # Attribute weights (unconstrained softplus input): no bounds
        bounds.extend([(None, None)] * (n_rules * n_attributes))

        # Precedent referential values: same logic as NumPy path
        for rv in rb.precedent_referential_values:
            for j in range(len(rv)):
                if fix_endpoints and (j == 0 or j == len(rv) - 1):
                    bounds.append((float(rv[j]), float(rv[j])))
                else:
                    bounds.append((float(rv[0]), float(rv[-1])))

        return bounds

    def _unflatten_params(self, flat: np.ndarray, validate: bool = True) -> RuleBase:
        """Reconstruct a RuleBase from a flat parameter vector.

        Args:
            flat: The flat parameter vector.
            validate: If ``False``, skip Pydantic validation (used during
                optimization to avoid overhead on every iteration).
        """
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes

        idx = 0

        # belief_degrees
        size = n_rules * n_consequents
        belief_degrees = flat[idx : idx + size].reshape(n_rules, n_consequents)
        idx += size

        # rule_weights
        rule_weights = flat[idx : idx + n_rules]
        idx += n_rules

        # attribute_weights
        size = n_rules * n_attributes
        attribute_weights = flat[idx : idx + size].reshape(n_rules, n_attributes)
        idx += size

        # precedent referential values (varying lengths)
        precedent_referential_values = []
        for length in self._ref_value_lengths:
            precedent_referential_values.append(flat[idx : idx + length].copy())
            idx += length

        fields = {
            "precedent_referential_values": precedent_referential_values,
            "consequent_referential_values": rb.consequent_referential_values,
            "consequent_group_sizes": rb.consequent_group_sizes,
            "belief_degrees": belief_degrees,
            "rule_weights": rule_weights,
            "attribute_weights": attribute_weights,
            "rule_antecedent_indices": rb.rule_antecedent_indices,
        }

        if validate:
            return RuleBase(**fields)
        return RuleBase.model_construct(**fields)

    def _perturb_params(
        self,
        flat_params: np.ndarray,
        rng: np.random.Generator,
        fix_endpoints: bool,
        normalize_rule_weights: bool = True,
    ) -> np.ndarray:
        """Perturb a flat parameter vector to produce a valid restart point.

        Belief degrees get additive uniform noise in [-0.1, 0.1] and rows are
        re-normalized to sum to 1. Rule weights get the same treatment (and
        are renormalized if ``normalize_rule_weights`` is True). Attribute
        weights get a multiplicative factor in [0.8, 1.2]. Interior
        referential values get additive noise scaled to ±10% of the average
        spacing, then are re-sorted to maintain ordering. Endpoints stay
        fixed when ``fix_endpoints`` is True.

        The returned vector satisfies all constraints, so it can be passed
        to ``_unflatten_params`` and used as an optimizer starting point.
        """
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes

        flat = flat_params.copy()

        # Belief degrees: additive noise + clip + row renormalize
        bd_size = n_rules * n_consequents
        bd = flat[:bd_size].reshape(n_rules, n_consequents)
        bd = bd + rng.uniform(-0.1, 0.1, size=bd.shape)
        bd = np.clip(bd, 0.0, 1.0)
        bd = self._project_beliefs(bd)
        flat[:bd_size] = bd.ravel()

        # Rule weights: additive noise + clip + always renormalize.
        # Renormalization is unconditional because the RuleBase validator
        # always requires sum-to-1; the activation formula is scale-invariant
        # in rule weights, so this is mathematically equivalent regardless
        # of ``normalize_rule_weights``.
        rw_start = bd_size
        rw_end = rw_start + n_rules
        rw = flat[rw_start:rw_end] + rng.uniform(-0.1, 0.1, size=n_rules)
        rw = np.clip(rw, 0.0, 1.0)
        rw_sum = rw.sum()
        if rw_sum > 0:
            rw = rw / rw_sum
        else:
            rw = np.full(n_rules, 1.0 / n_rules)
        flat[rw_start:rw_end] = rw

        # Attribute weights: multiplicative factor in [0.8, 1.2]
        aw_start = rw_end
        aw_end = aw_start + n_rules * n_attributes
        aw_factors = rng.uniform(0.8, 1.2, size=n_rules * n_attributes)
        flat[aw_start:aw_end] = np.maximum(flat[aw_start:aw_end] * aw_factors, 0.0)

        # Referential values: additive noise scaled to ±10% of mean spacing
        pos = aw_end
        for length in self._ref_value_lengths:
            rv = flat[pos : pos + length].copy()
            if length > 1:
                spacing = (rv[-1] - rv[0]) / (length - 1)
            else:
                spacing = 0.0
            noise = rng.uniform(-0.1, 0.1, size=length) * spacing
            if fix_endpoints:
                noise[0] = 0.0
                noise[-1] = 0.0
            rv_perturbed = np.sort(rv + noise)
            flat[pos : pos + length] = rv_perturbed
            pos += length

        return flat

    def _normalize_flat(self, flat: np.ndarray) -> np.ndarray:
        """Project a flat parameter vector onto the constraint surface.

        Re-normalizes belief degree rows and rule weights to sum to exactly 1,
        clips attribute weights to >= 0, and sorts referential values. This
        corrects for small constraint violations from the optimizer.
        """
        flat = flat.copy()
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes

        # Belief degrees: clip to [0,1] and renormalize rows
        bd_size = n_rules * n_consequents
        bd = flat[:bd_size].reshape(n_rules, n_consequents)
        flat[:bd_size] = self._project_beliefs(bd).ravel()

        # Rule weights: clip and renormalize
        rw_start = bd_size
        rw_end = rw_start + n_rules
        rw = flat[rw_start:rw_end]
        rw = np.clip(rw, 0.0, 1.0)
        rw_sum = rw.sum()
        if rw_sum > 0:
            rw /= rw_sum
        flat[rw_start:rw_end] = rw

        # Attribute weights: clip to >= 0
        aw_start = rw_end
        aw_end = aw_start + n_rules * n_attributes
        flat[aw_start:aw_end] = np.clip(flat[aw_start:aw_end], 0.0, None)

        # Referential values: sort each attribute's values
        pos = aw_end
        for length in self._ref_value_lengths:
            flat[pos : pos + length] = np.sort(flat[pos : pos + length])
            pos += length

        return flat

    def _boundary_rule_mask(self) -> np.ndarray:
        """Return a boolean mask of shape (n_rules,) that is True for rules
        whose antecedents include a boundary (first or last) referential value
        for any attribute."""
        rb = self.rule_base
        mask = np.zeros(rb.n_rules, dtype=bool)
        for i in range(rb.n_attributes):
            max_idx = len(rb.precedent_referential_values[i]) - 1
            col = rb.rule_antecedent_indices[:, i]
            mask |= (col == 0) | (col == max_idx)
        return mask

    def _build_bounds(
        self, fix_endpoints: bool, fix_endpoint_beliefs: bool = False
    ) -> list[tuple[float, float]]:
        """Construct per-element bounds for the optimizer."""
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes
        bounds: list[tuple[float, float]] = []

        # belief_degrees: each element in [0, 1], but fix boundary rule beliefs
        if fix_endpoint_beliefs:
            boundary = self._boundary_rule_mask()
            for k in range(n_rules):
                for c in range(n_consequents):
                    if boundary[k]:
                        val = float(rb.belief_degrees[k, c])
                        bounds.append((val, val))
                    else:
                        bounds.append((0.0, 1.0))
        else:
            bounds.extend([(0.0, 1.0)] * (n_rules * n_consequents))

        # rule_weights: each element in [0, 1]
        bounds.extend([(0.0, 1.0)] * n_rules)

        # attribute_weights: [0, 10], capped to prevent numerical instability
        # in the exponentiation alpha^delta_bar during activation weight computation
        bounds.extend([(0.0, 10.0)] * (n_rules * n_attributes))

        # precedent referential values
        for rv in rb.precedent_referential_values:
            for j in range(len(rv)):
                if fix_endpoints and (j == 0 or j == len(rv) - 1):
                    # Fix endpoint: bound to current value
                    bounds.append((rv[j], rv[j]))
                else:
                    bounds.append((rv[0], rv[-1]))

        return bounds

    def _build_constraints(
        self,
        fix_endpoint_beliefs: bool = False,
        normalize_rule_weights: bool = True,
    ) -> list[dict]:
        """Construct SLSQP equality constraints for parameter normalization."""
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes

        # When endpoint beliefs are fixed via bounds, skip their sum-to-1
        # constraints to avoid a singular Jacobian in SLSQP.
        skip_bd_rules: set[int] = set()
        if fix_endpoint_beliefs:
            boundary = self._boundary_rule_mask()
            skip_bd_rules = {int(i) for i in np.where(boundary)[0]}

        constraints: list[dict] = []

        # Belief degree rows sum to 1 (skip fixed boundary rules)
        bd_offset = 0
        for k in range(n_rules):
            if k in skip_bd_rules:
                continue
            for block in rb.consequent_slices:
                row_start = bd_offset + k * n_consequents + block.start
                row_end = bd_offset + k * n_consequents + block.stop
                self._append_belief_constraint(constraints, row_start, row_end)

        # Rule weights sum to 1 (only if normalization is enabled)
        if normalize_rule_weights:
            rw_start = n_rules * n_consequents
            rw_end = rw_start + n_rules

            def rw_constraint(flat: np.ndarray) -> float:
                return float(flat[rw_start:rw_end].sum() - 1.0)

            constraints.append({"type": "eq", "fun": rw_constraint})

        # Referential values ordering: rv[j] <= rv[j+1] for each attribute
        rv_offset = n_rules * n_consequents + n_rules + n_rules * n_attributes
        pos = rv_offset
        for length in self._ref_value_lengths:
            for j in range(length - 1):

                def rv_order(flat: np.ndarray, p: int = pos + j, p1: int = pos + j + 1) -> float:
                    return float(flat[p1] - flat[p])

                constraints.append({"type": "ineq", "fun": rv_order})
            pos += length

        return constraints

    def _build_trust_constr_constraints(
        self,
        n_params: int,
        fix_endpoint_beliefs: bool = False,
        normalize_rule_weights: bool = True,
    ) -> list[LinearConstraint]:
        """Build constraints in the format required by trust-constr.

        Returns a list of ``LinearConstraint`` objects covering:
        - belief degree block sums, either fixed at 1 or capped by it when
          incomplete rules are allowed (skipping fixed boundary rules)
        - rule weight sum = 1
        - referential value ordering inequalities

        The trust-constr method uses ``LinearConstraint`` / ``NonlinearConstraint``
        objects rather than the dict format used by SLSQP.
        """
        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes

        skip_bd_rules: set[int] = set()
        if fix_endpoint_beliefs:
            boundary = self._boundary_rule_mask()
            skip_bd_rules = {int(i) for i in np.where(boundary)[0]}

        constraints: list[LinearConstraint] = []

        # Belief degree rows sum to 1 — combine all free rows into one matrix
        free_rules = [k for k in range(n_rules) if k not in skip_bd_rules]
        blocks = rb.consequent_slices
        if free_rules:
            rows = [(k, block) for k in free_rules for block in blocks]
            A_bd = np.zeros((len(rows), n_params))
            for row_i, (k, block) in enumerate(rows):
                offset = k * n_consequents
                A_bd[row_i, offset + block.start : offset + block.stop] = 1.0
            # An incomplete rule may assign anything up to all of its belief.
            bd_lb = 0.0 if self._allow_incomplete else 1.0
            constraints.append(LinearConstraint(A_bd, lb=bd_lb, ub=1.0))

        # Rule weights sum to 1 (only if normalization is enabled)
        if normalize_rule_weights:
            A_rw = np.zeros((1, n_params))
            rw_start = n_rules * n_consequents
            A_rw[0, rw_start : rw_start + n_rules] = 1.0
            constraints.append(LinearConstraint(A_rw, lb=1.0, ub=1.0))

        # Referential value ordering: rv[j+1] - rv[j] >= 0
        rv_offset = n_rules * n_consequents + n_rules + n_rules * n_attributes
        ordering_rows = []
        pos = rv_offset
        for length in self._ref_value_lengths:
            for j in range(length - 1):
                row = np.zeros(n_params)
                row[pos + j] = -1.0
                row[pos + j + 1] = 1.0
                ordering_rows.append(row)
            pos += length
        if ordering_rows:
            A_ord = np.array(ordering_rows)
            constraints.append(LinearConstraint(A_ord, lb=0.0, ub=np.inf))

        return constraints

    def _mse_objective(self, flat_params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE for a given flat parameter vector.

        Skips validation during optimization for speed; the final result
        is validated in ``fit()``.
        """
        self.rule_base = self._unflatten_params(flat_params, validate=False)
        y_pred = self.predict_values(X)
        return self._mean_squared_error(y, y_pred)
