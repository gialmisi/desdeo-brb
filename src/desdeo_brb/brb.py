"""Main BRB model class with an sklearn-compatible interface.

Provides the ``BRBModel`` class which supports fitting, predicting, and
inspecting a Belief Rule-Based inference system.
"""

import warnings
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from desdeo_brb.inference import (
    compute_activation_weights,
    compute_combined_belief_degrees,
    compute_output,
    input_transform,
)
from desdeo_brb.models import InferenceResult, RuleBase
from desdeo_brb.utils import build_rule_antecedent_indices, pad_referential_values


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
        consequent_referential_values: np.ndarray,
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
        self._consequent_referential_values = np.asarray(
            consequent_referential_values, dtype=float
        )
        self._utility_fn = utility_fn
        self._ref_value_lengths = [len(rv) for rv in self._precedent_referential_values]

        if rule_base is not None:
            self.rule_base = rule_base
        else:
            self.rule_base = self._build_default_rule_base(initial_rule_fn)

    def _build_default_rule_base(
        self, initial_rule_fn: Callable[[np.ndarray], float] | None = None
    ) -> RuleBase:
        """Construct a default RuleBase from the referential values."""
        rule_antecedent_indices = build_rule_antecedent_indices(
            self._precedent_referential_values
        )
        n_rules = len(rule_antecedent_indices)
        n_consequents = len(self._consequent_referential_values)
        n_attributes = len(self._precedent_referential_values)

        rule_weights = np.full(n_rules, 1.0 / n_rules)
        attribute_weights = np.ones((n_rules, n_attributes))

        if initial_rule_fn is not None:
            belief_degrees = self._beliefs_from_fn(
                initial_rule_fn, rule_antecedent_indices
            )
        else:
            belief_degrees = np.full((n_rules, n_consequents), 1.0 / n_consequents)

        return RuleBase(
            precedent_referential_values=self._precedent_referential_values,
            consequent_referential_values=self._consequent_referential_values,
            belief_degrees=belief_degrees,
            rule_weights=rule_weights,
            attribute_weights=attribute_weights,
            rule_antecedent_indices=rule_antecedent_indices,
        )

    def _beliefs_from_fn(
        self,
        fn: Callable[[np.ndarray], float],
        rule_antecedent_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute initial belief degrees from a function over antecedent values."""
        crv = self._consequent_referential_values
        n_rules = len(rule_antecedent_indices)
        n_consequents = len(crv)
        belief_degrees = np.zeros((n_rules, n_consequents))

        for k in range(n_rules):
            # Build the antecedent value vector for rule k
            x_k = np.array(
                [
                    self._precedent_referential_values[i][rule_antecedent_indices[k, i]]
                    for i in range(len(self._precedent_referential_values))
                ]
            )
            y_k = float(fn(x_k))

            # Distribute belief via linear interpolation on consequent values
            if y_k <= crv[0]:
                belief_degrees[k, 0] = 1.0
            elif y_k >= crv[-1]:
                belief_degrees[k, -1] = 1.0
            else:
                j = np.searchsorted(crv, y_k, side="right") - 1
                j = min(j, n_consequents - 2)
                frac = (y_k - crv[j]) / (crv[j + 1] - crv[j])
                belief_degrees[k, j] = 1.0 - frac
                belief_degrees[k, j + 1] = frac

        return belief_degrees

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
        combined = compute_combined_belief_degrees(rb.belief_degrees, weights)
        output = compute_output(
            combined, rb.consequent_referential_values, self._utility_fn
        )

        return InferenceResult(
            input_belief_distributions=alphas,
            activation_weights=weights,
            combined_belief_degrees=combined,
            consequent_values=rb.consequent_referential_values,
            output=output,
        )

    def _predict_jax(self, X: np.ndarray) -> InferenceResult:
        """JAX inference path. Converts results back to NumPy for the public API."""
        from desdeo_brb.jax_backend import (
            compute_activation_weights_jax,
            compute_combined_belief_degrees_jax,
            compute_output_jax,
            input_transform_jax,
        )

        import jax.numpy as jnp

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
            jnp.asarray(rb.belief_degrees), weights
        )
        output = compute_output_jax(
            combined, jnp.asarray(rb.consequent_referential_values)
        )

        # Convert back to numpy; split padded alphas into list
        alphas_list = [
            np.asarray(alphas_3d[:, i, : int(rv_lengths[i])])
            for i in range(len(rv_lengths))
        ]

        return InferenceResult(
            input_belief_distributions=alphas_list,
            activation_weights=np.asarray(weights),
            combined_belief_degrees=np.asarray(combined),
            consequent_values=rb.consequent_referential_values,
            output=np.asarray(output),
        )

    def predict_values(self, X: np.ndarray) -> np.ndarray:
        """Convenience method returning only the scalar outputs.

        Args:
            X: Input array of shape ``(n_samples, n_attributes)``.

        Returns:
            1-D array of shape ``(n_samples,)``.
        """
        return self.predict(X).output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool = True,
        fix_endpoint_beliefs: bool = False,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """Train the model by minimizing MSE.

        Uses SLSQP with finite differences for the NumPy backend, or
        L-BFGS-B with exact ``jax.grad`` gradients for the JAX backend.

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
            verbose: If ``True``, print optimizer progress.
            **minimize_kwargs: Extra keyword arguments forwarded to
                ``scipy.optimize.minimize``.

        Returns:
            self
        """
        if self._backend == "jax":
            return self._fit_jax(
                X, y, fix_endpoints, fix_endpoint_beliefs, verbose, **minimize_kwargs
            )
        return self._fit_numpy(
            X, y, fix_endpoints, fix_endpoint_beliefs, verbose, **minimize_kwargs
        )

    def _fit_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fix_endpoints: bool = True,
        fix_endpoint_beliefs: bool = True,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """NumPy training path using SLSQP."""
        x0 = self._flatten_params()
        bounds = self._build_bounds(fix_endpoints, fix_endpoint_beliefs)
        constraints = self._build_constraints(fix_endpoint_beliefs)

        def objective(flat: np.ndarray) -> float:
            return self._mse_objective(flat, X, y)

        options = minimize_kwargs.pop("options", {})
        if not verbose:
            options.setdefault("disp", False)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
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
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """JAX training path using L-BFGS-B with exact gradients."""
        import jax
        import jax.numpy as jnp

        from desdeo_brb.jax_backend import full_inference_jax_unconstrained

        rb = self.rule_base
        X_jax = jnp.asarray(X)
        y_jax = jnp.asarray(y)
        crv_jax = jnp.asarray(rb.consequent_referential_values)
        rai_jax = jnp.asarray(rb.rule_antecedent_indices)
        rv_lengths_tuple = tuple(self._ref_value_lengths)

        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes

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
            )
            return jnp.mean((y_jax - y_pred) ** 2)

        grad_fn = jax.jit(jax.grad(mse_loss))

        def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
            flat_jax = jnp.asarray(flat)
            loss = float(mse_loss(flat_jax))
            grad = np.asarray(grad_fn(flat_jax))
            return loss, grad

        # Transform initial params to unconstrained space matching the
        # softmax/softplus reparameterization in full_inference_jax.
        x0 = self._flatten_params_unconstrained()
        bounds = self._build_bounds_jax(fix_endpoints, fix_endpoint_beliefs)

        options = minimize_kwargs.pop("options", {})

        result = minimize(
            objective_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options=options,
            **minimize_kwargs,
        )

        self.rule_base = self._unflatten_from_unconstrained(result.x)
        return self

    def _unflatten_from_unconstrained(self, flat: np.ndarray) -> RuleBase:
        """Convert unconstrained JAX optimizer output to a validated RuleBase.

        Applies softmax to belief degree rows and rule weights, softplus to
        attribute weights, and sorts referential values.
        """
        from scipy.special import softmax as sp_softmax

        rb = self.rule_base
        n_rules = rb.n_rules
        n_consequents = rb.n_consequents
        n_attributes = rb.n_attributes
        idx = 0

        bd_size = n_rules * n_consequents
        bd_raw = flat[idx : idx + bd_size].reshape(n_rules, n_consequents)
        belief_degrees = sp_softmax(bd_raw, axis=1)
        idx += bd_size

        rw_raw = flat[idx : idx + n_rules]
        rule_weights = sp_softmax(rw_raw)
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
            belief_degrees=belief_degrees,
            rule_weights=rule_weights,
            attribute_weights=attribute_weights,
            rule_antecedent_indices=rb.rule_antecedent_indices,
        )

    def fit_custom(
        self,
        loss_fn: Callable[["BRBModel"], float],
        constraints: list[dict] | None = None,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """Train using a user-supplied loss function.

        The loss function receives the model instance (with updated parameters)
        and should return a scalar loss value.

        For the JAX backend, ``jax.grad`` is attempted on the loss function.
        If the loss function is not JAX-compatible, falls back to SLSQP
        with finite differences and emits a warning.

        Args:
            loss_fn: Callable with signature ``loss_fn(model) -> float``.
            constraints: Additional SLSQP constraints (dicts with ``type``,
                ``fun`` keys). BRB-specific constraints are always included.
                Ignored when using the JAX/L-BFGS-B path.
            verbose: If ``True``, print optimizer progress.
            **minimize_kwargs: Extra keyword arguments forwarded to
                ``scipy.optimize.minimize``.

        Returns:
            self
        """
        if self._backend == "jax":
            return self._fit_custom_jax(
                loss_fn, constraints, verbose, **minimize_kwargs
            )
        return self._fit_custom_numpy(loss_fn, constraints, verbose, **minimize_kwargs)

    def _fit_custom_numpy(
        self,
        loss_fn: Callable[["BRBModel"], float],
        constraints: list[dict] | None = None,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """NumPy custom training via SLSQP."""
        x0 = self._flatten_params()
        bounds = self._build_bounds(fix_endpoints=False)
        all_constraints = self._build_constraints()
        if constraints is not None:
            all_constraints.extend(constraints)

        def objective(flat: np.ndarray) -> float:
            self.rule_base = self._unflatten_params(flat, validate=False)
            return loss_fn(self)

        options = minimize_kwargs.pop("options", {})
        if not verbose:
            options.setdefault("disp", False)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=all_constraints,
            options=options,
            **minimize_kwargs,
        )

        self.rule_base = self._unflatten_params(self._normalize_flat(result.x))
        return self

    def _fit_custom_jax(
        self,
        loss_fn: Callable[["BRBModel"], float],
        constraints: list[dict] | None = None,
        verbose: bool = False,
        **minimize_kwargs: Any,
    ) -> "BRBModel":
        """JAX custom training: try jax.grad, fall back to SLSQP."""
        import jax
        import jax.numpy as jnp

        x0 = self._flatten_params()

        # Try to build a JAX-differentiable objective
        def jax_objective(flat_jax):
            self.rule_base = self._unflatten_params(
                np.asarray(flat_jax), validate=False
            )
            return loss_fn(self)

        try:
            # Test if jax.grad works on the loss function
            test_grad = jax.grad(jax_objective)(jnp.asarray(x0))
            if not jnp.all(jnp.isfinite(test_grad)):
                raise ValueError("Non-finite gradients")

            grad_fn = jax.grad(jax_objective)

            def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
                flat_jax = jnp.asarray(flat)
                loss = float(jax_objective(flat_jax))
                grad = np.asarray(grad_fn(flat_jax))
                return loss, grad

            bounds = self._build_bounds(fix_endpoints=False)
            options = minimize_kwargs.pop("options", {})

            result = minimize(
                objective_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options=options,
                **minimize_kwargs,
            )

            self.rule_base = self._unflatten_from_unconstrained(result.x)
            return self

        except Exception:
            warnings.warn(
                "Custom loss function is not JAX-differentiable. "
                "Falling back to SLSQP with finite differences.",
                stacklevel=2,
            )
            return self._fit_custom_numpy(
                loss_fn, constraints, verbose, **minimize_kwargs
            )

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
                np.asarray(rv, dtype=float)
                for rv in params["precedent_referential_values"]
            ]
            self._ref_value_lengths = [
                len(rv) for rv in self._precedent_referential_values
            ]
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

        Args:
            X: Input array, shape ``(n_samples, n_attributes)``.
            y: True target values, shape ``(n_samples,)``.

        Returns:
            Negative mean squared error.
        """
        y_pred = self.predict_values(X)
        return -float(np.mean((y - y_pred) ** 2))

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

    def _flatten_params_unconstrained(self) -> np.ndarray:
        """Flatten parameters into unconstrained space for JAX optimization.

        Applies inverse softmax (log) to belief degrees and rule weights,
        and inverse softplus to attribute weights. Referential values are
        left as-is (ordering is enforced by jnp.sort in the JAX path).
        """
        rb = self.rule_base
        eps = 1e-12

        # Inverse softmax: log(x) (softmax is shift-invariant, so log works)
        bd_log = np.log(np.clip(rb.belief_degrees, eps, None))
        rw_log = np.log(np.clip(rb.rule_weights, eps, None))

        # Inverse softplus: log(exp(x) - 1); for x > ~20 this is just x
        aw = rb.attribute_weights
        aw_inv = np.where(aw > 20, aw, np.log(np.expm1(np.clip(aw, eps, None))))

        parts = [bd_log.ravel(), rw_log, aw_inv.ravel()]
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
            "belief_degrees": belief_degrees,
            "rule_weights": rule_weights,
            "attribute_weights": attribute_weights,
            "rule_antecedent_indices": rb.rule_antecedent_indices,
        }

        if validate:
            return RuleBase(**fields)
        return RuleBase.model_construct(**fields)

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
        bd = np.clip(bd, 0.0, 1.0)
        row_sums = bd.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        bd /= row_sums
        flat[:bd_size] = bd.ravel()

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
        for i, rv in enumerate(rb.precedent_referential_values):
            for j in range(len(rv)):
                if fix_endpoints and (j == 0 or j == len(rv) - 1):
                    # Fix endpoint: bound to current value
                    bounds.append((rv[j], rv[j]))
                else:
                    bounds.append((rv[0], rv[-1]))

        return bounds

    def _build_constraints(
        self, fix_endpoint_beliefs: bool = False
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
            skip_bd_rules = set(int(i) for i in np.where(boundary)[0])

        constraints: list[dict] = []

        # Belief degree rows sum to 1 (skip fixed boundary rules)
        bd_offset = 0
        for k in range(n_rules):
            if k in skip_bd_rules:
                continue
            row_start = bd_offset + k * n_consequents
            row_end = row_start + n_consequents

            def bd_constraint(
                flat: np.ndarray, s: int = row_start, e: int = row_end
            ) -> float:
                return float(flat[s:e].sum() - 1.0)

            constraints.append({"type": "eq", "fun": bd_constraint})

        # Rule weights sum to 1
        rw_start = n_rules * n_consequents
        rw_end = rw_start + n_rules

        def rw_constraint(flat: np.ndarray) -> float:
            return float(flat[rw_start:rw_end].sum() - 1.0)

        constraints.append({"type": "eq", "fun": rw_constraint})

        # Referential values ordering: rv[j] <= rv[j+1] for each attribute
        rv_offset = n_rules * n_consequents + n_rules + n_rules * n_attributes
        pos = rv_offset
        for i, length in enumerate(self._ref_value_lengths):
            for j in range(length - 1):

                def rv_order(
                    flat: np.ndarray, p: int = pos + j, p1: int = pos + j + 1
                ) -> float:
                    return float(flat[p1] - flat[p])

                constraints.append({"type": "ineq", "fun": rv_order})
            pos += length

        return constraints

    def _mse_objective(
        self, flat_params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute MSE for a given flat parameter vector.

        Skips validation during optimization for speed; the final result
        is validated in ``fit()``.
        """
        self.rule_base = self._unflatten_params(flat_params, validate=False)
        y_pred = self.predict_values(X)
        return float(np.mean((y - y_pred) ** 2))
