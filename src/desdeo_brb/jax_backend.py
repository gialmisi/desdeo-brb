"""JAX backend for the BRB inference pipeline.

Provides JIT-compiled versions of the inference functions and a
differentiable end-to-end inference function for gradient-based training.

Because ``jax.jit`` traces code and requires static array shapes, the
varying-length referential value arrays are padded into fixed-size 2D
arrays (see :func:`~desdeo_brb.utils.pad_referential_values`). Unused
entries are filled with ``np.inf`` and masked during computation.

The ``rv_lengths`` parameter is passed as a tuple of Python ints (not a
JAX array) and declared as a static argument for JIT, so that the lengths
are available as concrete values at trace time.
"""

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _check_jax() -> None:
    if not JAX_AVAILABLE:
        raise ImportError("Install JAX: pip install desdeo-brb[jax]")


# Input transformation


def input_transform_jax(
    X: "jnp.ndarray",
    padded_rv: "jnp.ndarray",
    rv_lengths: tuple[int, ...],
) -> "jnp.ndarray":
    """Transform raw inputs into belief distributions (JAX version).

    Args:
        X: Input array of shape ``(n_samples, n_attributes)``.
        padded_rv: Padded referential values of shape
            ``(n_attributes, max_n_ref_values)``. Unused entries are ``inf``.
        rv_lengths: Tuple of ints with the actual number of referential
            values per attribute. Must be concrete (not a traced array).

    Returns:
        3D array of shape ``(n_samples, n_attributes, max_n_ref_values)``
        with belief degrees. Padded positions are zero.
    """
    _check_jax()
    n_samples, n_attributes = X.shape
    max_rv = padded_rv.shape[1]

    def _transform_one_attr(h, rv, length):
        """Transform one attribute for all samples."""
        valid = jnp.arange(max_rv) < length
        safe_rv = jnp.where(valid, rv, 0.0)

        alpha = jnp.zeros((n_samples, max_rv))

        rv_min = safe_rv[0]
        rv_max = safe_rv[length - 1] if length > 1 else safe_rv[0]
        in_range = (h >= rv_min) & (h <= rv_max)

        # Find interval via cumulative comparison
        le_count = jnp.sum(
            (safe_rv[jnp.newaxis, :] <= h[:, jnp.newaxis]) & valid[jnp.newaxis, :],
            axis=1,
        )
        j = jnp.clip(le_count - 1, 0, max(length - 2, 0))

        rv_j = safe_rv[j]
        rv_j1 = safe_rv[jnp.minimum(j + 1, length - 1)]

        denom = rv_j1 - rv_j
        safe_denom = jnp.where(denom > 0, denom, 1.0)
        alpha_upper = jnp.where(denom > 0, (h - rv_j) / safe_denom, 0.0)
        alpha_lower = 1.0 - alpha_upper

        rows = jnp.arange(n_samples)
        alpha = alpha.at[rows, j].set(jnp.where(in_range, alpha_lower, 0.0))
        j1_clamped = jnp.minimum(j + 1, max_rv - 1)
        alpha = alpha.at[rows, j1_clamped].set(
            jnp.where(in_range & (j + 1 < length), alpha_upper, 0.0)
        )

        alpha = alpha * valid[jnp.newaxis, :]
        return alpha

    result = jnp.zeros((n_samples, n_attributes, max_rv))
    for i in range(n_attributes):
        alpha_i = _transform_one_attr(X[:, i], padded_rv[i], rv_lengths[i])
        result = result.at[:, i, :].set(alpha_i)

    return result


# Activation weights


def compute_activation_weights_jax(
    alphas: "jnp.ndarray",
    rule_antecedent_indices: "jnp.ndarray",
    thetas: "jnp.ndarray",
    deltas: "jnp.ndarray",
) -> "jnp.ndarray":
    """Compute activation weights (JAX version).

    Args:
        alphas: 3-D array from :func:`input_transform_jax`, shape
            ``(n_samples, n_attributes, max_n_ref_values)``.
        rule_antecedent_indices: Integer array ``(n_rules, n_attributes)``.
        thetas: Rule weights ``(n_rules,)``.
        deltas: Attribute weights ``(n_rules, n_attributes)``.

    Returns:
        2-D array of shape ``(n_samples, n_rules)``.
    """
    _check_jax()
    n_rules, n_attributes = rule_antecedent_indices.shape

    delta_max = deltas.max(axis=1, keepdims=True)
    safe_delta_max = jnp.where(delta_max > 0, delta_max, 1.0)
    delta_bar = jnp.where(delta_max > 0, deltas / safe_delta_max, 0.0)

    alpha_selected = jnp.stack(
        [alphas[:, i, rule_antecedent_indices[:, i]] for i in range(n_attributes)],
        axis=-1,
    )

    eps = 1e-30
    safe_alpha = jnp.maximum(alpha_selected, eps)
    log_alpha = jnp.log(safe_alpha)

    db = delta_bar[jnp.newaxis, :, :]
    log_product = jnp.sum(db * log_alpha, axis=-1)

    is_zero = alpha_selected < eps
    has_weight = db > 0
    any_zero = jnp.any(is_zero & has_weight, axis=-1)

    unnorm = thetas * jnp.where(any_zero, 0.0, jnp.exp(log_product))
    denom = unnorm.sum(axis=1, keepdims=True) + 1e-12
    return unnorm / denom


# Combined belief degrees


def compute_combined_belief_degrees_jax(
    bre_matrix: "jnp.ndarray",
    weights: "jnp.ndarray",
) -> "jnp.ndarray":
    """Combine belief degrees using the ER algorithm (JAX version).

    Args:
        bre_matrix: Shape ``(n_rules, n_consequents)``.
        weights: Shape ``(n_samples, n_rules)``.

    Returns:
        Shape ``(n_samples, n_consequents)``.
    """
    _check_jax()
    n_rules, n_consequents = bre_matrix.shape

    bre_row_sums = bre_matrix.sum(axis=1)

    w = weights
    w_expanded = w[:, :, jnp.newaxis]
    bre_expanded = bre_matrix[jnp.newaxis, :, :]
    brs = bre_row_sums[jnp.newaxis, :]

    c = w_expanded * bre_expanded + 1.0 - w[:, :, jnp.newaxis] * brs[:, :, jnp.newaxis]
    r = 1.0 - w * brs
    q = 1.0 - w

    tiny = jnp.finfo(jnp.float64).tiny

    log_c = jnp.log(jnp.maximum(c, tiny))
    log_prod_c = log_c.sum(axis=1)

    log_r = jnp.log(jnp.maximum(r, tiny))
    log_prod_r = log_r.sum(axis=1)

    log_q = jnp.log(jnp.maximum(q, tiny))
    log_prod_q = log_q.sum(axis=1)

    prod_c = jnp.exp(log_prod_c)
    prod_r = jnp.exp(log_prod_r)
    prod_q = jnp.exp(log_prod_q)

    numerator = prod_c - prod_r[:, jnp.newaxis]
    denominator = prod_c.sum(axis=1) - (n_consequents - 1) * prod_r - prod_q

    beta = numerator / (denominator[:, jnp.newaxis] + 1e-12)
    return beta


# Output


def compute_output_jax(
    belief_degrees: "jnp.ndarray",
    consequents: "jnp.ndarray",
) -> "jnp.ndarray":
    """Compute scalar outputs (JAX version, identity utility only).

    Args:
        belief_degrees: Shape ``(n_samples, n_consequents)``.
        consequents: Shape ``(n_consequents,)``.

    Returns:
        Shape ``(n_samples,)``.
    """
    _check_jax()
    return belief_degrees @ consequents


# Full differentiable inference


def full_inference_jax(
    flat_params: "jnp.ndarray",
    X: "jnp.ndarray",
    consequent_rv: "jnp.ndarray",
    rule_antecedent_indices: "jnp.ndarray",
    n_rules: int,
    n_consequents: int,
    n_attributes: int,
    rv_lengths: tuple[int, ...],
) -> "jnp.ndarray":
    """End-to-end differentiable inference from flat parameters to outputs.

    This is a pure function suitable for ``jax.jit`` and ``jax.grad``.
    It unflattens the parameter vector, runs all inference steps, and
    returns scalar outputs.

    Args:
        flat_params: 1-D parameter vector (same layout as
            :meth:`BRBModel._flatten_params`).
        X: Input array ``(n_samples, n_attributes)``.
        consequent_rv: Consequent referential values ``(n_consequents,)``.
        rule_antecedent_indices: Integer array ``(n_rules, n_attributes)``.
        n_rules: Number of rules (static).
        n_consequents: Number of consequent values (static).
        n_attributes: Number of attributes (static).
        rv_lengths: Tuple of Python ints with referential value lengths
            per attribute (static — required for JIT tracing).

    Returns:
        1D array of shape ``(n_samples,)`` with predicted outputs.
    """
    _check_jax()

    # Unflatten parameters
    idx = 0
    bd_size = n_rules * n_consequents
    belief_degrees = flat_params[idx : idx + bd_size].reshape(n_rules, n_consequents)
    idx += bd_size

    rule_weights = flat_params[idx : idx + n_rules]
    idx += n_rules

    aw_size = n_rules * n_attributes
    attribute_weights = flat_params[idx : idx + aw_size].reshape(n_rules, n_attributes)
    idx += aw_size

    # Referential values: unflatten into padded 2D array
    max_rv = max(rv_lengths)
    padded_rv = jnp.full((n_attributes, max_rv), jnp.inf)
    for i in range(n_attributes):
        length = rv_lengths[i]
        padded_rv = padded_rv.at[i, :length].set(flat_params[idx : idx + length])
        idx += length

    # Run inference pipeline
    alphas = input_transform_jax(X, padded_rv, rv_lengths)
    weights = compute_activation_weights_jax(
        alphas, rule_antecedent_indices, rule_weights, attribute_weights
    )
    combined = compute_combined_belief_degrees_jax(belief_degrees, weights)
    output = compute_output_jax(combined, consequent_rv)

    return output


def full_inference_jax_unconstrained(
    flat_params: "jnp.ndarray",
    X: "jnp.ndarray",
    consequent_rv: "jnp.ndarray",
    rule_antecedent_indices: "jnp.ndarray",
    n_rules: int,
    n_consequents: int,
    n_attributes: int,
    rv_lengths: tuple[int, ...],
) -> "jnp.ndarray":
    """End-to-end inference from unconstrained parameters.

    Wraps :func:`full_inference_jax` with differentiable reparameterization:
    softmax for belief degree rows and rule weights, softplus for attribute
    weights, and sort for referential values. This allows L-BFGS-B (box
    bounds only) to optimize without explicit equality constraints.

    Args:
        Same as :func:`full_inference_jax`, except ``flat_params`` is in
        unconstrained space (logits for belief degrees/rule weights,
        unconstrained reals for attribute weights).
    """
    _check_jax()
    idx = 0

    # Apply softmax to belief degrees
    bd_size = n_rules * n_consequents
    bd_raw = flat_params[idx : idx + bd_size].reshape(n_rules, n_consequents)
    bd = jax.nn.softmax(bd_raw, axis=1)
    idx += bd_size

    # Apply softmax to rule weights
    rw_raw = flat_params[idx : idx + n_rules]
    rw = jax.nn.softmax(rw_raw)
    idx += n_rules

    # Apply softplus to attribute weights
    aw_size = n_rules * n_attributes
    aw_raw = flat_params[idx : idx + aw_size]
    aw = jax.nn.softplus(aw_raw)
    idx += aw_size

    # Sort referential values
    rv_parts = []
    for i in range(n_attributes):
        length = rv_lengths[i]
        rv_parts.append(jnp.sort(flat_params[idx : idx + length]))
        idx += length

    # Reassemble constrained flat params
    constrained = jnp.concatenate([bd.ravel(), rw, aw] + rv_parts)

    return full_inference_jax(
        constrained,
        X,
        consequent_rv,
        rule_antecedent_indices,
        n_rules,
        n_consequents,
        n_attributes,
        rv_lengths,
    )


# JIT-compiled versions
if JAX_AVAILABLE:
    full_inference_jax_jit = jax.jit(
        full_inference_jax,
        static_argnames=("n_rules", "n_consequents", "n_attributes", "rv_lengths"),
    )
