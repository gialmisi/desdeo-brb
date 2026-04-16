"""Pure functions implementing the BRB inference pipeline.

Each function corresponds to one step in the BRB evidential reasoning process:
input transformation, activation weight computation, belief combination, and
output aggregation. All functions are vectorized over samples.
"""

from collections.abc import Callable

import numpy as np


def input_transform(X: np.ndarray, referential_values: list[np.ndarray]) -> list[np.ndarray]:
    """Transform raw inputs into belief distributions over referential values.

    Implements Eq. A-1 from Chen et al. (2011). For each attribute, computes
    matching degrees (alphas) describing how strongly each input relates to
    each referential value using piecewise-linear interpolation with wrapping
    boundary conditions.

    Args:
        X: Input array of shape ``(n_samples, n_attributes)``.
        referential_values: List of 1-D arrays, one per attribute, each
            containing the sorted referential values for that attribute.
            Arrays may have different lengths.

    Returns:
        List of 2-D arrays, one per attribute. The *i*-th array has shape
        ``(n_samples, len(referential_values[i]))`` and contains the belief
        degrees for each sample over the *i*-th attribute's referential values.

    Notes:
        Following the RIMER wrapping boundary conditions (Yang et al. 2006),
        inputs outside the referential value range are clamped to the nearest
        boundary. This means inputs below the minimum get belief degree 1.0
        at the first referential value, and inputs above the maximum get
        belief degree 1.0 at the last referential value. At most two adjacent
        referential values receive nonzero belief, and they always sum to 1.
    """
    n_samples = X.shape[0]
    alphas: list[np.ndarray] = []

    for i, rv in enumerate(referential_values):
        n_rv = len(rv)
        alpha = np.zeros((n_samples, n_rv))

        # Clamp inputs to the referential value range (RIMER boundary condition)
        h = np.clip(X[:, i], rv[0], rv[-1])

        if n_rv == 1:
            alpha[:, 0] = 1.0
        else:
            # Find the interval: j such that rv[j] <= h < rv[j+1]
            j = np.searchsorted(rv, h, side="right") - 1
            j = np.clip(j, 0, n_rv - 2)

            # Belief for the upper adjacent value
            denom = rv[j + 1] - rv[j]
            # When two adjacent referential values coincide (degenerate
            # interval during optimization), assign full belief to the lower.
            safe_denom = np.where(denom > 0, denom, 1.0)
            alpha_upper = np.where(denom > 0, (h - rv[j]) / safe_denom, 0.0)
            alpha_lower = 1.0 - alpha_upper

            rows = np.arange(n_samples)
            alpha[rows, j] = alpha_lower
            alpha[rows, j + 1] = alpha_upper

        alphas.append(alpha)

    return alphas


def compute_activation_weights(
    alphas: list[np.ndarray],
    rule_antecedent_indices: np.ndarray,
    thetas: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """Compute the activation weight of each rule given input belief distributions.

    Implements Eq. A-3 from Chen et al. (2011).

    Args:
        alphas: List of 2-D arrays from :func:`input_transform`, one per
            attribute. The *i*-th array has shape ``(n_samples, n_rv_i)``.
        rule_antecedent_indices: 2-D integer array of shape
            ``(n_rules, n_attributes)``. Entry ``[k, i]`` is the index into
            ``referential_values[i]`` that rule *k* uses for attribute *i*.
        thetas: 1-D array of rule weights, shape ``(n_rules,)``.
        deltas: 2-D array of attribute weights, shape
            ``(n_rules, n_attributes)``.

    Returns:
        2-D array of shape ``(n_samples, n_rules)`` with activation weights
        that sum to 1 across rules for each sample.

    Notes:
        When all matching degrees for a rule are zero the rule does not fire.
        A small epsilon (1e-12) is added only to the denominator during
        normalization to avoid division by zero; the numerator products are
        computed without any epsilon.
    """
    n_rules, n_attributes = rule_antecedent_indices.shape
    n_samples = alphas[0].shape[0]

    # Normalize attribute weights per rule: delta_bar_{i,k} = delta_{i,k} / max_i(delta_{i,k})
    delta_max = deltas.max(axis=1, keepdims=True)  # (n_rules, 1)
    # Avoid division by zero for rules with all-zero weights
    safe_delta_max = np.where(delta_max > 0, delta_max, 1.0)
    delta_bar = np.where(delta_max > 0, deltas / safe_delta_max, 0.0)  # (n_rules, n_attributes)

    # Gather the matching degree for each rule's antecedent per attribute
    # alpha_selected[i] has shape (n_samples, n_rules): for attribute i,
    # the matching degree at the index specified by each rule.
    alpha_selected = np.empty((n_attributes, n_samples, n_rules))
    for i in range(n_attributes):
        indices = rule_antecedent_indices[:, i]  # (n_rules,)
        alpha_selected[i] = alphas[i][:, indices]  # (n_samples, n_rules)

    # Compute unnormalized weights: theta_k * prod_i(alpha_{i,n}^{delta_bar_{i,k}})
    # Use log-space for the product: sum_i delta_bar_{i,k} * log(alpha_{i,n})
    # But alpha can be 0, so handle carefully.
    log_product = np.zeros((n_samples, n_rules))
    any_zero = np.zeros((n_samples, n_rules), dtype=bool)

    for i in range(n_attributes):
        a = alpha_selected[i]  # (n_samples, n_rules)
        db = delta_bar[:, i]  # (n_rules,)
        # Where alpha is 0 and delta_bar > 0, the product is 0
        zero_mask = (a == 0) & (db > 0)
        any_zero |= zero_mask
        # Safe log: replace 0 with 1 (log(1)=0) to avoid -inf, we handle zeros via any_zero
        safe_a = np.where(a > 0, a, 1.0)
        log_product += db * np.log(safe_a)

    unnorm = thetas * np.where(any_zero, 0.0, np.exp(log_product))  # (n_samples, n_rules)

    # Normalize across rules per sample
    denom = unnorm.sum(axis=1, keepdims=True) + 1e-12
    return unnorm / denom


def compute_combined_belief_degrees(bre_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine activated belief degrees using the analytical evidential reasoning algorithm.

    Implements Eq. A-15 from Chen et al. (2011) / Eq. 3.20 from the thesis.

    Args:
        bre_matrix: 2-D array of shape ``(n_rules, n_consequents)`` containing
            the belief degrees assigned to each consequent for every rule.
        weights: 2-D array of shape ``(n_samples, n_rules)`` with activation
            weights from :func:`compute_activation_weights`.

    Returns:
        2-D array of shape ``(n_samples, n_consequents)`` with the combined
        belief degrees.

    Notes:
        Products over rules are computed in log-space for numerical stability.
        When a rule's activation weight is zero for a given sample, that rule
        contributes a multiplicative factor of 1 (i.e., is skipped). This is
        achieved by setting the log-space contribution to 0 for inactive rules.

        The formula is:

        .. math::

            \\beta_n = \\frac{
                \\prod_k (w_k \\beta_{n,k} + 1 - w_k \\sum_j \\beta_{j,k})
                - \\prod_k (1 - w_k \\sum_j \\beta_{j,k})
            }{
                \\sum_j \\prod_k (w_k \\beta_{j,k} + 1 - w_k \\sum_j \\beta_{j,k})
                - (N-1) \\prod_k (1 - w_k \\sum_j \\beta_{j,k})
                - \\prod_k (1 - w_k)
            }
    """
    n_rules, n_consequents = bre_matrix.shape

    # Row sums of BRE matrix: sum_j beta_{j,k} for each rule
    bre_row_sums = bre_matrix.sum(axis=1)  # (n_rules,)

    # Precompute terms that appear in the products:
    # For each consequent n and rule k:
    #   c_{n,k} = w_k * beta_{n,k} + 1 - w_k * sum_j(beta_{j,k})
    # For the "residual" term:
    #   r_k = 1 - w_k * sum_j(beta_{j,k})
    # For the "inactive" term:
    #   q_k = 1 - w_k

    # w: (n_samples, n_rules)
    # bre_matrix: (n_rules, n_consequents) -> broadcast as (1, n_rules, n_consequents)
    # bre_row_sums: (n_rules,) -> (1, n_rules)

    w = weights  # (n_samples, n_rules)
    w_expanded = w[:, :, np.newaxis]  # (n_samples, n_rules, 1)
    bre_expanded = bre_matrix[np.newaxis, :, :]  # (1, n_rules, n_consequents)
    brs = bre_row_sums[np.newaxis, :]  # (1, n_rules)

    # c_{n,k} for each sample, rule, consequent
    c = w_expanded * bre_expanded + 1.0 - w[:, :, np.newaxis] * brs[:, :, np.newaxis]
    # (n_samples, n_rules, n_consequents)

    # r_k: residual term per sample, rule
    r = 1.0 - w * brs  # (n_samples, n_rules)

    # q_k: inactive term per sample, rule
    q = 1.0 - w  # (n_samples, n_rules)

    # --- Log-space products over rules (axis=1) ---
    # For inactive rules (w_k ≈ 0), all terms reduce to 1, contributing 0 in log-space.
    # We use log(max(val, tiny)) to avoid log(0) for edge cases.
    tiny = np.finfo(float).tiny

    # prod_k c_{n,k} for each consequent n: shape (n_samples, n_consequents)
    log_c = np.log(np.maximum(c, tiny))  # (n_samples, n_rules, n_consequents)
    log_prod_c = log_c.sum(axis=1)  # (n_samples, n_consequents)

    # prod_k r_k: shape (n_samples,)
    log_r = np.log(np.maximum(r, tiny))
    log_prod_r = log_r.sum(axis=1)  # (n_samples,)

    # prod_k q_k: shape (n_samples,)
    log_q = np.log(np.maximum(q, tiny))
    log_prod_q = log_q.sum(axis=1)  # (n_samples,)

    prod_c = np.exp(log_prod_c)  # (n_samples, n_consequents)
    prod_r = np.exp(log_prod_r)  # (n_samples,)
    prod_q = np.exp(log_prod_q)  # (n_samples,)

    # Numerator for each consequent n: prod_k(c_{n,k}) - prod_k(r_k)
    numerator = prod_c - prod_r[:, np.newaxis]  # (n_samples, n_consequents)

    # Denominator: sum_j prod_k(c_{j,k}) - (N-1)*prod_k(r_k) - prod_k(q_k)
    denominator = prod_c.sum(axis=1) - (n_consequents - 1) * prod_r - prod_q  # (n_samples,)

    beta = numerator / (denominator[:, np.newaxis] + 1e-12)

    return beta


def compute_output(
    belief_degrees: np.ndarray,
    consequents: np.ndarray,
    utility_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute scalar outputs from combined belief degrees and consequent values.

    Applies a utility function to the consequent referential values, then
    computes the weighted sum with the combined belief degrees.

    Args:
        belief_degrees: 2-D array of shape ``(n_samples, n_consequents)``
            with combined belief degrees.
        consequents: 1-D array of shape ``(n_consequents,)`` with
            consequent referential values.
        utility_fn: Optional function mapping consequent values to utilities.
            Signature: ``f(np.ndarray) -> np.ndarray``. If ``None``, the
            identity function is used (i.e., ``u(D_n) = D_n``).

    Returns:
        1-D array of shape ``(n_samples,)`` with the scalar outputs.
    """
    if utility_fn is not None:
        u = utility_fn(consequents)
    else:
        u = consequents

    return belief_degrees @ u
