"""Pyomo backend for symbolic BRB optimization with IPOPT.

Builds a Pyomo ``ConcreteModel`` that encodes the entire BRB inference
pipeline as symbolic expressions, suitable for solving with interior-point
solvers like IPOPT. The MSE objective is built from the predictions on
training data.

Two modes for handling the input transform:

- ``optimize_referential_values=False``: precompute alphas using the
  NumPy ``input_transform`` and treat them as constants in the Pyomo
  expression tree. Fast and avoids the smooth-approximation error from
  the symbolic input transform. The referential values themselves are
  stored as Pyomo Params (not Vars) and cannot move during optimization.
- ``optimize_referential_values=True``: express the input transform
  symbolically using smooth ``min``/``max`` approximations. The
  referential values become Vars; smooth approximations introduce a
  small error of order ``sqrt(eps_smooth)``.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

try:
    import pyomo.environ as pyo

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

if TYPE_CHECKING:
    from desdeo_brb.brb import BRBModel

from desdeo_brb.inference import input_transform


def _check_pyomo() -> None:
    if not PYOMO_AVAILABLE:
        raise ImportError("Install Pyomo: pip install desdeo-brb[pyomo]")


def _smooth_min(a, b, eps: float = 1e-4):
    """Smooth approximation of ``min(a, b)`` using a sqrt softening."""
    return 0.5 * (a + b - pyo.sqrt((a - b) ** 2 + eps))


def _smooth_max(a, b, eps: float = 1e-4):
    """Smooth approximation of ``max(a, b)`` using a sqrt softening."""
    return 0.5 * (a + b + pyo.sqrt((a - b) ** 2 + eps))


def _smooth_max_n(values: list, eps: float = 1e-4):
    """Smooth max over a list of values via chained pairwise smooth_max."""
    if len(values) == 1:
        return values[0]
    result = values[0]
    for v in values[1:]:
        result = _smooth_max(result, v, eps)
    return result


def build_pyomo_brb_model(
    brb_model: "BRBModel",
    X_train: np.ndarray,
    y_train: np.ndarray,
    fix_endpoints: bool = True,
    fix_endpoint_beliefs: bool = False,
    normalize_rule_weights: bool = True,
    optimize_referential_values: bool = True,
):
    """Build a Pyomo ConcreteModel encoding the BRB MSE objective.

    Args:
        brb_model: Source ``BRBModel``. Used for initial parameter values
            and structural metadata (n_rules, n_consequents, etc.).
        X_train: Training inputs, shape ``(n_samples, n_attributes)``.
        y_train: Training targets, shape ``(n_samples,)``.
        fix_endpoints: If ``True``, fix the first and last referential
            values for each attribute.
        fix_endpoint_beliefs: If ``True``, fix the belief degrees of rules
            at the boundary referential values.
        normalize_rule_weights: If ``True``, add a sum-to-1 equality
            constraint on rule weights.
        optimize_referential_values: If ``False``, precompute alphas with
            NumPy and treat them as constants (faster, no smooth error,
            but referential values cannot move). If ``True``, build the
            input transform symbolically using smooth min/max so that
            referential values are decision variables.

    Returns:
        A ``pyo.ConcreteModel`` with bounded variables, structural
        constraints, and an MSE objective.
    """
    _check_pyomo()

    rb = brb_model.rule_base
    n_rules = rb.n_rules
    n_consequents = rb.n_consequents
    n_attributes = rb.n_attributes
    n_samples = X_train.shape[0]

    rule_antecedent_indices = np.asarray(rb.rule_antecedent_indices)
    ref_value_lengths = [len(rv) for rv in rb.precedent_referential_values]
    consequent_rv = np.asarray(rb.consequent_referential_values)

    m = pyo.ConcreteModel("BRB_MSE")

    # Index sets
    m.SAMPLES = pyo.RangeSet(0, n_samples - 1)
    m.RULES = pyo.RangeSet(0, n_rules - 1)
    m.CONSEQUENTS = pyo.RangeSet(0, n_consequents - 1)
    m.ATTRIBUTES = pyo.RangeSet(0, n_attributes - 1)

    # Decision variables

    # Belief degrees: (n_rules, n_consequents) in [0, 1]
    bd_init = rb.belief_degrees

    def _bd_init(_m, k, n):
        return float(bd_init[k, n])

    m.beta = pyo.Var(m.RULES, m.CONSEQUENTS, bounds=(0.0, 1.0), initialize=_bd_init)

    # Rule weights: (n_rules,) in [0, 1]
    rw_init = rb.rule_weights

    def _rw_init(_m, k):
        return float(rw_init[k])

    m.theta = pyo.Var(m.RULES, bounds=(0.0, 1.0), initialize=_rw_init)

    # Attribute weights: (n_rules, n_attributes) in [eps, 10]
    aw_init = rb.attribute_weights

    def _aw_init(_m, k, i):
        return float(max(aw_init[k, i], 1e-6))

    m.delta = pyo.Var(m.RULES, m.ATTRIBUTES, bounds=(1e-6, 10.0), initialize=_aw_init)

    # Referential values: indexed by (attribute_idx, ref_value_idx)
    rv_indices = [
        (i, j) for i in range(n_attributes) for j in range(ref_value_lengths[i])
    ]

    def _rv_init(_m, i, j):
        return float(rb.precedent_referential_values[i][j])

    if optimize_referential_values:

        def _rv_bounds(_m, i, j):
            return (
                float(rb.precedent_referential_values[i][0]),
                float(rb.precedent_referential_values[i][-1]),
            )

        m.A = pyo.Var(rv_indices, initialize=_rv_init, bounds=_rv_bounds)
        if fix_endpoints:
            for i in range(n_attributes):
                length = ref_value_lengths[i]
                m.A[(i, 0)].fix(float(rb.precedent_referential_values[i][0]))
                m.A[(i, length - 1)].fix(float(rb.precedent_referential_values[i][-1]))
    else:
        m.A = pyo.Param(rv_indices, initialize=_rv_init, mutable=False)

    # Apply boundary belief fixing
    if fix_endpoint_beliefs:
        boundary_mask = brb_model._boundary_rule_mask()
        for k in range(n_rules):
            if boundary_mask[k]:
                for n in range(n_consequents):
                    m.beta[(k, n)].fix(float(bd_init[k, n]))

    # Structural constraints

    boundary_mask = brb_model._boundary_rule_mask() if fix_endpoint_beliefs else None

    def _bd_sum_rule(_m, k):
        if boundary_mask is not None and boundary_mask[k]:
            return pyo.Constraint.Skip
        return sum(_m.beta[k, n] for n in _m.CONSEQUENTS) == 1.0

    m.bd_sum = pyo.Constraint(m.RULES, rule=_bd_sum_rule)

    if normalize_rule_weights:

        def _rw_sum_rule(_m):
            return sum(_m.theta[k] for k in _m.RULES) == 1.0

        m.rw_sum = pyo.Constraint(rule=_rw_sum_rule)

    if optimize_referential_values:
        order_indices = [
            (i, j) for i in range(n_attributes) for j in range(ref_value_lengths[i] - 1)
        ]

        def _order_rule(_m, i, j):
            return _m.A[i, j] + 1e-6 <= _m.A[i, j + 1]

        m.rv_order = pyo.Constraint(order_indices, rule=_order_rule)

    # Build alpha[s, i, j]
    # Either precomputed (Path A) or symbolic (Path B).
    alpha: dict = {}

    if not optimize_referential_values:
        # Path A: compute alphas with NumPy and store as Python floats.
        rvs_list = [
            np.asarray(rb.precedent_referential_values[i]) for i in range(n_attributes)
        ]
        alphas_np = input_transform(X_train, rvs_list)
        for s in range(n_samples):
            for i in range(n_attributes):
                for j in range(ref_value_lengths[i]):
                    alpha[(s, i, j)] = float(alphas_np[i][s, j])
    else:
        # Path B: build symbolic alphas with smooth min/max.
        for s in range(n_samples):
            for i in range(n_attributes):
                length = ref_value_lengths[i]
                h_raw = float(X_train[s, i])
                # Clamp h to the [first, last] referential value range
                h = _smooth_min(_smooth_max(h_raw, m.A[i, 0]), m.A[i, length - 1])

                for j in range(length):
                    if length == 1:
                        alpha[(s, i, j)] = 1.0
                        continue

                    if j == 0:
                        # Boundary: only the right (forward) part
                        right = (m.A[i, j + 1] - h) / (m.A[i, j + 1] - m.A[i, j])
                        alpha[(s, i, j)] = _smooth_max(_smooth_min(right, 1.0), 0.0)
                    elif j == length - 1:
                        # Boundary: only the left (backward) part
                        left = (h - m.A[i, j - 1]) / (m.A[i, j] - m.A[i, j - 1])
                        alpha[(s, i, j)] = _smooth_max(_smooth_min(left, 1.0), 0.0)
                    else:
                        left = (h - m.A[i, j - 1]) / (m.A[i, j] - m.A[i, j - 1])
                        right = (m.A[i, j + 1] - h) / (m.A[i, j + 1] - m.A[i, j])
                        alpha[(s, i, j)] = _smooth_max(_smooth_min(left, right), 0.0)

    # Build activation weights w[s, k] (Eq. A-3)

    # delta_bar[k, i] = delta[k, i] / max_i(delta[k, i])  via smooth_max_n
    delta_bar: dict = {}
    for k in range(n_rules):
        delta_max_k = _smooth_max_n([m.delta[k, i] for i in range(n_attributes)])
        for i in range(n_attributes):
            delta_bar[(k, i)] = m.delta[k, i] / delta_max_k

    # match_prod[s, k] = prod_i(alpha[s, i, j_k_i] ** delta_bar[k, i])
    # In Path A, if any alpha is exactly 0 the rule's match_prod is 0
    # (matching the NumPy hard-mask behaviour). In Path B, alphas are
    # symbolic and we clamp to a tiny positive to avoid log(0).
    eps_alpha = 1e-8
    match_prod: dict = {}
    for s in range(n_samples):
        for k in range(n_rules):
            # Gather alphas for this rule's antecedents
            alphas_for_rule = [
                alpha[(s, i, int(rule_antecedent_indices[k, i]))]
                for i in range(n_attributes)
            ]

            if not optimize_referential_values:
                # Path A: alphas are constants. Replicate the NumPy mask:
                # if any alpha is 0, the rule contributes 0 to match_prod.
                if any(a == 0.0 for a in alphas_for_rule):
                    match_prod[(s, k)] = 0.0
                    continue

            # Build the symbolic product term-by-term
            terms = []
            for i in range(n_attributes):
                a = alphas_for_rule[i]
                if isinstance(a, (int, float)):
                    # Constant base, variable exponent
                    a_safe = max(float(a), eps_alpha)
                    if a_safe >= 1.0:
                        # log(>=1) = 0 → exp(0) = 1; skip
                        terms.append(1.0)
                    else:
                        log_a = float(np.log(a_safe))
                        terms.append(pyo.exp(delta_bar[(k, i)] * log_a))
                else:
                    # Symbolic base
                    terms.append(pyo.exp(delta_bar[(k, i)] * pyo.log(a + eps_alpha)))

            match_prod[(s, k)] = reduce(operator.mul, terms)

    # w[s, k] = theta[k] * match_prod[s, k] / (sum_l(theta[l] * match_prod[s, l]) + eps)
    w: dict = {}
    eps_w = 1e-12
    for s in range(n_samples):
        denom_terms = [m.theta[ll] * match_prod[(s, ll)] for ll in range(n_rules)]
        denom = sum(denom_terms) + eps_w
        for k in range(n_rules):
            w[(s, k)] = m.theta[k] * match_prod[(s, k)] / denom

    # Combined belief degrees beta_combined[s, n] (Eq. A-15)

    # beta_sum[k] = sum_n(beta[k, n])
    beta_sum: dict = {}
    for k in range(n_rules):
        beta_sum[k] = sum(m.beta[k, n] for n in m.CONSEQUENTS)

    eps_combined = 1e-12
    y_pred: dict = {}

    for s in range(n_samples):
        # prod_term[n] = prod_k(w[k] * beta[k, n] + 1 - w[k] * beta_sum[k])
        prod_terms_per_n = []
        for n in range(n_consequents):
            terms = [
                w[(s, k)] * m.beta[k, n] + 1.0 - w[(s, k)] * beta_sum[k]
                for k in range(n_rules)
            ]
            prod_terms_per_n.append(reduce(operator.mul, terms))

        # right_prod = prod_k(1 - w[k] * beta_sum[k])
        right_prod_terms = [1.0 - w[(s, k)] * beta_sum[k] for k in range(n_rules)]
        right_prod = reduce(operator.mul, right_prod_terms)

        # prod_one_minus_w = prod_k(1 - w[k])
        one_minus_w_terms = [1.0 - w[(s, k)] for k in range(n_rules)]
        prod_one_minus_w = reduce(operator.mul, one_minus_w_terms)

        # beta_combined[n] = (prod_term[n] - right_prod) / denom
        denom = (
            sum(prod_terms_per_n)
            - (n_consequents - 1) * right_prod
            - prod_one_minus_w
            + eps_combined
        )

        # y_pred[s] = sum_n(D[n] * beta_combined[s, n])
        y_pred[s] = sum(
            float(consequent_rv[n]) * (prod_terms_per_n[n] - right_prod) / denom
            for n in range(n_consequents)
        )

    # Objective: MSE

    def _obj_rule(_m):
        return (1.0 / n_samples) * sum(
            (y_pred[s] - float(y_train[s])) ** 2 for s in range(n_samples)
        )

    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    # Metadata for downstream extraction
    m._brb_n_rules = n_rules
    m._brb_n_consequents = n_consequents
    m._brb_n_attributes = n_attributes
    m._brb_ref_value_lengths = ref_value_lengths
    m._brb_rule_antecedent_indices = rule_antecedent_indices
    m._brb_consequent_referential_values = consequent_rv

    return m
