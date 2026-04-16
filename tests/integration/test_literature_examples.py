"""Integration tests based on published BRB application examples.

References:
- Chen et al. (2011), Appendix D / Table D.1 — pipeline leak detection
- Xu et al. (2007) — original pipeline leak detection expert system
- Yang et al. (2007) — expert-then-train BRB workflow
- Thesis Section 3.7 — cos(sqrt(x)) / (2 + cos(x)^2) example
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from desdeo_brb.brb import BRBModel
from desdeo_brb.models import RuleBase
from desdeo_brb.utils import build_rule_antecedent_indices

# Fixtures: Pipeline leak detection (Chen et al. 2011, Table D.1)

# Trained referential values from Chen et al. 2011 Section 5.2.3
FLOW_DIFF_RV = np.array([-10.0, -4.1, -2.8, -1.79, -0.79, 0.25, 2.0, 3.0])
PRESSURE_DIFF_RV = np.array([-0.01, -0.008, -0.005, 0.003, 0.0058, 0.008, 0.01])
LEAK_SIZE_RV = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

# Expert-defined belief degrees from Table D.1
# 56 rules (8 FlowDiff x 7 PressureDiff), 5 consequents [0, 2, 4, 6, 8]
EXPERT_BELIEFS = np.array(
    [
        # FlowDiff = NL (most negative, index 0)
        [0.00, 0.00, 0.00, 0.00, 1.00],  # R1:  NL, NL
        [0.00, 0.00, 0.00, 0.30, 0.70],  # R2:  NL, NM
        [0.00, 0.00, 0.20, 0.80, 0.00],  # R3:  NL, NS
        [0.00, 0.00, 0.80, 0.20, 0.00],  # R4:  NL, Z
        [0.65, 0.35, 0.00, 0.00, 0.00],  # R5:  NL, PS
        [0.85, 0.15, 0.00, 0.00, 0.00],  # R6:  NL, PM
        [0.95, 0.05, 0.00, 0.00, 0.00],  # R7:  NL, PL
        # FlowDiff = NM (index 1)
        [0.00, 0.00, 0.10, 0.90, 0.00],  # R8:  NM, NL
        [0.00, 0.00, 0.70, 0.30, 0.00],  # R9:  NM, NM
        [0.00, 0.70, 0.30, 0.00, 0.00],  # R10: NM, NS
        [0.00, 0.90, 0.10, 0.00, 0.00],  # R11: NM, Z
        [0.80, 0.20, 0.00, 0.00, 0.00],  # R12: NM, PS
        [0.90, 0.10, 0.00, 0.00, 0.00],  # R13: NM, PM
        [0.99, 0.01, 0.00, 0.00, 0.00],  # R14: NM, PL
        # FlowDiff = NS (index 2)
        [0.00, 0.00, 0.40, 0.60, 0.00],  # R15: NS, NL
        [0.00, 0.00, 0.80, 0.20, 0.00],  # R16: NS, NM
        [0.00, 0.30, 0.60, 0.10, 0.00],  # R17: NS, NS
        [0.10, 0.80, 0.10, 0.00, 0.00],  # R18: NS, Z
        [0.90, 0.10, 0.00, 0.00, 0.00],  # R19: NS, PS
        [0.95, 0.05, 0.00, 0.00, 0.00],  # R20: NS, PM
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R21: NS, PL
        # FlowDiff = Z (index 3)
        [0.00, 0.00, 0.50, 0.50, 0.00],  # R22: Z, NL
        [0.00, 0.10, 0.80, 0.10, 0.00],  # R23: Z, NM
        [0.00, 0.80, 0.20, 0.00, 0.00],  # R24: Z, NS
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R25: Z, Z
        [0.95, 0.05, 0.00, 0.00, 0.00],  # R26: Z, PS
        [0.98, 0.02, 0.00, 0.00, 0.00],  # R27: Z, PM
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R28: Z, PL
        # FlowDiff = PS (index 4)
        [0.00, 0.00, 0.60, 0.40, 0.00],  # R29: PS, NL
        [0.00, 0.20, 0.60, 0.20, 0.00],  # R30: PS, NM
        [0.10, 0.60, 0.30, 0.00, 0.00],  # R31: PS, NS
        [0.90, 0.10, 0.00, 0.00, 0.00],  # R32: PS, Z
        [0.95, 0.05, 0.00, 0.00, 0.00],  # R33: PS, PS
        [0.99, 0.01, 0.00, 0.00, 0.00],  # R34: PS, PM
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R35: PS, PL
        # FlowDiff = PM (index 5)
        [0.00, 0.00, 0.60, 0.40, 0.00],  # R36: PM, NL
        [0.00, 0.20, 0.70, 0.10, 0.00],  # R37: PM, NM
        [0.00, 0.70, 0.30, 0.00, 0.00],  # R38: PM, NS
        [0.95, 0.05, 0.00, 0.00, 0.00],  # R39: PM, Z
        [0.99, 0.01, 0.00, 0.00, 0.00],  # R40: PM, PS
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R41: PM, PM
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R42: PM, PL
        # FlowDiff = PL (index 6)
        [0.00, 0.10, 0.70, 0.20, 0.00],  # R43: PL, NL
        [0.00, 0.30, 0.60, 0.10, 0.00],  # R44: PL, NM
        [0.10, 0.70, 0.20, 0.00, 0.00],  # R45: PL, NS
        [0.98, 0.02, 0.00, 0.00, 0.00],  # R46: PL, Z
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R47: PL, PS
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R48: PL, PM
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R49: PL, PL
        # FlowDiff = PVL (most positive, index 7)
        [0.00, 0.10, 0.80, 0.10, 0.00],  # R50: PVL, NL
        [0.00, 0.30, 0.70, 0.00, 0.00],  # R51: PVL, NM
        [0.10, 0.80, 0.10, 0.00, 0.00],  # R52: PVL, NS
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R53: PVL, Z
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R54: PVL, PS
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R55: PVL, PM
        [1.00, 0.00, 0.00, 0.00, 0.00],  # R56: PVL, PL
    ]
)


def _make_pipeline_model() -> BRBModel:
    """Construct the pipeline leak detection BRBModel from expert knowledge."""
    prv = [FLOW_DIFF_RV, PRESSURE_DIFF_RV]
    n_rules = 56
    rule_antecedent_indices = build_rule_antecedent_indices(prv)

    rule_base = RuleBase(
        precedent_referential_values=prv,
        consequent_referential_values=LEAK_SIZE_RV,
        belief_degrees=EXPERT_BELIEFS,
        rule_weights=np.full(n_rules, 1.0 / n_rules),
        attribute_weights=np.ones((n_rules, 2)),
        rule_antecedent_indices=rule_antecedent_indices,
    )

    return BRBModel(prv, LEAK_SIZE_RV, rule_base=rule_base)


# Pipeline leak detection (Chen et al. 2011)


@pytest.mark.pipeline
def test_pipeline_expert_rule_base_construction():
    """Verify that a BRB with 8x7 varying-length referential values
    and 56 expert-defined rules can be constructed."""
    model = _make_pipeline_model()
    rb = model.rule_base

    assert rb.n_rules == 56
    assert rb.n_attributes == 2
    assert rb.n_consequents == 5
    assert len(rb.precedent_referential_values[0]) == 8
    assert len(rb.precedent_referential_values[1]) == 7


@pytest.mark.pipeline
def test_pipeline_inference_at_referential_values():
    """Verify inference at exact rule antecedent values produces outputs
    consistent with expert beliefs."""
    model = _make_pipeline_model()

    # Rule 1: FlowDiff=-10, PressureDiff=-0.01 -> beliefs [0,0,0,0,1] -> output=8.0
    X_r1 = np.array([[-10.0, -0.01]])
    result_r1 = model.predict(X_r1)
    assert_allclose(result_r1.output, [8.0], atol=0.1)

    # Rule 56: FlowDiff=3, PressureDiff=0.01 -> beliefs [1,0,0,0,0] -> output=0.0
    X_r56 = np.array([[3.0, 0.01]])
    result_r56 = model.predict(X_r56)
    assert_allclose(result_r56.output, [0.0], atol=0.1)

    # Rule 25: FlowDiff=-1.79, PressureDiff=0.003 -> beliefs [1,0,0,0,0] -> output=0.0
    X_r25 = np.array([[-1.79, 0.003]])
    result_r25 = model.predict(X_r25)
    assert_allclose(result_r25.output, [0.0], atol=0.1)


@pytest.mark.pipeline
def test_pipeline_physical_consistency():
    """Verify that the expert rule base produces physically consistent outputs:
    large negative FlowDiff + negative PressureDiff -> high LeakSize,
    zero FlowDiff + zero PressureDiff -> low LeakSize."""
    model = _make_pipeline_model()

    # Scenario 1: Normal operation (near zero differences) -> small leak
    X_normal = np.array([[-0.79, 0.003]])
    y_normal = model.predict_values(X_normal)[0]

    # Scenario 2: Large leak (very negative flow, very negative pressure)
    X_leak = np.array([[-10.0, -0.01]])
    y_leak = model.predict_values(X_leak)[0]

    # Scenario 3: Intermediate
    X_mid = np.array([[-4.1, -0.005]])
    y_mid = model.predict_values(X_mid)[0]

    # Physical consistency: large leak > intermediate > normal
    assert y_leak > y_mid, f"Large leak ({y_leak:.2f}) should exceed intermediate ({y_mid:.2f})"
    assert y_mid > y_normal, f"Intermediate ({y_mid:.2f}) should exceed normal ({y_normal:.2f})"

    # Normal operation should be near zero
    assert y_normal < 2.0, f"Normal operation leak size too high: {y_normal:.2f}"
    # Large leak should be near maximum
    assert y_leak > 6.0, f"Large leak size too low: {y_leak:.2f}"

    # Monotonicity trend: as FlowDiff decreases (more negative), LeakSize
    # generally increases. BRB interpolation between expert rules doesn't
    # guarantee strict monotonicity at every point, so we check the overall
    # trend between well-separated points.
    X_positive = np.array([[3.0, -0.005]])
    X_negative = np.array([[-10.0, -0.005]])
    y_pos = model.predict_values(X_positive)[0]
    y_neg = model.predict_values(X_negative)[0]
    assert y_neg > y_pos, (
        f"Trend violated: FlowDiff=-10 output ({y_neg:.2f}) should exceed "
        f"FlowDiff=3 output ({y_pos:.2f})"
    )


@pytest.mark.pipeline
def test_pipeline_varying_length_referential_values():
    """Verify that the model correctly handles 8 vs 7 referential values
    for the two attributes."""
    model = _make_pipeline_model()

    X = np.array([[-5.0, -0.003], [0.0, 0.005], [2.5, 0.009]])
    result = model.predict(X)

    # input_belief_distributions shapes reflect the different lengths
    assert result.input_belief_distributions[0].shape == (3, 8)
    assert result.input_belief_distributions[1].shape == (3, 7)

    # activation weights cover all 56 rules
    assert result.activation_weights.shape == (3, 56)

    # combined belief degrees cover all 5 consequents
    assert result.combined_belief_degrees.shape == (3, 5)

    # outputs are finite
    assert result.output.shape == (3,)
    assert np.all(np.isfinite(result.output))


# f(x) = cos(sqrt(x)) / (2 + cos(x)^2) from thesis Section 3.7


def _f_cos_sqrt(x: np.ndarray) -> float:
    """f(x) = cos(sqrt(x)) / (2 + cos(x)^2)."""
    return float(np.cos(np.sqrt(x[0])) / (2.0 + np.cos(x[0]) ** 2))


def _f_cos_sqrt_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized version for evaluation."""
    return np.cos(np.sqrt(x)) / (2.0 + np.cos(x) ** 2)


def test_cos_sqrt_untrained_at_referential_values():
    """Verify that the untrained model reproduces the function at
    referential value points exactly."""
    prv = [np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])]
    # Function range: f(0)≈0.333, f(5)≈-0.297; cover full range
    crv = np.array([-0.3, -0.15, 0.0, 0.15, 0.35])

    model = BRBModel(prv, crv, initial_rule_fn=_f_cos_sqrt)

    # Predict at the referential values
    X_ref = prv[0].reshape(-1, 1)
    y_true = _f_cos_sqrt_vec(prv[0])
    y_pred = model.predict_values(X_ref)

    # The model initialized from the function should reproduce it at
    # referential values (within interpolation accuracy of the consequent grid)
    assert_allclose(y_pred, y_true, atol=0.05)


def test_cos_sqrt_training_improves():
    """Verify training reduces MSE for f(x) = cos(sqrt(x))/(2+cos(x)^2)."""
    prv = [np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])]
    crv = np.array([-0.3, -0.15, 0.0, 0.15, 0.35])

    model = BRBModel(prv, crv, initial_rule_fn=_f_cos_sqrt)

    rng = np.random.default_rng(77)
    X_train = rng.uniform(0, 5, size=(200, 1))
    y_train = _f_cos_sqrt_vec(X_train[:, 0])

    X_eval = np.linspace(0, 5, 200).reshape(-1, 1)
    y_eval = _f_cos_sqrt_vec(X_eval[:, 0])

    # Untrained MSE
    y_pred_untrained = model.predict_values(X_eval)
    mse_untrained = float(np.mean((y_eval - y_pred_untrained) ** 2))
    assert mse_untrained > 0

    # Train
    model.fit(X_train, y_train, fix_endpoints=True)

    # Trained MSE
    y_pred_trained = model.predict_values(X_eval)
    mse_trained = float(np.mean((y_eval - y_pred_trained) ** 2))

    assert mse_trained < mse_untrained, (
        f"Training did not improve MSE: {mse_trained:.6f} >= {mse_untrained:.6f}"
    )
    assert mse_trained < 0.02, f"Trained MSE too high: {mse_trained:.6f}"


# Test 3: Expert-defined initial beliefs then trained (Yang et al. 2007)


def test_expert_initial_beliefs_then_trained():
    """Simulate the expert-then-train workflow from Yang et al. 2007.
    Start with inaccurate uniform beliefs, train to match a known function."""
    prv = [np.array([0.0, 0.25, 0.5, 0.75, 1.0])]
    crv = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    n_rules = 5
    n_consequents = 5

    # Deliberately inaccurate initial beliefs: uniform across all consequents
    uniform_beliefs = np.full((n_rules, n_consequents), 1.0 / n_consequents)

    rule_base = RuleBase(
        precedent_referential_values=prv,
        consequent_referential_values=crv,
        belief_degrees=uniform_beliefs,
        rule_weights=np.full(n_rules, 1.0 / n_rules),
        attribute_weights=np.ones((n_rules, 1)),
        rule_antecedent_indices=build_rule_antecedent_indices(prv),
    )

    model = BRBModel(prv, crv, rule_base=rule_base)

    # Training data from f(x) = sin(2*pi*x)
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 1, size=(200, 1))
    y_train = np.sin(2 * np.pi * X_train[:, 0])

    X_eval = np.linspace(0, 1, 100).reshape(-1, 1)
    y_eval = np.sin(2 * np.pi * X_eval[:, 0])

    # Initial MSE (with uniform beliefs, predictions are ~0)
    y_pred_initial = model.predict_values(X_eval)
    mse_initial = float(np.mean((y_eval - y_pred_initial) ** 2))

    # Train
    model.fit(X_train, y_train, fix_endpoints=True)

    y_pred_trained = model.predict_values(X_eval)
    mse_trained = float(np.mean((y_eval - y_pred_trained) ** 2))

    # Training should substantially improve from the uniform baseline
    assert mse_trained < mse_initial, (
        f"Training did not improve: {mse_trained:.4f} >= {mse_initial:.4f}"
    )
    assert mse_trained < 0.1, f"Trained MSE too high: {mse_trained:.4f}"


def test_expert_initial_beliefs_structure_preserved():
    """After training from expert beliefs, verify the rule base
    still satisfies all structural constraints."""
    prv = [np.array([0.0, 0.25, 0.5, 0.75, 1.0])]
    crv = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    n_rules = 5
    n_consequents = 5
    uniform_beliefs = np.full((n_rules, n_consequents), 1.0 / n_consequents)

    rule_base = RuleBase(
        precedent_referential_values=prv,
        consequent_referential_values=crv,
        belief_degrees=uniform_beliefs,
        rule_weights=np.full(n_rules, 1.0 / n_rules),
        attribute_weights=np.ones((n_rules, 1)),
        rule_antecedent_indices=build_rule_antecedent_indices(prv),
    )

    model = BRBModel(prv, crv, rule_base=rule_base)

    X_train = np.linspace(0, 1, 100).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * X_train[:, 0])
    model.fit(X_train, y_train, fix_endpoints=True)

    rb = model.rule_base

    # Belief degree rows sum to 1
    row_sums = rb.belief_degrees.sum(axis=1)
    assert_allclose(row_sums, np.ones(n_rules), atol=1e-6)

    # Rule weights sum to 1
    assert_allclose(rb.rule_weights.sum(), 1.0, atol=1e-6)

    # Attribute weights non-negative
    assert np.all(rb.attribute_weights >= 0)

    # Referential values sorted ascending
    for rv in rb.precedent_referential_values:
        assert np.all(rv[:-1] <= rv[1:])
