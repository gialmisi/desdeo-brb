"""Integration tests for the Pyomo backend."""

import numpy as np
import pytest

try:
    import pyomo.environ as pyo

    from desdeo_brb.pyomo_backend import PYOMO_AVAILABLE, build_pyomo_brb_model

    _solver = pyo.SolverFactory("ipopt")
    IPOPT_AVAILABLE = bool(_solver.available())
except Exception:  # pragma: no cover - import guard
    PYOMO_AVAILABLE = False
    IPOPT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (PYOMO_AVAILABLE and IPOPT_AVAILABLE),
    reason="Pyomo or IPOPT not available",
)


from desdeo_brb import BRBModel  # noqa: E402


def test_pyomo_model_builds():
    """Verify build_pyomo_brb_model returns a ConcreteModel without error."""
    prv = [np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 0.5, 1.0])
    brb = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] / 2.0)

    X = np.array([[0.5], [1.0], [1.5]])
    y = np.array([0.25, 0.5, 0.75])

    m = build_pyomo_brb_model(brb, X, y)
    assert isinstance(m, pyo.ConcreteModel)
    assert hasattr(m, "obj")
    assert hasattr(m, "beta")
    assert hasattr(m, "theta")
    assert hasattr(m, "delta")
    assert hasattr(m, "A")


def test_pyomo_matches_numpy_fixed_rv():
    """Pyomo expressions at initial values must match NumPy inference (Path A)."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 1.0, 2.0, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    brb = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] * np.sin(x[0] ** 2))

    X = np.array([[0.5], [1.5], [2.5]])
    y = np.array([f(0.5), f(1.5), f(2.5)])

    y_numpy = brb.predict_values(X)
    numpy_mse = float(np.mean((y - y_numpy) ** 2))

    m = build_pyomo_brb_model(brb, X, y, optimize_referential_values=False)
    pyomo_mse = float(pyo.value(m.obj))

    np.testing.assert_allclose(
        pyomo_mse,
        numpy_mse,
        rtol=1e-4,
        err_msg="Pyomo objective doesn't match NumPy MSE",
    )


def test_pyomo_matches_numpy_variable_rv():
    """Path B (symbolic input transform) matches NumPy within smooth-approx tol."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 1.0, 2.0, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    brb = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] * np.sin(x[0] ** 2))

    X = np.array([[0.5], [1.5], [2.5]])
    y = np.array([f(0.5), f(1.5), f(2.5)])

    y_numpy = brb.predict_values(X)
    numpy_mse = float(np.mean((y - y_numpy) ** 2))

    m = build_pyomo_brb_model(brb, X, y, optimize_referential_values=True)
    pyomo_mse = float(pyo.value(m.obj))

    np.testing.assert_allclose(
        pyomo_mse,
        numpy_mse,
        rtol=0.05,
        err_msg="Pyomo (variable RV) objective doesn't match NumPy MSE",
    )


def test_pyomo_solves_trivial():
    """Solve f(x)=x on [0, 2] with IPOPT and verify a small final MSE."""
    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0])]
    crv = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    brb = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0])

    # Sample count is kept small because the symbolic ER expression tree
    # makes IPOPT's per-iteration cost scale with n_samples.
    X = np.linspace(0, 2, 10).reshape(-1, 1)
    y = X[:, 0]

    m = build_pyomo_brb_model(brb, X, y, optimize_referential_values=False)
    initial_mse = float(pyo.value(m.obj))

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)

    assert str(result.solver.termination_condition) == "optimal"
    final_mse = float(pyo.value(m.obj))
    assert final_mse <= initial_mse + 1e-9, (
        f"MSE did not improve: {initial_mse} -> {final_mse}"
    )
    assert final_mse < 0.01, f"Final MSE too high: {final_mse}"


def test_pyomo_multi_attribute_builds():
    """Pyomo model builds and solves for a 2-attribute BRB (f(x1,x2)=x1+x2)."""
    prv = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    brb = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] + x[1])

    # Use a small training set: this test only verifies that the
    # multi-attribute pipeline builds and that IPOPT terminates. The
    # symbolic ER product expression scales superlinearly with sample
    # count, so keep it tight.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, size=(6, 2))
    y = X[:, 0] + X[:, 1]

    m = build_pyomo_brb_model(brb, X, y, optimize_referential_values=False)
    assert isinstance(m, pyo.ConcreteModel)

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert str(result.solver.termination_condition) == "optimal"
