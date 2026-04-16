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


def test_fit_ipopt_xsinx2():
    """Train f(x) = x sin(x^2) with IPOPT and verify good MSE."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    X_train = np.linspace(0, 3, 200).reshape(-1, 1)
    y_train = f(X_train[:, 0])

    model.fit(X_train, y_train, fix_endpoints=True, method="ipopt")

    X_eval = np.linspace(0, 3, 200).reshape(-1, 1)
    y_pred = model.predict_values(X_eval)
    y_true = f(X_eval[:, 0])
    mse = float(np.mean((y_true - y_pred) ** 2))

    assert mse < 0.05, f"IPOPT MSE too high: {mse}"


def test_fit_ipopt_endpoint_accuracy():
    """IPOPT should achieve reasonable endpoint accuracy.

    Like the scipy backends, IPOPT finds local minima where boundary
    rule belief degrees drift, so this test uses fix_endpoint_beliefs=True
    to preserve the initial good boundary beliefs.
    """

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    X_train = np.linspace(0, 3, 200).reshape(-1, 1)
    y_train = f(X_train[:, 0])
    model.fit(
        X_train,
        y_train,
        fix_endpoints=True,
        fix_endpoint_beliefs=True,
        method="ipopt",
    )

    y_end = model.predict_values(np.array([[3.0]]))
    error = float(abs(y_end[0] - f(3.0)))
    assert error < 0.2, f"IPOPT endpoint error: {error}"


def test_fit_ipopt_with_options():
    """Verify custom IPOPT options are accepted."""
    prv = [np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 0.5, 1.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] / 2.0)

    X = np.linspace(0, 2, 20).reshape(-1, 1)
    y = X[:, 0] / 2.0

    model.fit(
        X,
        y,
        method="ipopt",
        optimizer_options={"max_iter": 100, "tol": 1e-6},
    )


def test_fit_ipopt_multistart():
    """Verify n_restarts works with IPOPT."""

    def f(x):
        return x * np.sin(x**2)

    prv = [np.array([0.0, 1.0, 2.0, 3.0])]
    crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: f(x[0]))

    X_train = np.linspace(0, 3, 100).reshape(-1, 1)
    y_train = f(X_train[:, 0])

    model.fit(
        X_train, y_train, fix_endpoints=True, method="ipopt", n_restarts=3
    )

    y_pred = model.predict_values(X_train)
    mse = float(np.mean((y_train - y_pred) ** 2))
    assert mse < 0.5, f"IPOPT multistart MSE: {mse}"


def test_fit_ipopt_multi_attribute():
    """Train a 2-attribute model with IPOPT.

    Sample count is kept tight (3x3=9 points) because the symbolic ER
    product expression scales superlinearly with sample count and IPOPT
    becomes very slow with many samples.
    """

    def f(x):
        return float(x[0] + x[1])

    prv = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    model = BRBModel(prv, crv, initial_rule_fn=f)

    x1 = np.linspace(0, 2, 3)
    x2 = np.linspace(0, 2, 3)
    X1, X2 = np.meshgrid(x1, x2)
    X_train = np.column_stack([X1.ravel(), X2.ravel()])
    y_train = np.array([f(x) for x in X_train])

    model.fit(X_train, y_train, fix_endpoints=True, method="ipopt")

    y_pred = model.predict_values(X_train)
    mse = float(np.mean((y_train - y_pred) ** 2))
    assert mse < 0.5, f"IPOPT multi-attribute MSE: {mse}"


def test_update_from_pyomo():
    """Verify update_from_pyomo correctly extracts parameters."""
    prv = [np.array([0.0, 1.0, 2.0])]
    crv = np.array([0.0, 0.5, 1.0])
    model = BRBModel(prv, crv, initial_rule_fn=lambda x: x[0] / 2.0)

    X = np.linspace(0, 2, 20).reshape(-1, 1)
    y = X[:, 0] / 2.0

    m = build_pyomo_brb_model(model, X, y, optimize_referential_values=False)
    pyo.SolverFactory("ipopt").solve(m, tee=False)

    model.update_from_pyomo(m)

    y_pred = model.predict_values(X)
    mse = float(np.mean((y - y_pred) ** 2))
    assert mse < 0.1


def test_fit_ipopt_handles_error_gracefully():
    """IPOPT error should not crash; should use best solution found or warn."""

    def f(x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    rv_1d = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    prv = [rv_1d, rv_1d]
    crv = np.array([0.0, 200.0, 500.0, 1000.0, 2200.0])
    model = BRBModel(prv, crv, initial_rule_fn=f)

    # Small grid to keep the test fast
    x = np.linspace(-6, 6, 5)
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    X_train = np.column_stack([X1.ravel(), X2.ravel()])
    y_train = np.array([f(r) for r in X_train])

    y_before = model.predict_values(X_train)
    mse_before = float(np.mean((y_train - y_before) ** 2))

    # Should NOT raise even if IPOPT hits numerical issues
    model.fit(
        X_train,
        y_train,
        fix_endpoints=True,
        method="ipopt",
        optimizer_options={"max_iter": 50},
    )

    y_after = model.predict_values(X_train)
    mse_after = float(np.mean((y_train - y_after) ** 2))

    # Either improved or at worst unchanged (no crash)
    assert mse_after <= mse_before + 1.0


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
