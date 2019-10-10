from desdeo_brb import brb
import numpy as np


def test_input_transformation_1d():
    """Check the correct transformation of inputs for a simple inputs

    """
    x1 = np.array([0.6])
    x2 = np.array([1.25])
    x3 = np.array([1.75])
    hs = np.array([[0.5, 1, 1.5, 2]])

    alphas1 = brb.input_transformation(x1, hs)
    alphas2 = brb.input_transformation(x2, hs)
    alphas3 = brb.input_transformation(x3, hs)

    summed_x1 = np.sum(alphas1 * hs, axis=1)
    summed_x2 = np.sum(alphas2 * hs, axis=1)
    summed_x3 = np.sum(alphas3 * hs, axis=1)

    assert np.all(np.isclose(x1, summed_x1))
    assert np.all(np.isclose(x2, summed_x2))
    assert np.all(np.isclose(x3, summed_x3))


def test_input_transformation_1d_outside_value():
    """If the input is not feasible, the transformation should be non sensical

    """
    x = np.array([0.4])
    hs = np.array([[0.5, 1, 1.5, 2]])

    alphas = brb.input_transformation(x, hs)

    summed_x = np.sum(alphas * hs, axis=1)

    assert not np.all(np.isclose(x, summed_x))


def test_input_transformation_2d():
    """Test an input with multiple elements

    """
    xs = np.array([0.4, 0.5, 1.2])
    hs = np.array([
        [0.1, 0.5, 2],
        [0.5, 0.75, 1.5],
        [0, 0.5, 3],
    ])

    alphas = brb.input_transformation(xs, hs)

    summed_xs = np.sum(alphas * hs, axis=1)

    assert np.all(np.isclose(xs, summed_xs))


def test_cartesian_product_1d():
    """Test the simple case where the cartesian product is taken using just one
    set.

    """
    a = np.array([1, 2, 3])
    prod = brb.cartesian_product([a])

    assert np.all(np.isclose(a, prod))


def test_cartesian_product():
    """Test the case of the product of multiple sets.

    """
    a = np.array([1, 2, 3])
    b = np.array([-1, -2])
    c = np.array([10, 9, 8, 7])

    prod = brb.cartesian_product([a, b, c])

    # check the lengths
    assert len(prod) == len(a) * len(b) * len(c)

    # check the elements
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                assert [a[i], b[j], c[k]] in prod
