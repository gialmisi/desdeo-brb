import numpy as np


def belief_distribution(
    xs: np.ndarray, hs: np.ndarray
) -> np.ndarray:
    """Given a set of referential values, transform x into corresponding belief
    distributions.

    Arguments:
        xs (np.narray): A 1D array with each element representing the i:th
        element of the input.
        hs (np.narray): A 2D array with the i:th row containing the
        referential values for the i:th element in the input variable.

    Returns:
        (np.ndarray): A 2D array with the i:th row containing the belief
        degrees
        for the i:th element in the input variable.

    TODO:
        Input assertion is absent

    """
    n_x = xs.shape[0]
    n_h = hs.shape[1]

    alphas = np.zeros((n_x, n_h))

    for n in range(n_h):
        # using the modulus, we achieve a circular indexing for the array
        alphas[:, n - 1 % n_h] = np.min(
            [
                np.max(
                    [
                        (hs[:, n] - xs) / (hs[:, n] - hs[:, n - 1 % n_h]),
                        np.zeros(n_x),
                    ],
                    axis=0,
                ),
                np.max(
                    [
                        (xs - hs[:, n - 2 % n_h])
                        / (hs[:, n - 1 % n_h] - hs[:, n - 2 % n_h]),
                        np.zeros(n_x),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )

    return alphas


def new_belief_distribution(x, hs):
    hs_fwd = np.roll(hs, 1)
    hs_bck = np.roll(hs, -1)
    zeros = np.zeros((x.shape[0], hs.shape[1]))

    max1 = np.fmax((hs_fwd - x) / (hs_fwd - hs), zeros)
    max1 = np.where((max1 >= 0), max1, 0)

    max2 = np.fmax((x - hs_bck) / (hs - hs_bck), zeros)
    max2 = np.where((max2 >= 0), max2, 0)

    min_term = np.fmin(max1, max2)
    return np.where(min_term <= 1, min_term, 1)


if __name__ == "__main__":
    # 1d case
    a = np.array([[0.25], [0.27]]).reshape(-1, 1)
    a = np.random.uniform(0, 0.3, (100000, 1))

    hs = np.array([[0, 0.1, 0.2, 0.3]])

    import timeit

    start = timeit.default_timer()
    res = new_belief_distribution(a, hs)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    start = timeit.default_timer()
    res_old = np.apply_along_axis(belief_distribution, 1, a, hs)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # 2d case
