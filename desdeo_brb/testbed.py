import numpy as np


def belief_distribution(xs: np.ndarray, hs: np.ndarray) -> np.ndarray:
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
    res = np.zeros((x.shape[0],) + hs.shape)
    hs_fwd = np.roll(hs, 1)
    hs_bck = np.roll(hs, -1)
    zeros = np.zeros((x[:, 0].shape[0],) + hs[0].shape)

    for i in range(x.shape[1]):
        # hs_fwd = np.roll(hs[i], 1)
        # hs_bck = np.roll(hs[i], -1)

        # print(x[:, i].reshape(-1, 1))
        # print(hs_fwd)
        # a = (hs_fwd - x[:, i].reshape(-1, 1))
        # b = (hs_fwd - hs[i])
        # print(np.fmax((a / b), zeros))

        max1 = np.fmax(
            (hs_fwd[i] - x[:, i].reshape(-1, 1))
            / (hs_fwd[i] - hs[i].reshape(1, -1)),
            zeros,
        )
        max1 = np.where((max1 >= 0), max1, 0)

        max2 = np.fmax(
            (x[:, i].reshape(-1, 1) - hs_bck[i]) / (hs[i] - hs_bck[i]), zeros
        )
        max2 = np.where((max2 >= 0), max2, 0)

        min_term = np.fmin(max1, max2)
        res[:, i, :] = np.where(min_term <= 1, min_term, 1)

    return res


def cartesian_product(mat) -> np.ndarray:
        """Takes the element wise cartesian product of the vectors contained in a list.

        Arguments:
            mat (List[np.ndarray]): A list with 1D numpy arrays used to form
            the element wise cartesian product.

        Returns:
            (np.ndarray): The cartesian product as a 2D array. It's dimensions
            will be (product of the lengths of the vectors contained in mat,
            length of the longest array in mat)

        """
        prods = np.zeros((mat.shape[0], mat.shape[2]**mat.shape[1], mat.shape[1]))
        print(prods.shape)


        for i in range(len(mat)):
            n_referential_sets = len(mat[i])
            print(n_referential_sets)

            if n_referential_sets == 1:
                # if math has just one set of referential values, return it
                prod = mat[i][0].reshape((-1, 1))

            else:
                grid = np.array(np.meshgrid(*mat[i]))
                prod = grid.T.reshape(-1, n_referential_sets)

            prods[i] = prod

        return prods
    


if __name__ == "__main__":
        # 1d case
    a = np.array([[0.25], [0.33]])

    hs = np.array([[0, 0.1, 0.2, 0.3]])

    res = new_belief_distribution(a, hs)
    print(cartesian_product(res))
