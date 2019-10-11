import numpy as np
from typing import List, Callable, Optional
import matplotlib.pyplot as plt


class BRB:
    """Constructs a trainable BRB model.

    Arguments:
        precedents (np.ndarray): 2D dimensional array with the ith row
        representing the referential values for the ith attribute in the input
        of the model.
        consequents (np.ndarray): 1D array of the possible values for the
        output of the model.
        rule_weights (np.ndarray, optional): 1D array with the ith element
        representing the initial invidual rule weight for the ith rule.
        attr_weights (np.ndarray, optional): 2D array with the jth column
        represent the initial invidual attribute weight in the k (row) rule.
        bre_m (np.ndarray, optional): 2D array with the initial belief rule
        expressions.
        f (Callable, optional): The real system.

    Attributes:

    """

    def __init__(
        self,
        precedents: np.ndarray,
        consequents: np.ndarray,
        rule_weights: Optional[np.ndarray] = None,
        attr_weights: Optional[np.ndarray] = None,
        bre_m: Optional[np.ndarray] = None,
        f: Optional[Callable] = lambda x: x * np.sin(x ** 2),
    ):
        self.precedents = precedents
        self.consequents = consequents
        # TODO check these and skip initialization of tule base
        self.rule_weights = rule_weights
        self.attr_weights = attr_weights
        self.bre_m = bre_m
        self.f = f

        self.thetas, self.deltas, self.bre_m = self.construct_inital_rule_base(
            self.precedents, self.consequents, self.f
        )

    def predict(
        self, x: np.ndarray, utility: Optional[Callable] = lambda x: x
    ) -> float:
        alphas = self.belief_distribution(x, self.precedents)
        rules = self.cartesian_product(alphas)
        ws = self.calculate_activation_weights(rules, self.thetas, self.deltas)
        betas = self.calculate_combined_belief_degrees(self.bre_m, ws)
        y = np.sum(utility(consequents) * betas)

        return y

    def construct_inital_rule_base(
        self, precedents: np.ndarray, consequents: np.ndarray, f: Callable
    ):
        rules = self.cartesian_product(precedents)
        thetas = np.ones((len(rules), 1)) / len(rules)
        deltas = np.array([1])
        bre_m = self.construct_initial_belief_rule_exp_matrix(
            precedents, consequents, f
        )

        return thetas, deltas, bre_m

    def belief_distribution(
        self, xs: np.ndarray, hs: np.ndarray
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
            # using the modulus, we achieve a circular indexinf for the array
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

    def cartesian_product(self, mat: List[np.ndarray]) -> np.ndarray:
        """Takes the element wise cartesian product of the vectors contained in a list.

        Arguments:
            mat (List[np.ndarray]): A list with 1D numpy arrays used to form
            the element wise cartesian product.

        Returns:
            (np.ndarray): The cartesian product as a 2D array. It's dimensions
            will be (product of the lengths of the vectors contained in mat,
            length of the longest array in mat)

        """
        n_referential_sets = len(mat)
        if n_referential_sets == 1:
            # if math has just one set of referential values, return it
            return mat[0].reshape((-1, 1))

        grid = np.array(np.meshgrid(*mat))
        prod = grid.T.reshape(-1, n_referential_sets)

        return prod

    def calculate_activation_weights(
        self, alphas: np.ndarray, thetas: np.ndarray, deltas: np.ndarray
    ):
        """Calculate the rule activation weights

        Arguments:
            alphas (np.ndarray): 2D array with the referential values. Each row
            represents a rule with the ith element representing the invidual
            matching degree, i.e. the belief degree that the antecedent x_i is
            assessed to the A_i value in the kth rule.
            thetas (np.ndarray): 1D array with the individual rule weights.
            deltas (np.ndarray): 2D or 1D array with the rows representing a
            rule and the ith element representing the ith attribute's
            weight. Note that the individual weights can vary between rules. IF
            thetas is a 1D array, the same attribute weights are assumed in all
            rules.

        Returns:
            (np.ndarray): 1D array containing the rule activation weights for
            each rule.

        Note:
            An AND connective rule is assumed between rules.

        TODO:
            The connective should be specified as an argument and not hard
            coded.

        """
        # number of total rules
        n_rules = len(alphas)
        unnormalized_ws = np.zeros(n_rules)

        # if the attribute weights are given just for the first rule, copy them
        # for each rule.
        if deltas.ndim == 1:
            deltas = np.tile(deltas, (n_rules, 1))

        # compute the normalized attribute weights (None just adds an extra
        # axis so that broadcasting works)
        normed_deltas = deltas / np.max(deltas, axis=1)[:, None]

        for k in range(n_rules):
            # the + 0j is to convert the alpha values to complex numbers so
            # that numpy doesn't complain about rising negative real numbers to
            # fractional powers.
            unnormalized_ws[k] = thetas[k] * np.prod(
                ((alphas[k, :] + 0j) ** normed_deltas[k, :]).real
            )

        # normalize the activation weights
        ws = unnormalized_ws / np.sum(unnormalized_ws)

        return ws

    def construct_initial_belief_rule_exp_matrix(
        self, refs: np.ndarray, consequents: np.ndarray, fun: Callable
    ):
        """Calculate the initial belief rule degree matrix using a known mapping
        from input to output on the referential value set. Each row should
        represent the output of each rule, when only said rule is activated.

        Attributes:
            refs (np.ndarray): 2D dimensional array of referential values. The
            ith row represents the possible referential values the ith
            attribute can take.
            consequents (np.ndarray): 1D array with the possible referential
            values the output of the BRB model may take. The output is assumed
            to be a scalar.
            fun (Callable): A mapping from a vector of attributes to a single
            scalar value.

        Returns:
            (np.ndarray): A matrix with the ith row representing the ith rule's
            belief degrees.

        Note:
            Works only with scalar valued functions at the moment.

        """
        ys = fun(refs)
        return self.belief_distribution(ys[0], consequents)

    def calculate_combined_belief_degrees(
        self, bre: np.ndarray, ws: np.ndarray
    ):
        """Calculate the combined belief degree of a BRB system defined by a belief
        rule expression matrix and rule activation weights.

        Attributes:
            bre (np.ndarray): A brelief rule expression matrix with the ith row
            representing the ith rule's belief degreees.
            ws (np.ndarray): 1D array with the ith element representing the ith
            rule's activation weight.

        Returns:
            (np.ndarray): 1D array of the combined belief degrees.

        """
        # total number of rules in the BRB
        n = bre.shape[1]
        # recurrent calculations, done just once
        wb_sum = ws * np.sum(bre, axis=1)
        wb_prod = ws * bre.T
        # calculate each of the products present in the expression for beta_n
        prod_1 = np.prod(wb_prod + 1 - wb_sum, axis=1)
        prod_2 = np.prod(1 - wb_sum)
        prod_3 = np.sum(prod_1)
        prod_4 = (n - 1) * prod_2
        prod_5 = np.prod(1 - ws)

        beta = (prod_1 - prod_2) / (prod_3 - prod_4 - prod_5)

        return beta


# Testing
def f(x):
    return x * np.sin(x ** 2)


refs = np.array([[0, 0.5, 1, 1.5, 2, 2.5, 3]])
consequents = np.array([[-2.5, -1, 1, 2, 3]])

# Train an initial model
brb = BRB(refs, consequents, f=f)
print(brb.bre_m)

# Test it
xs = np.linspace(0, 3, 50)
ys = np.zeros(50)

for i in range(50):
    ys[i] = brb.predict(np.array([xs[i]]))

print(ys)
plt.plot(xs, ys, label="BRB prediciton")
plt.plot(xs, f(xs), label="x*sin(x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim([-3, 3])
plt.title("Untrained BRB prediction vs actual value")
plt.legend()
plt.show()
