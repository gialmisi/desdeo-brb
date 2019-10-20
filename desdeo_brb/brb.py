import numpy as np
import pandas as pd
from typing import List, Callable, Optional
from collections import namedtuple
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import copy


class BRBResult(
    namedtuple(
        "BRBResult",
        [
            "precedents",
            "precedents_belief_degrees",
            "consequents",
            "consequent_belief_degrees",
        ],
    )
):
    def __str__(self):
        precedents_distributions = []
        for i in range(len(self.precedents)):
            distribution = []
            distribution.append(
                [
                    (a, b)
                    for a, b in zip(
                        self.precedents[i], self.precedents_belief_degrees[i]
                    )
                ]
            )
            precedents_distributions.append(distribution[0])
        return str(precedents_distributions)


class Trainables(
    namedtuple(
        "Trainables",
        [
            "flat_trainables",
            "n_attributes",
            "n_precedents",
            "n_rules",
            "n_consequents",
        ],
    )
):
    """Defines a named tuple containing information on the parameters of a
    BRB-model in a flattened format. This format is useful with various
    optimization routines.

    """

    def __str__(self):
        bre_m, thetas, deltas, precedents = BRB._unflatten_parameters(self)
        bre_m_df = pd.DataFrame(
            data=bre_m,
            index=[f"A_{i+1}" for i in range(bre_m.shape[0])],
            columns=[f"D_{i+1}" for i in range(bre_m.shape[1])],
        )
        thetas_df = pd.DataFrame(
            data=thetas,
            index=[f"θ_{i+1}" for i in range(thetas.shape[0])],
            columns=["Rule weight"],
        )
        deltas_df = pd.DataFrame(
            data=deltas,
            index=[f"δ_{i+1}" for i in range(deltas.shape[0])],
            columns=[f"x_{i+1}" for i in range(deltas.shape[1])],
        )
        precedents_df = pd.DataFrame(
            data=precedents,
            index=[f"A_{i+1}j" for i in range(precedents.shape[0])],
            columns=[f"A_i{i+1}" for i in range(precedents.shape[1])],
        )
        string = (
            "Belief rule expression matrix:\n{}\nRule weights:\n{}\n"
            "Attribute weights:\n{}\nPrecedents:\n{}"
        ).format(
            bre_m_df.round(2).to_string(),
            thetas_df.round(2).to_string(),
            deltas_df.round(2).to_string(),
            precedents_df.round(2).to_string(),
        )
        return string


class BRB:
    """Constructs a trainable BRB model.

    Arguments:
        precedents (np.ndarray): 2D dimensional array with the ith row
        representing the referential values for the ith attribute in the input
        of the model. The referential values should be in a hierarchial order
        where the first element of each row is the smallest element of the row,
        and the last element the greatest.
        consequents (np.ndarray): 1D array of the possible values for the
        output of the model.
        rule_weights (np.ndarray, optional): 1D array with the ith element
        representing the initial invidual rule weight for the ith rule.
        attr_weights (np.ndarray, optional): 2D array with the jth column
        represent the initial invidual attribute weight in the k (row) rule.
        bre_m (np.ndarray, optional): 2D array with the initial belief rule
        expressions.
        f (Callable, optional): A function representing the system to be
        modeled.  Can be just a mapping from x -> y. Used to construct an
        initial rule base.

    Attributes:
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
        f (Callable, optional): A function representing the system to be
        modeled.  Can be just a mapping from x -> y. Used to construct an
        initial rule base.
        trained (bool): Indicastes whether the model has been trained.

    """

    def __init__(
        self,
        precedents: np.ndarray,
        consequents: np.ndarray,
        rule_weights: Optional[np.ndarray] = None,
        attr_weights: Optional[np.ndarray] = None,
        bre_m: Optional[np.ndarray] = None,
        f: Optional[Callable] = lambda x: x * np.sin(x ** 2),
        utility: Optional[Callable] = lambda y: y,
    ):
        self.precedents = precedents
        self.consequents = consequents
        # TODO check these and skip initialization of tule base
        # self.rule_weights = rule_weights
        # self.attr_weights = attr_weights
        # self.bre_m = bre_m
        self.f = f
        self.utility = utility

        self.thetas, self.deltas, self.bre_m = self.construct_inital_rule_base(
            self.precedents, self.consequents, self.f
        )

        self.trained = False

    def __str__(self):
        trainables = self._flatten_parameters()
        return str(trainables)

    def predict(self, x: np.ndarray) -> BRBResult:
        """Predictt an outcome using the current parameters set in the
        BRB-model for an input.

        Arguments:
            x (np.ndarray): A 1D array with n elements, where n is the number
            of attributes the BRB system expects.

        Returns:

        """
        return self._predict(
            x,
            self.precedents,
            self.consequents,
            self.thetas,
            self.deltas,
            self.bre_m,
        )

    def _predict(
        self,
        x: np.ndarray,
        precedents: np.ndarray,
        consequents: np.ndarray,
        thetas: np.ndarray,
        deltas: np.ndarray,
        bre_m: np.ndarray,
    ) -> BRBResult:
        """Like predict, but the parameters of the BRB-model can be given
        explicitly. Used in for training. See BRB.predict and the top level
        documentation for this class for further details.

        """
        alphas = self.belief_distribution(x, precedents)
        rules = self.cartesian_product(alphas)
        ws = self.calculate_activation_weights(rules, thetas, deltas)
        betas = self.calculate_combined_belief_degrees(bre_m, ws)

        res = BRBResult(precedents, alphas, consequents, betas)
        return res

    def construct_inital_rule_base(
        self, precedents: np.ndarray, consequents: np.ndarray, f: Callable
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """Constructs the initial rule base using precedents and consequents,
        and a given mapping from input to expected output. See the top level
        documentation for this class for details on the attributes.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): The rule weights, the
            attibute weights in each rule and the belief rule expression
            matrix.

        """
        rules = self.cartesian_product(precedents)
        thetas = np.ones((len(rules), 1)) / len(rules)
        deltas = np.array([1])
        bre_m = self.construct_initial_belief_rule_exp_matrix(
            rules, consequents, f
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
        self, rules: np.ndarray, consequents: np.ndarray, fun: Callable
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
        ys = np.apply_along_axis(fun, 1, rules)
        return self.belief_distribution(ys, consequents)

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

    def train(
        self, xs: np.ndarray, ys: np.ndarray, _trainables: Trainables
    ) -> Trainables:
        """Train the BRB using input-output pairs. And update the model's parameters.

        Arguments:
            xs (np.ndarray): 2D array of the inputs with the n:th row being one
            sample with the elements representing the attribute values of that
            sample.
            ys (np.ndarray): 1D array of scalars with the n:th element
            representing the expected output for the n:th input in xs.
            _trainables (Trainables): A named tuple used to construct the
            optimization problem to train the BRB-model. Functions as an
            initial guess for the optimizer as well. See the documentation.

        Returns:
            Tainables: A named tuple containg the trained variables in a
            flattened format that define a trained BRB-model. If the
            optimization is not successfull, return the initial guess.

        """
        print("Training model...")
        trainables = copy(_trainables)

        # construct bounds
        # belief degrees between 0 and 1
        bre_m_bounds = np.repeat(
            np.array([[0, 1]]),
            trainables.n_rules * trainables.n_consequents,
            axis=0,
        )

        # rule weight between 0 and 1
        theta_bounds = np.repeat(
            np.array([[0, 1]]), trainables.n_rules, axis=0
        )

        # attribute weights between 0 and 1
        delta_bounds = np.repeat(
            np.array([[0, 1]]),
            trainables.n_rules * trainables.n_attributes,
            axis=0,
        )
        # precedents are unbound
        precedent_bounds = np.repeat(
            np.array([[-np.inf, np.inf]]),
            trainables.n_attributes * trainables.n_precedents,
            axis=0,
        )

        all_bounds = np.concatenate(
            (bre_m_bounds, theta_bounds, delta_bounds, precedent_bounds)
        )

        # construct constraints
        # each row in the BRE matrix must sum to 1
        n_row = trainables.n_rules
        n_col = trainables.n_consequents
        cons_betas = []

        for row in range(n_row):
            con = dict(
                type="eq",
                fun=lambda x, row, n_col: sum(
                    x[row * n_col : (row + 1) * n_col]
                )
                - 1,
                args=[row, n_col],
            )
            cons_betas.append(con)

        # precedents must be hierarchial
        n_total_precedents = trainables.n_attributes * trainables.n_precedents
        precedents_start = (
            trainables.flat_trainables.shape[0] - n_total_precedents
        )
        precedents_end = trainables.flat_trainables.shape[0]
        cons_precedents = []

        for j in range(
            precedents_start, precedents_end, trainables.n_precedents
        ):
            for i in range(j, j + trainables.n_precedents - 1):
                con = dict(
                    type="ineq",
                    fun=lambda x, v: -x[i] + x[i + 1] + v,
                    args=[0],
                )
                cons_precedents.append(con)

        all_cons = cons_betas + cons_precedents

        opt_res = minimize(
            self._objective,
            trainables.flat_trainables,
            args=(trainables, xs, ys),
            method="SLSQP",
            bounds=all_bounds,
            constraints=all_cons,
            options={"ftol": 1e-6, "disp": False},
            callback=lambda _: print("."),
        )

        if opt_res.success:
            print("Training successfull!")
            x = opt_res.x
            trainables.flat_trainables[:] = x

            # update parameters
            bre_m, thetas, deltas, precedents = BRB._unflatten_parameters(
                trainables
            )
            self.bre_m = bre_m
            self.thetas = thetas
            self.deltas = deltas
            self.precedents = precedents
            self.trained = True
            return trainables
        else:
            print("Training NOT success!")
            return _trainables

    def _objective(
        self,
        flat_trainables: np.ndarray,
        trainables: Trainables,
        xs: np.ndarray,
        ys_bar: np.ndarray,
    ):
        trainables.flat_trainables[:] = flat_trainables
        (
            self._train_bre_m,
            self._train_thetas,
            self._train_deltas,
            self._train_precedents,
        ) = BRB._unflatten_parameters(trainables)
        ys = np.apply_along_axis(lambda x: self._predict_train(x), 1, xs)
        res = (1 / len(xs)) * np.sum((ys_bar - ys) ** 2)
        return res

    def _flatten_parameters(self) -> Trainables:
        """Flattens the parameters of the current BRB model so that they can be
        used in training. Created a namedtuple with the flattened parameters
        and relevant information to rebuild the original paramaeters when
        needed.

        """
        n_attributes = self.precedents.shape[0]
        n_precedents = self.precedents.shape[1]
        n_rules = self.thetas.shape[0]
        n_consequents = self.consequents.shape[1]

        flat_bre_m = self.bre_m.flatten()
        flat_rules = self.thetas.flatten()
        flat_attws = np.ones(
            n_rules * n_attributes
        )  # TODO: actually use given ones
        flat_prece = self.precedents.flatten()

        flat_trainables = np.concatenate(
            (flat_bre_m, flat_rules, flat_attws, flat_prece)
        )

        trainables = Trainables(
            flat_trainables, n_attributes, n_precedents, n_rules, n_consequents
        )

        return trainables

    def _unflatten_parameters(
        trainables: Trainables
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Unflatten an instance of Trainables and return the paramters
        defining a BRB-model.

        Arguments:
            trainables (Trainables): A named tuple with flattened parameters
            defining a BRB-model.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray): The belief rule
            expression matrix, the rule weights, the attribute weights in each
            rule and the precedents.

        """
        # running index
        idx = 0
        bre_m = np.reshape(
            trainables.flat_trainables[
                0 : (trainables.n_rules * trainables.n_consequents)
            ],
            (trainables.n_rules, trainables.n_consequents),
        )
        idx = trainables.n_rules * trainables.n_consequents

        thetas = np.reshape(
            trainables.flat_trainables[idx : (idx + trainables.n_rules)],
            (trainables.n_rules, 1),
        )
        idx += trainables.n_rules

        deltas = np.reshape(
            trainables.flat_trainables[
                idx : (idx + trainables.n_rules * trainables.n_attributes)
            ],
            (trainables.n_rules, trainables.n_attributes),
        )
        idx += trainables.n_rules * trainables.n_attributes

        precedents = np.reshape(
            trainables.flat_trainables[
                idx : (idx + trainables.n_attributes * trainables.n_precedents)
            ],
            (trainables.n_attributes, trainables.n_precedents),
        )
        idx += trainables.n_attributes * trainables.n_precedents

        return bre_m, thetas, deltas, precedents

    def _predict_train(self, x: np.ndarray) -> float:
        """Predicts outcomes during training

        Arguments:
            trainables (Trainables): A named tuple with flattened parameters
            defining a BRB-model.
            x (np.ndarray): Input to the BRB-model.

        Returns:
            (float): A prediction

        """
        res = self._predict(
            x,
            self._train_precedents,
            self.consequents,
            self._train_thetas,
            self._train_deltas,
            self._train_bre_m,
        )
        return sum(
            self.utility(self.consequents[0]) * res.consequent_belief_degrees
        )


# Testing
def article2():
    def himmelblau(x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def linspace2d(low, up, n):
        step_s = (up - low) / n
        return (
            np.mgrid[
                low[0] : up[0] + 0.1 : step_s[0],
                low[1] : up[1] + 0.1 : step_s[1],
            ]
            .reshape(2, -1)
            .T
        )

    #    refs = np.array([[0, 1, 2, 2.5, 3]])
    #    consequents = np.array([[-3, -1, 0, 2, 3]])
    refs = np.array([[-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6]])
    consequents = np.array([[0, 200, 500, 1000, 2200]])

    # Construct an initial model
    brb = BRB(refs, consequents, f=himmelblau)

    # generate a random set of inputs and outputs
    low = np.array([-6, -6])
    up = np.array([6, 6])
    n = 4
    xs_train = linspace2d(low, up, n)
    ys_train = np.array(list(map(himmelblau, xs_train)))

    # Real data to compare to
    xs = linspace2d(low, up, 14)
    ys = np.array(list(map(himmelblau, xs)))

    # untrained data
    ys_untrained = [
        np.sum(res.consequents * res.consequent_belief_degrees)
        for res in map(brb.predict, xs)
    ]

    # train the BRB
    brb.train(xs_train, ys_train, brb._flatten_parameters())
    print(brb)

    # trained data
    ys_trained = [
        np.sum(res.consequents * res.consequent_belief_degrees)
        for res in map(brb.predict, xs)
    ]

    plt.plot(np.linspace(0, len(ys), len(ys)), ys, label="function")
    plt.plot(np.linspace(0, len(ys), len(ys)), ys_untrained, label="Untrained")
    plt.plot(np.linspace(0, len(ys), len(ys)), ys_trained, label="Trained")
    plt.legend()
    plt.show()


def article1():
    # define the problem and limits for the input
    def f(x):
        return np.sin(x[0]) * np.cos(x[0] ** 2) * np.exp(np.sin(x[0]))

    low = 0
    up = 3

    # create training data
    n_train = 100
    xs_train = np.random.uniform(low, up, (n_train, 1))
    ys_train = np.apply_along_axis(f, 1, xs_train)

    # create evaluation data
    n_eval = 1000
    xs = np.linspace(low, up, n_eval).reshape(-1, 1)
    ys = np.apply_along_axis(f, 1, xs)

    # create a brb model with given referential values
    precedents = np.array([[0, 0.5, 1, 1.5, 2, 2.5, 3]])
    consequents = np.array([[-2.5, -1, 1, 2, 3]])

    # construct an initial BRB model
    brb = BRB(precedents, consequents, f=f)
    print("Before training")
    print(brb)

    # untrained predictions on evaluation data
    ys_untrained = np.array(
        [
            np.sum(res.consequents * res.consequent_belief_degrees)
            for res in map(brb.predict, xs)
        ]
    )

    # train the model
    brb.train(xs_train, ys_train, brb._flatten_parameters())

    print("After training")
    print(brb)

    ys_trained = np.array(
        [
            np.sum(res.consequents * res.consequent_belief_degrees)
            for res in map(brb.predict, xs)
        ]
    )

    plt.plot(xs, ys, label="f", ls="dotted")
    plt.plot(xs, ys_untrained, label="untrained", ls="--")
    plt.plot(xs, ys_trained, label="trained")
    plt.ylim((-3, 3))
    plt.legend()

    plt.show()


if __name__ == "__main__":
    article1()
