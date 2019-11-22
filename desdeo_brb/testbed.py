import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as sklearn_minmax_scaler
from brb import BRB, BRBPref
from reasoner import Reasoner
from typing import List, Tuple


def main():
    data_dir = "/home/kilo/workspace/forest-opt/data/"
    # load payoff table and setup nadir and ideal
    payoff_f = "payoff.dat"
    payoff = np.loadtxt(data_dir + payoff_f)

    # assumin maximization for all objectives
    nadir = np.atleast_2d(np.min(payoff, axis=0))
    ideal = np.atleast_2d(np.copy(np.diag(payoff)))

    print("payoff\n", payoff)
    print("nadir\n", nadir)
    print("ideal\n", ideal)

    # load pareto fornt and clean data
    paretofront_f = "test_run.dat"
    paretofront = np.loadtxt(data_dir + paretofront_f, delimiter=" ")
    # drop zero rows
    paretofront = paretofront[
        np.all(paretofront > np.zeros(paretofront.shape[1]), axis=1)
    ]

    # drop non-feasible points
    paretofront = paretofront[
        np.all(paretofront >= nadir, axis=1)
        & np.all(paretofront <= ideal, axis=1)
    ]

    print("paretofront\n", paretofront)

    # scale the data according to the nadir and ideal
    scaler = sklearn_minmax_scaler(copy=False)  # inplace!!!!
    scaler.fit(np.vstack((nadir, ideal)))

    scaler.transform(payoff)
    scaler.transform(nadir)
    scaler.transform(ideal)

    print("normalized payoff\n", payoff)
    print("normalized nadir\n", nadir)
    print("normalized ideal\n", ideal)

    scaler.transform(paretofront)

    print("normalized paretofront\n", paretofront)

    # define parameters for the BRB
    precedents = np.array([[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]])
    consequents = np.array([[0, 0.25, 0.5, 0.75, 1]])

    print("precedents\n", precedents)
    print("consequents\n", consequents)

    ref = np.atleast_2d([0.77, 0.11, 0.11])
    # project to PF
    ref_on_pf = paretofront[
        np.argmin(np.sum(np.sqrt((ref - paretofront) ** 2), axis=1))
    ]

    print("ref\n", ref)
    print("ref on PF\n", ref_on_pf)

    worst = paretofront[
        np.argmax(np.sum(np.sqrt((ref - paretofront) ** 2), axis=1))
    ]

    print("worst on PF\n", worst)

    # mapping to construct initial rules
    def mapping(x, best=ref_on_pf, worst=worst):
        """Compares the distance of x to best and worst.
        Return a number between 0 and 1. If x is best, return 1, if x is worst,
        return 0.

        """
        if x.ndim == 3:
            x = np.squeeze(x)
        dist_best = np.sum(np.sqrt((np.atleast_2d(x) - best) ** 2), axis=1)
        dist_worst = np.sum(np.sqrt((np.atleast_2d(x) - worst) ** 2), axis=1)
        dist_tot = dist_best + dist_worst

        # normalize between 0 and 1
        return (
            ((-dist_best / dist_tot + dist_worst / dist_tot) - -1) / 2
        ).reshape(1, -1, 1)

    def mapping2(x):
        if x.ndim == 3:
            x = np.squeeze(x)

        return np.atleast_3d(np.sum(x, axis=1) / 3)

    # construct BRB
    point = ref_on_pf
    brb = BRBPref(precedents, consequents, f=mapping2)

    check_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])

    res = brb.predict(np.atleast_2d(point))
    print("\n\n\n###### RESULT before ######")
    print(f"The point {(np.atleast_2d(point))} is found to be:")
    reasoner = Reasoner(
        [
            ["low", "fair", "high"],
            ["low", "fair", "high"],
            ["low", "fair", "high"],
        ],
        [
            [
                "dissatisfying",
                "somewhat dissatisfying",
                "neutral",
                "somewhat satisfying",
                "satisfying",
            ]
        ],
        ["Income", "Carbon", "AHSI"],
        "Quality of solution",
    )

    print(reasoner.explain(res))

    brb.train(None, None, brb._flatten_parameters(), obj_args=(ref_on_pf, payoff))
    check_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])
    print(brb)

    res = brb.predict(np.atleast_2d(point))
    print("\n\n\n###### RESULT after ######")
    print(f"The point {(np.atleast_2d(point))} is found to be:")
    reasoner = Reasoner(
        [
            ["low", "fair", "high"],
            ["low", "fair", "high"],
            ["low", "fair", "high"],
        ],
        [
            [
                "dissatisfying",
                "somewhat dissatisfying",
                "neutral",
                "somewhat satisfying",
                "satisfying",
            ]
        ],
        ["Income", "Carbon", "AHSI"],
        "Quality of solution",
    )

    print(reasoner.explain(res))

    plt.show()
    

    # point = paretofront[4]

    # res = brb.predict(np.atleast_2d(point))

    # print("\n\n\n###### RESULT ######")
    # print(f"The point {(np.atleast_2d(point))} is found to be:")
    # reasoner = Reasoner(
    #     [
    #         ["low", "fair", "high"],
    #         ["low", "fair", "high"],
    #         ["low", "fair", "high"],
    #     ],
    #     [
    #         [
    #             "dissatisfying",
    #             "somewhat dissatisfying",
    #             "neutral",
    #             "somewhat satisfying",
    #             "satisfying",
    #         ]
    #     ],
    #     ["Income", "Carbon", "AHSI"],
    #     "Quality of solution",
    # )

    # print(reasoner.explain(res))



    # reasoner = Reasoner(
    #     [["bad", "fair", "good"], ["low", "medium", "high"]],
    #     [["poor", "rich", "excellent"]],
    #     ["condition", "price"],
    #     "deal quality"
    # )

    # # train brb
    # train_set = np.vstack((
    #     paretofront,
    #     np.vstack((
    #         ref_on_pf, worst
    #     ))
    # ))
    # brb.train(paretofront, mapping(paretofront), brb._flatten_parameters(), fix_endpoints=True)
    # print(brb)

    # res = brb.predict(paretofront)
    # assessed = np.sum(
    #     res.consequents * res.consequent_belief_degrees, axis=1
    # )

    # for point, score in zip(paretofront, assessed):
    #     print(f"{point} got score {score:.4f}")


def check_utility_monotonicity(brb: BRB, limits: List[Tuple[float, float]], n=50):
    """Check the monotonicity of a BRB sytem by varying one of the attribtutes
    and keeping the others constant. The constant value is set to be the middle
    point between the minimum and maximum values for each attribute.

    """
    mid_points = np.array([(a + b) / 2 for (a, b) in limits])

    points = np.repeat(mid_points, n).reshape(len(limits), -1).T
    fig, axs = plt.subplots(len(limits))    
    fig.suptitle("Monotonicity of the utility model")
    xs = np.array(np.linspace(1, n, n))

    for i, (a_min, a_max) in enumerate(limits):
        points_ = np.copy(points)
        varying = np.linspace(a_min, a_max, n)
        points_[:, i] = varying

        res = brb.predict(points_)

        ys = np.sum(
            res.consequents * res.consequent_belief_degrees, axis=1
        )

        axs[i].plot(xs, ys)
        axs[i].set_title(f"Varying attribute {i+1}")
        axs[i].set_xlabel("arb")
        axs[i].set_ylabel("Utility")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()


if __name__ == "__main__":
    main()
    # check_monotonicity(None, [(0, 5), (-2, 2), (-4, -2), (6, 12)])
    pass
