import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as sklearn_minmax_scaler
from brb import BRB, BRBPref, Rule
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

    # drop non-unique entries
    paretofront = np.unique(paretofront, axis=0)


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
    precedents = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
    ])
    consequents = np.array([[0, 0.25, 0.5, 0.75, 1]])

    print("precedents\n", precedents)
    print("consequents\n", consequents)

    ref = np.atleast_2d([0.9, 0.8, 0.9])
    print("ref\n", ref)

    # construct BRB
    #brb = BRBPref(precedents, consequents, f=lambda x: dist_to_ref(x, ref))
    rules = [
        Rule([0, 0, 0], 0),
        Rule([1, 1, 1], 1)
    ]

    print(rules)
    brb = BRBPref(precedents, consequents,
                  f=zero_mapping,
                  rules=rules)
    print(brb)

    check_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])

    # Phase 1 - Check what is the closest point on the PF that matches the DM's reference point
    # no training yet
    res_ref = brb.predict(ref)
    score_ref = np.sum(res_ref.consequents * res_ref.consequent_belief_degrees)
    print(f"Score of ref {score_ref}")

    res_pf = brb.predict(paretofront)
    score_pf = np.sum(res_pf.consequents * res_pf.consequent_belief_degrees, axis=1)
    best_pf_ind = np.argmax(score_pf)
    print(f"Best point on PF is {paretofront[best_pf_ind]} with score {score_pf[best_pf_ind]}")
    print(np.sort(score_pf))

    plt.show()
    exit()

    # train the model to satisfy monotonicity
    xs_train = np.atleast_2d(ref)
    xs_train = np.atleast_2d([[0.9, 0.8, 0.9], [0, 0, 0]])
    ys_train = np.array([1, 0.5])
    # brb.train(None, None, brb._flatten_parameters(), obj_args=(paretofront, ref, score_ref, payoff))
    brb.train(xs_train, ys_train, brb._flatten_parameters(), fix_endpoints=True)

    check_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])

    print(brb)

    res_ref = brb.predict(ref)
    score_ref = np.sum(res_ref.consequents * res_ref.consequent_belief_degrees)
    print(f"Score of ref {score_ref}")

    res_pf = brb.predict(paretofront)
    score_pf = np.sum(res_pf.consequents * res_pf.consequent_belief_degrees, axis=1)
    best_pf_ind = np.argmax(score_pf)
    print(f"ref point {ref}")
    print(f"Best point on PF is {paretofront[best_pf_ind]} with score {score_pf[best_pf_ind]}")
    print(np.sort(score_pf))

    plt.show()
    return

    # res = brb.predict(np.atleast_2d(ref))
    # print("\n\n\n###### RESULT before ######")
    # print(f"The point {(np.atleast_2d(ref))} is found to be:")
    # reasoner = Reasoner(
    #     [
    #         ["low", "high"],
    #         ["low", "high"],
    #         ["low", "high"],
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

    # brb.train(None, None, brb._flatten_parameters(), obj_args=(paretofront, ref, payoff))
    # check_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])
    # print(brb)

    # res = brb.predict(np.atleast_2d(ref))
    # print("\n\n\n###### RESULT after ######")
    # print(f"The point {(np.atleast_2d(ref))} is found to be:")

    # print(reasoner.explain(res))

    # predicted_front = brb.predict(paretofront)
    # scores = (
    #     np.sum(
    #         predicted_front.consequents * predicted_front.consequent_belief_degrees, axis=1,
    #     )
    # )

    # for i in range(len(scores)):
    #     print(f"Point {paretofront[np.argsort(-scores)[i]]} has score {scores[np.argsort(-scores)[i]]}")

    plt.show()


def check_utility_monotonicity(brb: BRB, limits: List[Tuple[float, float]], n=100):
    """Check the monotonicity of a BRB sytem by varying one of the attribtutes
    and keeping the others constant. The constant value is set to be the middle
    point between the minimum and maximum values for each attribute.

    """
    mid_points = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    fig, axs = plt.subplots(len(limits))    
    fig.suptitle("Monotonicity of the utility model")

    for (k, mid_point) in enumerate(mid_points):
        points = np.repeat(np.repeat(mid_point, len(limits)), n).reshape(len(limits), -1).T

        for i, (a_min, a_max) in enumerate(limits):
            points_ = np.copy(points)
            varying = np.linspace(a_min, a_max, n)
            points_[:, i] = varying

            res = brb.predict(points_)
            print(res.consequent_belief_degrees)
            ys = np.sum(
                res.consequents * res.consequent_belief_degrees, axis=1
            )

            print(ys)

            axs[i].plot(varying, ys, label=f"Constant {mid_point}")

            if k == 0:
                axs[i].set_title(f"Varying attribute {i+1}")
                axs[i].set_xlabel(f"Attribute {i+1}")
                axs[i].set_ylabel("Utility")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend()
    plt.draw()


def simple_mapping(x):
    if x.ndim == 3:
        x = np.squeeze(x)

    return np.atleast_3d(np.sum(x, axis=1) / 3)


def zero_mapping(x):
    if x.ndim == 3:
        x = np.squeeze(x)

    return np.atleast_3d(np.zeros(len(x)))


def random_mapping(x):
    if x.ndim == 3:
        x = np.squeeze(x)

    return np.atleast_3d(np.random.uniform(0, 1, len(x)))


def dist_to_ref(x, ref):
    if x.ndim == 3:
        x = np.squeeze(x)

    distances = np.linalg.norm(x - ref, axis=1)
    max_distance = np.max(distances) + 1e-6
    return np.atleast_3d(((max_distance - distances)/max_distance))


if __name__ == "__main__":
    main()
    # check_monotonicity(None, [(0, 5), (-2, 2), (-4, -2), (6, 12)])
    pass
