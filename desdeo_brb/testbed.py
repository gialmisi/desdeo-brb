import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as sklearn_minmax_scaler
from brb import BRB, BRBPref, Rule
from utility import load_and_scale_data, plot_utility_monotonicity, plot_3d_ranks_colored, const_mapping
from reasoner import Reasoner
from typing import List, Tuple


def main():
    dir_path = "/home/kilo/workspace/forest-opt/data/"
    fname_po = "payoff.dat"
    fname_pf = "test_run.dat"

    nadir, ideal, paretofront, payoff, scaler = load_and_scale_data(dir_path, fname_po, fname_pf)

    # define parameters for the BRB
    precedents = np.array([
        [0, 0.5, 1],
        [0, 0.5, 1],
        [0, 0.5, 1],
    ])
    consequents = np.array([[0, 0.25, 0.5, 0.75, 1]])

    print("precedents\n", precedents)
    print("consequents\n", consequents)

    # construct BRB

    brb = BRBPref(precedents, consequents, f=simple_mapping, utility=utility)
    print(brb)

    refs = []
    ref_scores = []

    while True:
        res_pf = brb.predict(paretofront)
        score_pf = np.sum(res_pf.consequents * res_pf.consequent_belief_degrees, axis=1)
        best_pf_ind = np.argmax(score_pf)

        score_pf = np.sum(res_pf.consequents * res_pf.consequent_belief_degrees, axis=1)
        best_pf_ind = np.argmax(score_pf)
        print(brb)
        print(f"BRB thinks that the best solutions on the PF is {paretofront[best_pf_ind]} with score {score_pf[best_pf_ind]}")

        plot_3d_ranks_colored(brb, paretofront)
        plot_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])
        plt.show()

        plt.hist(score_pf, bins=100)
        plt.show()
        
        if refs and ref_scores:
            print("Previous points have been:")
            for ref, score in zip(refs, ref_scores):
                res_brb = brb.predict(np.atleast_2d(ref))
                score_brb = np.sum(res_brb.consequents * res_brb.consequent_belief_degrees, axis=1)
                print(f"{ref} which you scored {score} ({score_brb})")

        user_score = float(input("How would you rate this solution on a scale from 0 (worst) to 1 (best)?\n>"))
        if abs(user_score) > 1 or user_score < 0:
            print("Score must be between 0 and 1!")
            continue

        refs.append(paretofront[best_pf_ind])
        ref_scores.append(user_score)

        brb.train(None, None, brb._flatten_parameters(), obj_args=(nadir, ideal, np.atleast_2d(refs), np.atleast_2d(ref_scores)), use_de=True)

        plot_utility_monotonicity(brb, [(0, 1), (0, 1), (0, 1)])

        plt.show()


    plt.show()





    # fig, axs = plt.subplots(1, 3)
    # axs[0].scatter(pf[:, 0], pf[:, 1])
    # axs[0].scatter(extrema[0, 0], extrema[0, 1], label="nadir")
    # axs[0].scatter(extrema[1, 0], extrema[1, 1], label="ideal")
    # axs[0].set_title("Income and Carbon")
    # axs[0].legend()

    # axs[1].scatter(pf[:, 0], pf[:, 2])
    # axs[1].scatter(extrema[0, 0], extrema[0, 2], label="nadir")
    # axs[1].scatter(extrema[1, 0], extrema[1, 2], label="ideal")
    # axs[1].set_title("Income and AHSI")
    # axs[1].legend()

    # axs[2].scatter(pf[:, 1], pf[:, 2])
    # axs[2].scatter(extrema[0, 1], extrema[0, 2], label="nadir")
    # axs[2].scatter(extrema[1, 1], extrema[1, 2], label="ideal")
    # axs[2].set_title("Carbon and AHSI")
    # axs[2].legend()

def utility(y):
    return y

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
