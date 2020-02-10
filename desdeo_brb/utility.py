import numpy as np
import matplotlib.pyplot as plt
from brb import BRB
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler as sklearn_minmax_scaler


def load_and_scale_data(dir_path, fname_po, fname_pf):
    """Loads the Pareto front and payoff table. Scales them between 0 and 1, according to the
    ideal and nadir points. Returns the scaled Paretofront, payoff table, nadir, idela, and
    scaler

    """
    payoff = np.loadtxt(dir_path + fname_po)

    # assumin maximization for all objectives
    nadir = np.atleast_2d(np.min(payoff, axis=0))
    ideal = np.atleast_2d(np.copy(np.diag(payoff)))

    # load pareto fornt and clean data
    paretofront = np.loadtxt(dir_path + fname_pf, delimiter=" ")

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

    # scale the data according to the nadir and ideal
    scaler = sklearn_minmax_scaler(copy=True)  # inplace!!!!
    scaler.fit(np.vstack((nadir, ideal)))

    scaled_po = scaler.transform(payoff)
    scaled_nadir = scaler.transform(nadir)
    scaled_ideal = scaler.transform(ideal)
    scaled_pf = scaler.transform(paretofront)

    return scaled_nadir, scaled_ideal, scaled_pf, scaled_po, scaler


def plot_utility_monotonicity(brb: BRB, limits: List[Tuple[float, float]], n=50):
    """Plots the monotonicity of a BRB sytem by varying one of the attribtutes
    and keeping the others constant. The constant value is set to be the middle
    point between the minimum and maximum values for each attribute.

    """
    mid_points = np.array([0, 0.25, 0.5, 0.75, 1])
    fig, axs = plt.subplots(1, len(limits))    
    fig.suptitle("Monotonicity of the utility model")

    for (k, mid_point) in enumerate(mid_points):
        points = np.repeat(np.repeat(mid_point, len(limits)), n).reshape(len(limits), -1).T

        for i, (a_min, a_max) in enumerate(limits):
            points_ = np.copy(points)
            varying = np.linspace(a_min, a_max, n)
            points_[:, i] = varying

            res = brb.predict(points_)
            ys = np.sum(
                res.consequents * res.consequent_belief_degrees, axis=1
            )

            axs[i].plot(varying, ys, label=f"Constant {mid_point}")

            if k == 0:
                axs[i].set_title(f"Varying attribute {i+1}")
                axs[i].set_xlabel(f"Attribute {i+1}")
                axs[i].set_ylabel("Utility")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend()
    plt.draw()


def plot_3d_ranks_colored(brb, paretofront):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib
    import matplotlib.cm as cmx

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    res_pf = brb.predict(paretofront)
    score_pf = np.sum(res_pf.consequents * res_pf.consequent_belief_degrees, axis=1)

    cm = plt.get_cmap("jet")
    c_norm = matplotlib.colors.Normalize(vmin=np.min(score_pf), vmax=np.max(score_pf))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    best_index = np.argmax(score_pf)

    ax.scatter(paretofront[:, 0], paretofront[:, 1], paretofront[:, 2], c=scalar_map.to_rgba(score_pf))
    ax.scatter(paretofront[best_index, 0], paretofront[best_index, 1], paretofront[best_index, 2], marker="D", c="black")
    fig.colorbar(scalar_map)
    ax.set_xlabel("Income")
    ax.set_ylabel("Carbon")
    ax.set_zlabel("CHSI")
    plt.draw()

def main():
   pass

main()






