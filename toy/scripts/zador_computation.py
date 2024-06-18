import os
import sys

my_home = os.getenv('MY_HOME')
os.chdir(my_home)
sys.path.append(my_home)

from src.data.dataset import rotating_two_moons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KernelDensity


def compute_integral(f, interval, n=1000):
    volume = np.prod([x_max - x_min for x_min, x_max in interval])
    expectation = 0
    x = np.random.uniform(
        low=interval[:, 0], high=interval[:, 1], size=(n, interval.shape[0])
    )
    for k in range(n):
        expectation += f(x[k, :])
    return (expectation / n) * volume


def zador(
    p,
    n_list,
    d=2,
    interval=np.array([(-10, 10), (-10, 10)]),
    monte_carlo_step=50000,
    J_empirical=None,
):
    J = (
        1 / 12
        if d == 1
        else 5 / (18 * np.sqrt(3)) if d == 2 else d / (2 * np.pi * np.exp(1))
    )
    J = J if J_empirical is None else J_empirical
    p_lim = lambda x: p(x) ** (d / (d + 2))
    p_norm = compute_integral(p_lim, interval, n=monte_carlo_step) ** ((d + 2) / d)
    zador_list = J * p_norm * np.array([n ** (-2 / d) for n in n_list])
    return zador_list


def conditional_zador(
    p,
    n_list,
    d=2,
    input_interval=np.array([-10, 10]),
    target_interval=np.array([(-10, 10), (-10, 10)]),
    input_step=100,
    target_step=1000,
    J_empirical=None,
):
    J = (
        1 / 12
        if d == 1
        else 5 / (18 * np.sqrt(3)) if d == 2 else d / (2 * np.pi * np.exp(1))
    )
    J = J if J_empirical is None else J_empirical
    # x in 1D and y is 2D in the following line
    conditional_p_norm = lambda x: compute_integral(
        lambda y: p(np.concatenate([x, y])) ** (d / (d + 2)),
        target_interval,
        n=target_step,
    ) ** ((d + 2) / d)
    p_norm = compute_integral(conditional_p_norm, input_interval, n=input_step)
    zador_list = J * p_norm * np.array([n ** (-2 / d) for n in n_list])
    return zador_list


if __name__ == "__main__":

    n_model_list = [9, 16, 25, 49, 100]
    dataset_list = ["uncentered_gaussian", "rotating_moons", "changing_damier"]
    input_step, target_step = 10, 10000

    # build density estimator for rotating moon dataset using kde
    data = rotating_two_moons(n_samples=1000)

    def kde_fit(data, t):
        return KernelDensity(kernel="gaussian", bandwidth=0.2).fit(
            data.generate_dataset_distribution(t, 1000)
        )

    # x[0]: scalar input
    # x[1:]: coordinates of the point.
    # densities are function of x = x[0],x[1],x[2] here

    # compare with theoretical
    density_dict = {
        "uncentered_gaussian": lambda x: scipy.stats.multivariate_normal(
            mean=[0.25, 0.25], cov=[[0.2**2, 0], [0, 0.2**2]]
        ).pdf(x[1:]),
        "changing_damier": lambda x: (
            (1 - x[0]) / 2
            if (((x[1] * 4 + 4) // 2 % 2) == ((x[2] * 4 + 4) // 2 % 2))
            else x[0] / 2
        ),
        "rotating_moons": lambda x: np.exp(
            kde_fit(data, x[0]).score_samples(np.array(x[1:]).reshape(1, -1))
        ),
    }

    interval_dict = {
        "uncentered_gaussian": (
            np.array([(0, 1)]),
            np.array([(-0.5, 1.5), (-0.5, 1.5)]),
        ),
        "changing_damier": (np.array([(0, 1)]), np.array([(-1, 1), (-1, 1)])),
        "rotating_moons": (np.array([(0, 1)]), np.array([(-1, 1), (-1, 1)])),
    }

    for dataset_name in dataset_list:
        zador_risk = conditional_zador(
            density_dict[dataset_name],
            n_model_list,
            d=2,
            input_interval=interval_dict[dataset_name][0],
            target_interval=interval_dict[dataset_name][1],
            input_step=input_step,
            target_step=target_step,
        )
        volume = np.prod(
            [x_max - x_min for x_min, x_max in interval_dict[dataset_name][1]]
        )
        grid_risk = [1 / (6 * n) * volume for n in n_model_list]

        # Create individual DataFrames
        df1 = pd.DataFrame(
            {
                "n": n_model_list,
                "risk": zador_risk,
                "model": ["Theoretical WTA"] * len(n_model_list),
            }
        )

        df2 = pd.DataFrame(
            {
                "n": n_model_list,
                "risk": grid_risk,
                "model": ["Theoretical Histogram"] * len(n_model_list),
            }
        )

        # Concatenate the DataFrames
        zador_df = pd.concat([df1, df2], ignore_index=True)

        if dataset_name == "uncentered_gaussian":
            dataset_name = "gaussnotcentered"
        elif dataset_name == "rotating_moons":
            dataset_name = "rotatingmoons"
        elif dataset_name == "changing_damier":
            dataset_name = "changingdamier"
        zador_df.to_csv(
            f"{my_home}/results/saved_zador/zador_{dataset_name}.csv", index=False
        )
        print("Done for", dataset_name)
