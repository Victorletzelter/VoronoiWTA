import argparse
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import math
import torch
from scipy.spatial import distance
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display
from scipy.spatial import Voronoi, ConvexHull
from torch.utils.data import DataLoader
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

plt.rcParams['text.usetex'] = True

def forward_single_sample_adapted(model, test_loader, device="cpu", gauss_output=False):
    """Function for performing a forward pass with model on a single sample from test_loader."""
    # model.to(device)
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data_t = data[0].to(device)

            # Forward pass
            if gauss_output is False:

                hyps, confs = model(
                    torch.tensor(data_t.float(), device=model.device)
                )  # .reshape(-1,1))
                hyps = hyps.cpu().numpy()
                confs = confs.cpu().numpy()
                model.train()
                return hyps, confs

            else:
                mu, sigma, pi = model(data_t.float())  # .reshape(-1,1))
                mu = mu.cpu().numpy()
                sigma = sigma.cpu().numpy()
                pi = pi.cpu().numpy()
                return mu, sigma, pi


def generate_plot_adapted(
    model,
    dataset_ms,
    dataset_ss,
    path_plot,
    model_type="rMCL",
    list_x_values=[0.1, 0.6, 0.9],
    n_samples_gt_dist=5000,
    num_hypothesis=20,
    log_var_pred=False,
    save_mode=True,
    device="cpu",
    plot_title="Predictions",
    plot_voronoi=True,
    plot_title_bool=False,
):
    """Function to plot the results similarly to those presented in the submission paper.

    Args:
        model_rmcl: rMCL* model.
        model_smcl: sMCL model.
        model_rmcl_all: rMCL model.
        ensemble_models (dict): dictionnary of single hypothesis sMCL for performing ensemble prediction.
        list_x_values (list): List of input t values for which the results are plotted.
        n_samples_gt_dist (int): Number of samples from the ground-truth distribution to plot (green points) at each time step (in list_x_values).
        n_samples_centroids_compt (int): Number of samples from the ground-truth distribution to use for the centroids computation.
        emd_results_saving_path (str): Path of the EMD scores results.
        num_hypothesis (int, optional): Number of hypothesis used. Defaults to 20.
    """

    index = 0
    fig = plt.figure(figsize=(39, 20))
    gs = gridspec.GridSpec(
        1, 6, width_ratios=[21.2, 0.2, 21.2, 0.2, 21.2, 0.2], height_ratios=[1]
    )

    axes = [
        plt.subplot(gs[0, 0]),
        plt.subplot(gs[0, 0]),
        plt.subplot(gs[0, 2]),
        plt.subplot(gs[0, 2]),
        plt.subplot(gs[0, 4]),
        plt.subplot(gs[0, 4]),
    ]

    i = -2

    for t in list_x_values:

        i += 2

        # Create a DataLoader for the test dataset
        test_dataset = dataset_ms(n_samples=1, t=t)
        test_loader = DataLoader(test_dataset, batch_size=1)
        np.random.seed(42)
        samples = dataset_ss(n_samples=n_samples_gt_dist).generate_dataset_distribution(
            t=t, n_samples=n_samples_gt_dist
        )
        np.random.seed(None)
        axes[i].scatter(samples[:, 0], samples[:, 1], c="lightgreen", s=5)
        cmap = plt.get_cmap("Blues")

        list_conf_min = []
        list_conf_max = []

        if model_type == "rMCL":
            # # Evaluate the models
            hyps_rmcl, confs_rmcl = forward_single_sample_adapted(
                model, test_loader, device="cpu"
            )
            confs_rmcl = confs_rmcl / np.sum(confs_rmcl, axis=1, keepdims=True)
            confs_viz = confs_rmcl
            cmap_norm = plt.Normalize(
                vmin=np.min(confs_rmcl[index, :, :]),
                vmax=np.max(confs_rmcl[index, :, :]),
            )
            colors = [
                cmap(cmap_norm(confs_viz[index, k, 0])) for k in range(num_hypothesis)
            ]
            points = hyps_rmcl[index, :, :]

            list_conf_min.append(np.min(confs_rmcl[index, :, :]))
            list_conf_max.append(np.max(confs_rmcl[index, :, :]))

        elif "MDN" in model_type:
            mu_stacked, sigma_stacked, pi_stacked = forward_single_sample_adapted(
                model, test_loader, device=device, gauss_output=True
            )
            num_modes = num_hypothesis
            mu_stacked = np.array(mu_stacked.cpu())
            sigma_stacked = np.array(sigma_stacked.cpu())
            pi_stacked = np.array(pi_stacked.cpu())

            cmap_norm = plt.Normalize(
                vmin=np.min(pi_stacked[index, :, :]),
                vmax=np.max(pi_stacked[index, :, :]),
            )
            colors = [
                cmap(cmap_norm(pi_stacked[index, k, 0])) for k in range(num_modes)
            ]
            axes[i].scatter(
                [mu_stacked[index, k, 0] for k in range(num_modes)],
                [mu_stacked[index, k, 1] for k in range(num_modes)],
                c=colors,
                s=200,
                edgecolors="black",
            )

            for mode in range(num_modes):

                if log_var_pred is False:
                    sigma = sigma_stacked[index, mode, 0]
                else:
                    sigma = np.exp(sigma_stacked[index, mode, 0] / 2)

                x = mu_stacked[index, mode, 0]
                y = mu_stacked[index, mode, 1]
                sigma_x = sigma
                sigma_y = sigma

                # Generate points for the ellipse
                theta = np.linspace(0, 2 * np.pi, 1000)  # 100 points around the ellipse
                ellipse_x = x + sigma_x * np.cos(theta)
                ellipse_y = y + sigma_y * np.sin(theta)

                # Plotting
                # print(colors_red)
                axes[i].scatter(
                    ellipse_x,
                    ellipse_y,
                    s=1,
                    c="red",
                    alpha=(pi_stacked[index, mode, 0] - np.min(pi_stacked[index, :, 0]))
                    / (
                        np.max(pi_stacked[index, :, 0])
                        - np.min(pi_stacked[index, :, 0])
                    ),
                )
                points = mu_stacked[index, :, :]
                sm = plt.cm.ScalarMappable(
                    cmap=cmap,
                    norm=plt.Normalize(
                        vmin=np.min(pi_stacked), vmax=np.max(pi_stacked)
                    ),
                )

        if plot_voronoi is True:
            # Compute the Voronoi tessellation
            vor = Voronoi(points)

            # Plot the Voronoi diagram
            voronoi_plot_2d(vor, ax=axes[i + 1], show_vertices=False, show_points=False)

        if plot_title_bool is True:
            fig.suptitle(plot_title, fontsize=40)
            # set the title of the figure

        axes[i + 1].scatter(
            samples[:, 0], samples[:, 1], c="lightgreen", s=10, label="samples GT dist"
        )
        axes[i + 1].scatter(
            points[:, 0],
            points[:, 1],
            marker="o",
            c=colors,
            s=400,
            label="hypothesis",
            edgecolors="black",
        )

        # Customize plot
        axes[i + 1].set_xlim(-1, 1)
        axes[i + 1].set_ylim(-1, 1)

        plt.rcParams['text.usetex'] = True

        axes[i + 1].set_xlabel(f"$x = {t}$", labelpad=25, fontsize=100)
        axes[i + 1].set_aspect("equal")

    x_label_y_coord = axes[0].xaxis.label.get_position()[1] - 0.05

    for i, ax in enumerate(axes):
        ax.xaxis.set_label_coords(
            axes[i].xaxis.label.get_position()[0], x_label_y_coord
        )  # Adjust y value to align labels

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(list_conf_min), vmax=max(list_conf_max))
    )
    # Set custom ticks

    axes[i].set_xlim(-1, 1)
    axes[i].set_ylim(-1, 1)
    axes[i].set_aspect("equal")

    cax_position = [0.9, 0.24, 0.02, 0.51]  # [left, bottom, width, height]
    cax = fig.add_axes(cax_position)

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([])  # Adjust 'num' for more/less ticks
    cbar.ax.tick_params(labelsize=40)

    for i in range(1, len(axes)):
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    axes[0].set_xticks([-1, 1])
    axes[0].set_yticks([-1, 1])

    axes[0].set_xticklabels(["-1", "1"])
    axes[0].set_yticklabels(["", "1"])  # Remove the '-1' label on the y-axis

    for label in axes[0].get_xticklabels():
        if label.get_text() == "-1":
            # label.set_verticalalignment('center')  # Center vertically
            label.set_horizontalalignment("right")  # Align right to match the y-tick

    plt.subplots_adjust(hspace=0.09, wspace=0.01)

    axes[0].tick_params(axis="x", labelsize=50)
    axes[0].tick_params(axis="y", labelsize=50)
    axes[1].set_zorder(-1)

    # After each subplot creation or at the end of your plotting code
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)  # Adjust the width as needed

    for spine in cax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)  # Adjust the width as needed

    if save_mode is True:
        plt.savefig(os.path.join(path_plot, plot_title + ".png"))
        plt.close()


def binary_search_min(f, a, b, tol=1e-5):
    """
    Perform a binary search to find the minimum of a unimodal function.

    :param f: The function to minimize.
    :param a: The lower bound of the search interval.
    :param b: The upper bound of the search interval.
    :param tol: Tolerance for the convergence criterion.
    :return: The point which is the minimum of the function f in [a, b].
    """
    while abs(b - a) > tol:
        mid1 = a + (b - a) / 3
        mid2 = b - (b - a) / 3

        if f(mid1) < f(mid2):
            b = mid2
        else:
            a = mid1

    return (a + b) / 2


def gss(f, a, b, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]
    Ref: https://en.wikipedia.org/wiki/Golden-section_search

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """

    gr = (math.sqrt(5) + 1) / 2

    while abs(b - a) > tol:
        # print(it)
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        if f(c) < f(d):  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

    return (b + a) / 2


def mirror_points(points):
    mirrored_points = points.tolist()
    for x, y in points:
        # Add mirrored points
        mirrored_points.extend(
            [
                (x, -3 - y),  # mirror across y = -1
                (-3 - x, y),  # mirror across x = -1
                (3 - x, y),  # mirror across x = 1
                (x, 3 - y),  # mirror across y = 1
            ]
        )
    return np.array(mirrored_points)


def plot_voronoi(points):
    all_points = mirror_points(points)
    vor = Voronoi(all_points)

    plt.figure(figsize=(8, 8))

    # Plot Voronoi diagram
    voronoi_plot_2d(
        vor,
        show_vertices=False,
        line_colors="blue",
        line_width=1.5,
        line_alpha=0.6,
        point_size=2,
    )

    # Highlight original points
    plt.plot(points[:, 0], points[:, 1], "ro")

    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.5, 2.5)
    plt.title("Voronoi Diagram with Mirrored Points")
    plt.grid(True)
    plt.show()


def voronoi_volumes_unbounded(original_points, include_unbounded=False):

    print("volumes computation")

    if include_unbounded is True:
        all_points = mirror_points(original_points)
        v = Voronoi(all_points)

    else:
        v = Voronoi(original_points)

    vol = np.zeros(len(original_points))
    for i, point in enumerate(original_points):
        # Find the Voronoi region for the original point
        point_index = np.where((v.points == point).all(axis=1))[0][0]
        region_index = v.point_region[point_index]
        indices = v.regions[region_index]

        if -1 not in indices:
            # Calculate volume only if the region is bounded
            print(i)
            print("done")
            vol[i] = ConvexHull(v.vertices[indices]).volume
        else:
            # Handle unbounded regions appropriately
            # Intersection with [-1,1]^2 square is needed here
            vol[i] = np.inf
            # pass  # You need to implement this part

    print("finished")

    return vol


def sample_hyps_gaussian_mixture(mus, sigmas, pis, N_samples=20):
    # Sample the modes
    modes = torch.multinomial(
        pis.squeeze(), N_samples, replacement=True
    )  # shape [N_samples]

    # Expand and select mus and sigmas based on sampled modes
    selected_mus = mus[modes]  # shape [N_samples, output_dim]
    selected_sigmas = sigmas[modes].reshape(-1, 1)  # shape [N_samples, 1]

    # Sample offsets
    offsets = torch.normal(0, 1, size=(N_samples, mus.shape[1]))

    selected_sigmas_expanded = selected_sigmas.expand_as(offsets)
    offsets = offsets * selected_sigmas_expanded

    # Add offsets to the selected means
    output_hyps = selected_mus + offsets  # shape [N_samples, output_dim]

    return output_hyps


def plot2D_samples_mat(xs, xt, G, weights_source, axis, thr=1e-8, c=[0.5, 0.5, 1]):
    r"""Ref: https://pythonot.github.io/_modules/ot/plot.html#plot2D_samples_mat"""
    mx = G.max()
    scale = 1
    for i in range(xs.shape[0]):
        if weights_source[i] > 0:
            for j in range(xt.shape[0]):
                if G[i, j] / mx > thr:
                    axis.plot(
                        [xs[i, 0], xt[j, 0]],
                        [xs[i, 1], xt[j, 1]],
                        alpha=G[i, j] / mx * scale,
                        c=c,
                    )


def find_voronoi_cell(sample, vor):
    """Function for finding the Voronoi cell of a given sample."""
    min_distance = np.inf
    closest_region = -1

    for i, point in enumerate(vor.points):
        d = distance.euclidean(sample, point)
        if d < min_distance:
            min_distance = d
            closest_region = i

    return closest_region


def gss(f, a, b, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]
    Ref: https://en.wikipedia.org/wiki/Golden-section_search

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """

    gr = (math.sqrt(5) + 1) / 2
    N = 0

    while abs(b - a) > tol and N < 50:
        # print(it)
        N += 1
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        if f(c) < f(d):  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

    x_min = (a + b) / 2
    f_min = f(x_min)

    return x_min, f_min


def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


def voronoi_volume_check(
    predicted_centroids,
    cell_index_to_check,
    vol_to_check,
):
    v = voronoi_volumes(predicted_centroids[:, 1:])
    vol_cell = v[cell_index_to_check]

    assert vol_cell == vol_to_check
