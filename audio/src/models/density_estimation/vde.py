import numpy as np

from .base_estimator import BaseDensityEstimator
from .sampling_utils import multi_vde_reject_sampling, find_closest_centroids
from src.utils.utils import compute_angular_distance_vec
import time
from scipy.stats import qmc

import numpy as np


def regular_grid_on_sphere(num_latitudes, num_longitudes):
    """
    Create a regular grid on the sphere.
    Args:
        num_latitudes: Number of latitude divisions (not counting poles)
        num_longitudes: Number of longitude divisions
    Returns:
        grid_points: numpy array of shape [num_points, 2] containing [phi, theta] pairs.
    """

    # Array of latitudes (-pi/2 to pi/2) and longitudes (0 to 2*pi)
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, num_latitudes + 2)[
        1:-1
    ]  # exclude poles
    longitudes = np.linspace(
        0, 2 * np.pi, num_longitudes, endpoint=False
    )  # full circle

    # Create a meshgrid of latitude and longitude values
    phi_grid, theta_grid = np.meshgrid(longitudes, latitudes)
    phi_grid = phi_grid.flatten()
    theta_grid = theta_grid.flatten()

    # Stack into a single array
    grid_points = np.stack((phi_grid, theta_grid), axis=1)  # [num_points, 2]

    return grid_points


def uniform_sampling_sphere(n_samples):

    # Sobol sampling for improved stability.
    # sampler = qmc.Sobol(d=2, scramble=False)  # Use d=2 for the sphere
    # # Generate sample points
    # # Transform Sobol samples to spherical coordinates
    # samples = sampler.random(n_samples)
    # phi = samples[:, 0] * 2 * np.pi  # Azimuth from 0 to 2*pi
    # theta = np.arccos(2 * samples[:, 1] - 1) - np.pi / 2  # Elevation from -pi/2 to pi/2

    # Vanilla method
    phi = 2 * np.pi * np.random.uniform(size=n_samples)  # belong to [0,2pi]
    theta = (
        np.arccos(2 * np.random.uniform(size=n_samples) - 1) - np.pi / 2
    )  # belong to [-pi/2,pi/2]

    return np.stack((phi, theta), axis=1)  # shape [n_samples, 2]


def estimate_uniform_vol(hypotheses, k, n_samples, max_samples=100000):
    while n_samples <= max_samples:
        # Uniform sampling on the sphere
        points_on_sphere = uniform_sampling_sphere(n_samples)

        # Compute number of points falling inside the Voronoi cell
        closest_centroid, _ = find_closest_centroids(points_on_sphere, hypotheses)
        n_points_in_cell = np.sum(closest_centroid == k)

        if n_points_in_cell > 0 or n_samples >= max_samples:
            break

        # Double the number of samples
        n_samples *= 2

    if n_points_in_cell == 0:
        raise ValueError("No points found in the cell")

    # Return the fraction of points falling in the cell, times the total sphere surface area
    return (n_points_in_cell / n_samples) * (4 * np.pi), points_on_sphere[
        closest_centroid == k, :
    ]


def kernel_vol_helper(
    hypotheses,
    k,
    n_samples,
    max_n_tries,
    scaling_factor,
    kernel_type,
):
    """Compute the normalizing constant of the kernel (Voronoi cell volume)
    using Monte-Carlo integration and reject sampling.
    """
    # compute Voronoi cell volume (with uniform measure)
    uniform_vol, samples_uniform_cell_k = estimate_uniform_vol(
        hypotheses=hypotheses,
        k=k,
        n_samples=n_samples,
    )

    if kernel_type == "uniform":
        return uniform_vol, samples_uniform_cell_k

    # else:
    # Create fake confidence with prob=1 on the cell whose volume should be
    # computed --> used to sample from that cell only
    fake_conf = np.zeros(hypotheses.shape[0])
    fake_conf[k] = 1

    # sample uniformly from the cell
    samples = multi_vde_reject_sampling(
        n_samples=n_samples,
        hyps_pred=hypotheses,
        confs_pred=fake_conf,
        scaling_factor=scaling_factor,
        kernel_type="uniform",
        # max_n_tries=max_n_tries,
        max_n_tries=n_samples * hypotheses.shape[0] * 10,
    )

    return uniform_vol, samples


class VoronoiDE(BaseDensityEstimator):
    def __init__(
        self,
        kernel_type=None,
        scaling_factor=None,
        kde_mode=False,
        kde_weighted=False,
    ):
        super().__init__(
            kernel_type=kernel_type,
            scaling_factor=scaling_factor,
            kde_mode=kde_mode,
            kde_weighted=kde_weighted,
        )

    def batch_compute_negative_log_likelihood(
        self, predictions, targets, n_samples=2048
    ):
        # predictions of shape [batch,T,num_hypothesisxoutput_dim], [batch,T,num_hypothesisx1]
        # targets of shape [batch,Max_sourcesx(output_dim+1)]
        # n_samples: Number of samples used to compute the negative log likelihood.
        NLL = 0.0
        batch = predictions[0].shape[0]
        T = predictions[0].shape[1]

        # Put everything to cpu/numpy
        source_activity_target = targets[0].unsqueeze(-1).detach().cpu().numpy()
        target_position = targets[1].detach().cpu().numpy()

        hyps = predictions[0].detach().cpu().numpy()
        confs = predictions[1].detach().cpu().numpy()

        assert np.allclose(
            np.sum(confs, axis=-2), 1.0, atol=1e-3
        ), "Confs should sum close to 1 in the hypotheses axis"

        for t in range(T):
            NLL_t = 0.0
            N_computations = 0.0
            for batch_elt in range(batch):
                if np.sum(source_activity_target[batch_elt, t]) == 0:
                    continue

                if self.kde_mode == True:
                    NLL_elt = self.np_compute_negative_log_likelihood_kde(
                        hyps[batch_elt, t],
                        confs[batch_elt, t],
                        source_activity_target[batch_elt, t],
                        target_position[batch_elt, t],
                        weighted=self.kde_weighted,
                    )

                else:
                    NLL_elt = self.np_compute_negative_log_likelihood(
                        hyps[batch_elt, t],
                        confs[batch_elt, t],
                        source_activity_target[batch_elt, t],
                        target_position[batch_elt, t],
                        n_samples=n_samples,
                    )

                if np.isnan(NLL_elt):
                    continue
                NLL_t += NLL_elt
                N_computations += 1
            if np.sum(source_activity_target[:, t]) == 0:
                continue
            NLL += NLL_t / N_computations

        if isinstance(NLL, float):
            return NLL / T
        else:
            return NLL[0] / T

    def np_compute_negative_log_likelihood(
        self,
        hyps_pred,
        confs_pred,
        source_activity_target,
        target_position,
        # predictions,
        # targets,
        n_samples=100,
        max_n_tries=100,
    ):
        # hyps_pred of shape [self.num_hypothesisxoutput_dim]
        # confs_pred of shape [self.num_hypothesisx1]
        # source_activity_target of shape [Max_sources]
        # target_position of shape [Max_sourcesxoutput_dim]

        signature_pred = np.concatenate((confs_pred, hyps_pred), axis=-1)
        signature_target = np.concatenate(
            (source_activity_target, target_position), axis=-1
        )  # shape [Max_sourcesx(output_dim+1)]

        # Compute the negative log likelihood of the target given the predictions

        nll = 0
        num_hyps = signature_pred.shape[0]
        num_targets = signature_target.shape[0]
        N_samples_nll = 0
        for i in range(num_targets):
            if signature_target[i, 0] == 0:
                continue
            y = signature_target[i, 1:]  # [2]
            j = np.argmin(
                compute_angular_distance_vec(
                    signature_pred[:, 1:],
                    y.reshape(1, 2).repeat(num_hyps, axis=0),
                ),
                axis=0,
            )
            norm_cst = self.kernel_normalizing_constant(
                kernel_mu=signature_pred[j, 1:],
                hypotheses=signature_pred[:, 1:],
                k=j,
                n_samples=n_samples,
                max_n_tries=max_n_tries,
            )

            if np.isnan(norm_cst) or norm_cst == 0:
                # raise an error
                raise ValueError("Normalizing constant is nan or 0")

            prob_pred_y = (
                signature_pred[j, 0]
                * self.kernel_compute(
                    vector_x=y,
                    mu=signature_pred[j, 1:],
                )
                / norm_cst
            )

            if prob_pred_y > 0:
                N_samples_nll += 1
                nll += -np.log(prob_pred_y + 1e-7)

            else:
                continue

        if N_samples_nll == 0:
            return np.nan
        else:
            return nll / N_samples_nll

    def np_compute_negative_log_likelihood_kde(
        self,
        hyps_pred,
        confs_pred,
        source_activity_target,
        target_position,
        weighted=False,
    ):
        # hyps_pred of shape [self.num_hypothesisxoutput_dim]
        # confs_pred of shape [self.num_hypothesisx1]
        # source_activity_target of shape [Max_sources]
        # target_position of shape [Max_sourcesxoutput_dim]
        # weighted: Whether or not each kernel is weighted by its predicted confidence.

        signature_pred = np.concatenate((confs_pred, hyps_pred), axis=-1)
        signature_target = np.concatenate(
            (source_activity_target, target_position), axis=-1
        )  # shape [Max_sourcesx(output_dim+1)]

        # Compute the negative log likelihood of the target given the predictions

        nll = 0
        num_hyps = signature_pred.shape[0]
        num_targets = signature_target.shape[0]
        N_samples_nll = 0

        for i in range(num_targets):
            if signature_target[i, 0] == 0:
                continue
            y = signature_target[i, 1:]  # [2]

            prob_pred_y = 0

            # In KDE, the kernel are diffused over the whole space

            if weighted == True:
                for j in range(num_hyps):
                    prob_pred_y += signature_pred[j, 0] * self.kernel_compute(
                        vector_x=y,
                        mu=signature_pred[j, 1:],
                    )
            else:
                for j in range(num_hyps):
                    prob_pred_y += self.kernel_compute(
                        vector_x=y,
                        mu=signature_pred[j, 1:],
                    )
                prob_pred_y /= num_hyps

            nll += -np.log(prob_pred_y + 1e-7)
            N_samples_nll += 1

        return nll / N_samples_nll

    def kernel_normalizing_constant(
        self,
        kernel_mu,
        hypotheses,
        k,
        n_samples,
        max_n_tries,
    ):
        uniform_vol, samples = kernel_vol_helper(
            hypotheses=hypotheses,
            k=k,
            n_samples=n_samples,
            max_n_tries=max_n_tries,
            kernel_type="uniform",
            scaling_factor=self.scaling_factor,
        )

        if self.kernel_type == "uniform":
            return uniform_vol

        mc_sum = self.kernel_compute(
            vector_x=samples,
            mu=np.repeat(kernel_mu[None, :], samples.shape[0], axis=0),
        ).sum()

        return uniform_vol * mc_sum / samples.shape[0]

    def sample(
        self,
        n_samples,
        hyps_pred,
        confs_pred,
        max_n_tries=1000,
    ):
        """Samples `n_samples` points from the learned conditional
        distribution given the predicted Voronoi centroids and corresponding
        confidences
        """
        return multi_vde_reject_sampling(
            n_samples=n_samples,
            hyps_pred=hyps_pred,
            confs_pred=confs_pred,
            scaling_factor=self.scaling_factor,
            kernel_type=self.kernel_type,
            max_n_tries=max_n_tries,
        )
