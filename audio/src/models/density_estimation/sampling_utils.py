import numpy as np
from scipy.stats import vonmises

from src.utils.utils import compute_angular_distance, compute_angular_distance_vec
import torch


def uniform_sampling_sphere(n_samples):

    # Vanilla method
    phi = 2 * np.pi * np.random.uniform(size=n_samples)  # belong to [0,2pi]
    theta = (
        np.arccos(2 * np.random.uniform(size=n_samples) - 1) - np.pi / 2
    )  # belong to [-pi/2,pi/2]

    return np.stack((phi, theta), axis=1)  # shape [n_samples, 2]


def single_vde_reject_sampling(
    hyps_pred,
    confs_pred,
    scaling_factor,
    kernel_type,
    max_n_tries,
    return_cell=False,
):
    """Samples one point from a trained Voronoi density estimator by
    sequentially sampling candidates and rejecting them
    """
    # Sample hypothesis (and corresponding cell) with the predicted confidences
    p_i = np.random.choice(
        confs_pred.shape[0],
        p=confs_pred,
    )

    for _ in range(max_n_tries):
        # Sample candidate point
        if (
            kernel_type == "gauss"
            or kernel_type == "gauss_normalized"
            or kernel_type == "von_mises_fisher"
        ):

            # # approximate isotropic VonMises-Fisher sampling by:
            # # 1. sampling an angular direction d=(theta, phi)
            # tangent_direction_angle = 2 * np.pi * np.random.uniform()
            # tangent_direction = np.array(
            #     [
            #         np.cos(tangent_direction_angle),
            #         np.sin(tangent_direction_angle),
            #     ]
            # )
            # angular_direction = np.arccos(tangent_direction)
            # # 2. sampling displacement t from a centered 1D VonMises dist
            # displacement = vonmises(loc=0, kappa=scaling_factor).rvs(1)
            # # 3. building sample x = p + t*d
            # candidate_point = hyps_pred[p_i] + displacement * angular_direction

            candidate_point = np.zeros((2))

            candidate_point[0] = vonmises(loc=0, kappa=1 / scaling_factor**2).rvs(1)

            candidate_point[1] = vonmises(loc=0, kappa=1 / scaling_factor**2).rvs(1)

        elif kernel_type == "uniform":
            candidate_point = 2 * np.pi * np.random.uniform(size=2)
            candidate_point[1] -= np.pi
        else:
            raise ValueError(f"Unsupported kernel_type={kernel_type}")

        # Check whether it is in the Voronoi cell by finding the closest
        # centroid
        min_centroid_dist = np.infty
        closest_centroid = None
        for p_j in range(hyps_pred.shape[0]):
            dist = compute_angular_distance(hyps_pred[p_j], candidate_point)
            if dist < min_centroid_dist:
                min_centroid_dist = dist
                closest_centroid = p_j

        # Stop when a candidate within the cell is found, otherwise retry
        if closest_centroid is not None and closest_centroid == p_i:
            if return_cell:
                return candidate_point, p_i
            else:
                return candidate_point

    # Raise error if not valid candidate is found within allocated budget
    raise RuntimeError(
        "Could not sample a valid point within the maximum number "
        f"of tries set ({max_n_tries}). Try increasing this number."
    )


def compute_angular_distance_vec(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """

    return np.arccos(
        np.sin(x[:, 1]) * np.sin(y[:, 1])
        + np.cos(x[:, 1]) * np.cos(y[:, 1]) * np.cos(y[:, 0] - x[:, 0])
    )


def compute_angular_distance_parallel(x, y):
    # x shape: (n_hyps, 1, 2)
    # y shape: (n_samples, 2)
    sin_x1 = np.sin(x[:, :, 1])
    sin_y1 = np.sin(y[:, 1])
    cos_x1 = np.cos(x[:, :, 1])
    cos_y1 = np.cos(y[:, 1])
    cos_diff = np.cos(y[:, 0] - x[:, :, 0])

    cos_angle = np.clip(sin_x1 * sin_y1 + cos_x1 * cos_y1 * cos_diff, -1.0, 1.0)

    return np.arccos(cos_angle)


def find_closest_centroids(points, hyps_pred):
    # points shape: (n_samples, 2)
    # hyps_pred shape: (n_hyps, 2)
    # Reshape hyps_pred to (n_hyps, 1, 2) for broadcasting
    hyps_pred_reshaped = hyps_pred[:, np.newaxis, :]
    dist = compute_angular_distance_parallel(hyps_pred_reshaped, points)
    min_dist_indices = np.argmin(dist, axis=0)

    return min_dist_indices, None


def multi_vde_reject_sampling(
    n_samples,
    hyps_pred,
    confs_pred,
    scaling_factor,
    kernel_type,
    max_n_tries,
):
    """Samples `n_samples` points from a trained Voronoi density estimator by
    sequentially sampling candidates and rejecting them
    """
    if isinstance(hyps_pred, torch.Tensor):
        if hyps_pred.device.type == "cuda":
            hyps_pred = hyps_pred.detach().cpu().numpy()
            confs_pred = confs_pred.detach().cpu().numpy()

    n_valid_samples = 0
    remaining_samples = n_samples
    valid_candidates = list()

    for _ in range(max_n_tries):
        # Sample hypothesis (and corresp. cell) w/ the predicted confidences
        p_i = np.random.choice(
            confs_pred.shape[0], p=confs_pred, size=remaining_samples
        )

        # Sample candidate point
        if (
            kernel_type == "gauss"
            or kernel_type == "gauss_normalized"
            or kernel_type == "von_mises_fisher"
        ):

            candidate_point = np.zeros((remaining_samples, 2))

            ##########
            kappa = 1 / scaling_factor**2

            # Generate centered Von Mises samples
            centered_samples = vonmises(kappa=kappa).rvs(size=(remaining_samples, 2))

            # Add the means from hyps_pred
            candidate_point = centered_samples + hyps_pred[p_i]
            ##########

        elif kernel_type == "uniform":

            candidate_point = uniform_sampling_sphere(remaining_samples)

            # candidate_point = 2 * np.pi * np.random.uniform(
            # size=2 * remaining_samples
            # ) - np.pi
            # candidate_point = candidate_point.reshape(remaining_samples, 2)
            # candidate_point[:, 1] /= 2
        else:
            raise ValueError(f"Unsupported kernel_type={kernel_type}")

        # Check whether it is in the Voronoi cell by finding the closest
        # centroid
        closest_centroid, _ = find_closest_centroids(
            candidate_point,
            hyps_pred,
        )

        valid_mask = closest_centroid == p_i

        n_valid_samples += np.sum(valid_mask)
        valid_candidates.append(candidate_point[valid_mask])

        # Stop when n_sample valid samples are found, otherwise retry for the
        # remaining number of samples
        if n_valid_samples >= n_samples:
            return np.concatenate(valid_candidates, axis=0)
        else:
            remaining_samples = n_samples - n_valid_samples

    # Raise error if not valid candidate is found within allocated budget
    raise RuntimeError(
        "Could not sample a valid point within the maximum number "
        f"of tries set ({max_n_tries}). Try increasing this number."
    )
