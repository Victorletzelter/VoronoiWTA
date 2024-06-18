# %%
import numpy as np
from sampling_utils import single_vde_reject_sampling, multi_vde_reject_sampling


# %%
fake_hyps = np.array(
    [
        [0, 0],
        [np.pi / 2, 0],
        [np.pi, 0],
        [-np.pi / 2, 0],
        [0, np.pi / 2],
        [0, -np.pi / 2],
    ]
)

np.random.seed(44)
fake_scores = np.random.uniform(size=6)
fake_scores /= np.sum(fake_scores)

# %%
single_vde_reject_sampling(
    hyps_pred=fake_hyps,
    confs_pred=fake_scores,
    scaling_factor=0.1,
    kernel_type="gauss",
    max_n_tries=10,
    return_cell=True,
)

# %%
from scipy.stats import vonmises
from src.utils.utils import compute_angular_distance_vec


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
    n_valid_samples = 0
    remaining_samples = n_samples
    valid_candidates = list()

    for _ in range(max_n_tries):
        # Sample hypothesis (and corresp. cell) w/ the predicted confidences
        p_i = np.random.choice(
            confs_pred.shape[0], p=confs_pred, size=remaining_samples
        )

        # Sample candidate point
        if kernel_type == "gauss" or kernel_type == "gauss_normalized":
            # approximate isotropic VonMises-Fisher sampling by:
            # 1. sampling an angular direction d=(theta, phi)
            tangent_direction_angle = (
                2 * np.pi * np.random.uniform(size=remaining_samples)
            )
            tangent_direction = np.stack(
                [
                    np.cos(tangent_direction_angle),
                    np.sin(tangent_direction_angle),
                ],
                axis=1,
            )
            angular_direction = np.arccos(tangent_direction)
            # 2. sampling displacement t from a centered 1D VonMises dist
            displacement = vonmises(loc=0, kappa=scaling_factor).rvs(remaining_samples)[
                :, None
            ]
            # 3. building sample x = p + t*d
            candidate_point = hyps_pred[p_i] + displacement * angular_direction
        elif kernel_type == "uniform":
            candidate_point = (
                2 * np.pi * np.random.uniform(size=2 * remaining_samples) - np.pi
            )
            candidate_point = candidate_point.reshape(remaining_samples, 2)
            candidate_point[:, 1] /= 2
        else:
            raise ValueError(f"Unsupported kernel_type={kernel_type}")

        # Check whether it is in the Voronoi cell by finding the closest
        # centroid
        min_centroid_dist = np.infty * np.ones(remaining_samples)
        closest_centroid = -np.ones(remaining_samples)
        for p_j in range(hyps_pred.shape[0]):
            # shape n_sample
            dist = compute_angular_distance_vec(
                np.repeat(hyps_pred[p_j][None, :], remaining_samples, axis=0),
                candidate_point,
            )
            update_mask = dist < min_centroid_dist
            min_centroid_dist[update_mask] = dist[update_mask]
            closest_centroid[update_mask] = p_j

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


# %%
samples = multi_vde_reject_sampling(
    n_samples=100,
    hyps_pred=fake_hyps,
    confs_pred=fake_scores,
    scaling_factor=0.0001,
    kernel_type="gauss",
    max_n_tries=100,
)

# %%
import matplotlib.pyplot as plt

# %%

fig, ax = plt.subplots()
ax.scatter(
    x=fake_hyps[:, 0],
    y=fake_hyps[:, 1],
    c=fake_scores,
    cmap="Reds",
)
ax.scatter(
    x=fake_hyps[:, 0],
    y=fake_hyps[:, 1],
    c="k",
    marker="+",
)
ax.scatter(
    x=samples[:, 0],
    y=samples[:, 1],
    c="blue",
    marker=".",
)
# %%
