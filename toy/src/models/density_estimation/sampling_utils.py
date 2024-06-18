import numpy as np

from .directional_radius import (
    compute_double_directional_radius,
    directional_radius,
)


def sample_in_gaussian_seg(
    p,
    z,
    direction,
    seg_min,
    seg_max,
    scaling_factor,
    max_n_tries,
):
    t = np.infty
    b = 1 / (2 * direction)
    mean_along_seg = (p - z).dot(b)
    std_along_seg = np.abs(np.sum(b * scaling_factor))
    for _ in range(max_n_tries):
        # while t > radius_pos or t < -radius_neg:
        t = np.random.normal(
            loc=mean_along_seg,
            scale=std_along_seg,
        )
        if t <= seg_max and t >= seg_min:
            return t

    # If after max_n_tries we still cannot sample in the segment,
    # then the variance is probably too large or the segment is too short.
    # In both cases, we approximate the distribution by a uniform distribution.
    return np.random.uniform(low=seg_min, high=seg_max)


def single_mc_step(
    z,
    chosen_hyp_idx,
    chosen_dir_idx,
    hyps_pred,
    pp_prods,
    dirp_prods,
    zp_prods,
    directions,
    kernel_type,
    scaling_factor,
    max_n_tries,
    square_size,
):
    # renaming
    p_i = chosen_hyp_idx
    dir_i = chosen_dir_idx

    p = hyps_pred[p_i].copy()

    # compute radius
    radius_neg, radius_pos = compute_double_directional_radius(
        origin=z,
        chosen_hyp_idx=p_i,
        chosen_dir_idx=dir_i,
        direction=directions[dir_i],
        hyps_pred=hyps_pred,
        pp_prods=pp_prods,
        dirp_prods=dirp_prods,
        zp_prods=zp_prods,
        square_size=square_size,
    )

    # sample from croppped kernel
    if kernel_type == "gauss" or kernel_type == "gauss_normalized":
        t = sample_in_gaussian_seg(
            p=p,
            z=z,
            direction=directions[dir_i],
            seg_min=-radius_neg,
            seg_max=radius_pos,
            scaling_factor=scaling_factor,
            max_n_tries=max_n_tries,
        )
    elif kernel_type == "uniform":
        t = np.random.uniform(low=-radius_neg, high=radius_pos)
    else:
        raise ValueError(f"Unsupported kernel_type={kernel_type}")

    # markov chain step
    z += t * directions[dir_i]

    # update <z,p> matrix
    zp_prods += t * dirp_prods[dir_i]

    return z, zp_prods


def optimized_single_vde_sampling(
    n_directions,
    hyps_pred,
    confs_pred,
    n_steps_sampling,
    scaling_factor,
    kernel_type,
    max_n_tries,
    return_cell=False,
    square_size=1,
):
    """Samples one point from a trained Voronoi density estimator following
    Alg.2 from [1]_

    References
    ----------
    .. [1] Polianskii et al., "Voronoi Density Estimator for High-Dimensional
       Data", Conference on Uncertainty in Artificial Intelligence, 2022.
    """
    # sample set of directions once
    directions_theta = np.random.uniform(
        0,
        2 * np.pi,
        size=n_directions,
    )
    directions = np.stack([np.cos(directions_theta), np.sin(directions_theta)]).T

    # create <p,p> and <dir,p> matrices once
    pp_prods = hyps_pred @ hyps_pred.T
    dirp_prods = directions @ hyps_pred.T

    # sample cell
    p_i = np.random.choice(
        confs_pred.shape[0],
        p=confs_pred,
    )

    # init sampled point z and product <z,p>
    z = hyps_pred[p_i].copy()
    zp_prods = z @ hyps_pred.T

    for _ in range(n_steps_sampling):
        # sample direction
        dir_i = np.random.choice(n_directions)

        z, zp_prods = single_mc_step(
            z=z,
            chosen_dir_idx=dir_i,
            chosen_hyp_idx=p_i,
            hyps_pred=hyps_pred,
            pp_prods=pp_prods,
            dirp_prods=dirp_prods,
            zp_prods=zp_prods,
            directions=directions,
            kernel_type=kernel_type,
            scaling_factor=scaling_factor,
            max_n_tries=max_n_tries,
            square_size=square_size,
        )

    if return_cell:
        return z, p_i
    else:
        return z


def naive_single_vde_sampling(
    hyps_pred,
    confs_pred,
    scaling_factor,
    kernel_type,
    max_n_tries,
    return_cell=False,
    square_size=1,
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
        if kernel_type == "gauss" or kernel_type == "gauss_normalized":
            candidate_point = np.random.multivariate_normal(
                mean=hyps_pred[p_i],
                cov=(scaling_factor**2) * np.eye(2),
            )
        elif kernel_type == "uniform":
            candidate_point = np.random.uniform(low=-1, high=1, size=2)
        else:
            raise ValueError(f"Unsupported kernel_type={kernel_type}")

        # Check whether it is in the Voronoi cell by computing the directional
        # radius
        effective_direction = candidate_point - hyps_pred[p_i]
        effective_t = np.linalg.norm(
            effective_direction,
            ord=2,
        )
        effective_direction /= effective_t

        radius = directional_radius(
            generator_cell=hyps_pred[p_i],
            direction=effective_direction,
            hypotheses=hyps_pred,
            index_generator_cell=p_i,
            square_size=square_size,
        )

        # Stop when a candidate within the cell is found, otherwise retry
        if effective_t < radius:
            if return_cell:
                return candidate_point, p_i
            else:
                return candidate_point

    # Raise error if not valid candidate is found within allocated budget
    raise RuntimeError(
        "Could not sample a valid point within the maximum number "
        f"of tries set ({max_n_tries}). Try increasing this number."
    )


def naive_single_kde_sampling(
    hyps_pred,
    confs_pred,
    scaling_factor,
    kernel_type,
    max_n_tries,
    return_cell=False,
    square_size=1,
):
    """Samples one point from a trained Voronoi density estimator by
    sequentially sampling candidates and rejecting them
    """
    # Sample hypothesis (and corresponding cell) with the predicted confidences
    p_i = np.random.choice(
        confs_pred.shape[0],
        p=confs_pred,
    )

    # Sample candidate point
    if kernel_type == "gauss" or kernel_type == "gauss_normalized":
        candidate_point = np.random.multivariate_normal(
            mean=hyps_pred[p_i],
            cov=(scaling_factor**2) * np.eye(2),
        )
    elif kernel_type == "uniform":
        candidate_point = np.random.uniform(low=-1, high=1, size=2)
    else:
        raise ValueError(f"Unsupported kernel_type={kernel_type}")

    if return_cell:
        return candidate_point, p_i
    else:
        return candidate_point


def compute_euclidean_distance_vec(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def is_in_square(candidate_point):
    return np.logical_and(
        np.abs(candidate_point[:, 0]) <= 1,
        np.abs(candidate_point[:, 1]) <= 1,
    )


def find_closest_centroids(points, hyps_pred):
    n_samples = points.shape[0]
    min_centroid_dist = np.infty * np.ones(n_samples)
    closest_centroid = -np.ones(n_samples)
    for p_j in range(hyps_pred.shape[0]):
        # shape n_sample
        dist = compute_euclidean_distance_vec(
            np.repeat(hyps_pred[p_j][None, :], n_samples, axis=0),
            points,
        )
        update_mask = dist < min_centroid_dist
        min_centroid_dist[update_mask] = dist[update_mask]
        closest_centroid[update_mask] = p_j
    return closest_centroid, min_centroid_dist


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
        )  # (remaining_samples,)

        candidate_point = np.zeros((remaining_samples, 2))

        # Sample candidate point
        if kernel_type == "gauss" or "gauss_normalized":

            A = np.random.multivariate_normal(
                mean=np.zeros(2),
                cov=(scaling_factor**2) * np.eye(2),
                size=remaining_samples,
            )
            candidate_point = A + hyps_pred[p_i]

        elif kernel_type == "uniform":
            candidate_point = np.random.uniform(
                low=-1, high=1, size=(remaining_samples, 2)
            )
        else:
            raise ValueError(f"Unsupported kernel_type={kernel_type}")

        # Check whether it is in the Voronoi cell by finding the closest
        # centroid
        closest_centroid, _ = find_closest_centroids(
            candidate_point,
            hyps_pred,
        )

        valid_mask = np.logical_and(
            closest_centroid == p_i, is_in_square(candidate_point)
        )
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
