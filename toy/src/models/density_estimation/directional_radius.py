import numpy as np


def directional_radius(
    generator_cell, direction, hypotheses, index_generator_cell, square_size
):
    """generator_cell of shape [2]
    direction of shape [2]
    hypotheses of shape [self.num_hypothesisx2]
    """
    # This implementation does not correspond to the optimized
    # algorithm (Alg.1) described in Polianskii et al. 2022

    # The following values allow to constrain the radius to the unit square
    list_directional_radius = list(
        (square_size * np.sign(direction) - hypotheses[index_generator_cell, :])
        / direction
    )

    for i in range(hypotheses.shape[0]):
        if i != index_generator_cell:
            scalar_prodct = 2 * np.dot(direction, hypotheses[i, :] - generator_cell)
            if scalar_prodct > 0:
                quantity = (
                    np.linalg.norm(hypotheses[i, :] - generator_cell) ** 2
                    / scalar_prodct
                )
                list_directional_radius.append(quantity)

    return np.min(list_directional_radius)


def compute_double_directional_radius(
    origin,
    chosen_hyp_idx,
    chosen_dir_idx,
    direction,
    hyps_pred,
    pp_prods,
    dirp_prods,
    zp_prods,
    square_size=1,
):
    # convenient renamings
    p_i = chosen_hyp_idx
    dir_i = chosen_dir_idx

    # radius_pos = radius_neg = np.infty
    radius_pos = np.min((square_size * np.sign(direction) - origin) / direction)

    radius_neg = np.min((square_size * np.sign(direction) + origin) / direction)

    for q_i in range(hyps_pred.shape[0]):
        if q_i == p_i:
            continue

        radius_ = (
            pp_prods[q_i, q_i]
            - pp_prods[p_i, p_i]
            - 2 * zp_prods[q_i]
            + 2 * zp_prods[p_i]
        )
        radius_ /= 2 * (dirp_prods[dir_i, q_i] - dirp_prods[dir_i, p_i])

        if radius_ > 0:
            radius_pos = min(radius_pos, radius_)
        else:
            radius_neg = min(radius_neg, -radius_)

    return radius_neg, radius_pos
