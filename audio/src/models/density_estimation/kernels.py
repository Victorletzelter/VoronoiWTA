import numpy as np
from src.utils.utils import compute_angular_distance_vec


def spherical_to_cartesian(azimuth, elevation):
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.stack((x, y, z), axis=-1)


def von_mises_fisher_kernel(vector_x, mu, kappa):
    # vector_x (n, 2) where columns are azimuth and elevation
    # mu (n, 2) where columns are azimuth and elevation
    # kappa ()

    # Convert to Cartesian coordinates
    vector_x_cartesian = spherical_to_cartesian(vector_x[:, 0], vector_x[:, 1])
    mu_cartesian = spherical_to_cartesian(mu[:, 0], mu[:, 1])

    # Ensure kappa is a column vector for broadcasting
    # kappa = kappa.reshape(-1, 1)

    # Normalization constant C_3(kappa)
    log_C_3_kappa = np.log(kappa) - np.log(4 * np.pi * np.sinh(kappa))
    # C_3_kappa = kappa / (4 * np.pi * np.sinh(kappa))

    # Dot product of vector_x and mu in Cartesian coordinates
    dot_product = np.sum(mu_cartesian * vector_x_cartesian, axis=1, keepdims=True)

    # Compute the exponential term
    log_exp_term = kappa * dot_product

    # Compute the probability density
    log_output = log_C_3_kappa + log_exp_term

    output = np.exp(log_output)

    return output.squeeze()  # Removing the extra dimension


def gaussian_kernel(vector_x, mu, sigma):
    # mu [n,2]
    # sigma [n,2]
    return np.exp(
        -np.power(compute_angular_distance_vec(vector_x, mu), 2.0)
        / (2 * np.power(sigma, 2.0))
    )


def normalized_gaussian_kernel(vector_x, mu, sigma, d=2):
    # NB: This formula works only in 2D.
    return (1 / (2 * np.pi * (sigma) ** 2)) * np.exp(
        -np.power(compute_angular_distance_vec(vector_x, mu), 2.0)
        / (2 * np.power(sigma, 2.0))
    )


def reparam_gaussian_kernel(self, r, kernel_mu, kernel_sigma):
    return gaussian_kernel(kernel_mu + kernel_sigma * r, kernel_mu, kernel_sigma)


def uniform_kernel(vector_x, x_min=-np.pi, x_max=np.pi, y_min=-np.pi, y_max=np.pi):

    return np.ones(vector_x.shape[0])
