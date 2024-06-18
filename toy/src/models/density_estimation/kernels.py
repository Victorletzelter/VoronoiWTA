import numpy as np


def gaussian_kernel(vector_x, mu, sigma):
    return np.exp(
        -np.power(np.linalg.norm(vector_x - mu), 2.0) / (2 * np.power(sigma, 2.0))
    )


def normalized_gaussian_kernel(vector_x, mu, sigma, d):
    return (1 / (np.power(2 * np.pi, d / 2) * np.power(sigma, d))) * np.exp(
        -np.power(np.linalg.norm(vector_x - mu), 2) / (2 * sigma**2)
    )


def reparam_gaussian_kernel(r, kernel_mu, kernel_sigma):
    return gaussian_kernel(kernel_mu + kernel_sigma * r, kernel_mu, kernel_sigma)


def uniform_kernel(vector_x, square_size, d):
    """
    Checks if each component of the vector 'vector_x' lies within the corresponding bounds given in 'bounds'.

    :param vector_x: A list or array representing a point in N-dimensional space.
    :param bounds: A list of tuples, where each tuple represents the minimum and maximum bounds for the corresponding dimension.
    :return: 1 if the point lies within the bounds for all dimensions, 0 otherwise.
    """
    bounds = [(-square_size, square_size) for _ in range(d)]

    for component, (min_bound, max_bound) in zip(vector_x, bounds):
        if not (min_bound < component < max_bound):
            return 0
    return 1
