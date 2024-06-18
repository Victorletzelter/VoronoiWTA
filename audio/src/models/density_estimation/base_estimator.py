from torch import nn

from .kernels import (
    gaussian_kernel,
    uniform_kernel,
    normalized_gaussian_kernel,
    von_mises_fisher_kernel,
)


class BaseDensityEstimator(nn.Module):
    def __init__(self, kernel_type, scaling_factor, kde_mode, kde_weighted):
        super().__init__()
        self.kernel_type = kernel_type
        self.scaling_factor = scaling_factor
        self.kde_mode = kde_mode
        self.kde_weighted = kde_weighted

    def kernel_compute(self, vector_x, mu):
        # vector_x (n,2)
        # mu (n,2)

        if vector_x.ndim == 1:
            vector_x = vector_x[None, :]
        if mu.ndim == 1:
            mu = mu[None, :]

        if self.kernel_type == "gauss":
            return gaussian_kernel(vector_x, mu, self.scaling_factor)
        elif self.kernel_type == "gauss_normalized":
            return normalized_gaussian_kernel(vector_x, mu, self.scaling_factor)
        elif self.kernel_type == "uniform":
            return uniform_kernel(vector_x)
        elif self.kernel_type == "von_mises_fisher":
            return von_mises_fisher_kernel(
                vector_x, mu, kappa=1 / self.scaling_factor**2
            )
