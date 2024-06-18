from torch import nn

from .kernels import gaussian_kernel, uniform_kernel, normalized_gaussian_kernel


class BaseDensityEstimator(nn.Module):
    def __init__(self, kernel_type, scaling_factor):
        super().__init__()
        self.kernel_type = kernel_type
        self.scaling_factor = scaling_factor

    def kernel_compute(self, vector_x, mu):
        if self.kernel_type == "gauss":
            return gaussian_kernel(vector_x=vector_x, mu=mu, sigma=self.scaling_factor)
        elif self.kernel_type == "gauss_normalized":
            return normalized_gaussian_kernel(
                vector_x=vector_x, mu=mu, sigma=self.scaling_factor, d=self.output_dim
            )
        elif self.kernel_type == "uniform":
            return uniform_kernel(
                vector_x, square_size=self.square_size, d=self.output_dim
            )
