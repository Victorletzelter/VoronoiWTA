import numpy as np

from .kernels import gaussian_kernel, uniform_kernel, normalized_gaussian_kernel
from .base_estimator import BaseDensityEstimator
from .directional_radius import directional_radius
from .sampling_utils import (
    single_mc_step,
    naive_single_vde_sampling,
    naive_single_kde_sampling,
    multi_vde_reject_sampling,
)
from scipy.special import gammainc


def sample_direction_d(dimension):
    # Generate a random vector where each component is from a normal distribution
    random_vector = np.random.normal(0, 1, dimension)
    # Normalize the vector to have unit length
    unit_vector = random_vector / np.linalg.norm(random_vector)
    return unit_vector


def kernel_normalizing_constant_mc(
    kernel_mu,
    hypotheses,
    k,
    N_samples_directions,
    N_samples_dir,
    kernel_type,
    kernel_sigma,
    square_size,
):
    """Compute the normalizing constant of the kernel (Voronoi cell volume)
    using two nested Monte-Carlo integrations.
    """
    normalizing_constant = 0
    output_dim = hypotheses.shape[-1]
    for i in range(N_samples_directions):
        normalizing_constant_dir = 0

        direction = sample_direction_d(dimension=output_dim)

        # direction_theta = np.random.uniform(0, 2 * np.pi)
        # direction = np.array(
        # [np.cos(direction_theta), np.sin(direction_theta)]
        # )
        directional_radius_l = directional_radius(
            generator_cell=hypotheses[k, :],
            direction=direction,
            hypotheses=hypotheses,
            index_generator_cell=k,
            square_size=square_size,
        )

        # Manage the case where the direction_radius_l is 0
        if np.abs(directional_radius_l) < 1e-7:
            while np.abs(directional_radius_l) < 1e-7:

                direction = sample_direction_d(dimension=output_dim)

                directional_radius_l = directional_radius(
                    generator_cell=hypotheses[k, :],
                    direction=direction,
                    hypotheses=hypotheses,
                    index_generator_cell=k,
                    square_size=square_size,
                )

        for j in range(N_samples_dir):
            t_j = np.random.uniform(low=0.0, high=1.0) * directional_radius_l
            if kernel_type == "gauss":
                normalizing_constant_dir += (
                    gaussian_kernel(
                        vector_x=kernel_mu + kernel_sigma * t_j * direction,
                        mu=kernel_mu,
                        sigma=kernel_sigma,
                    )
                    * t_j
                )
            elif kernel_type == "gauss_normalized":
                normalizing_constant_dir += (
                    normalized_gaussian_kernel(
                        vector_x=kernel_mu + kernel_sigma * t_j * direction,
                        mu=kernel_mu,
                        sigma=kernel_sigma,
                    )
                    * t_j
                )
            elif kernel_type == "uniform":
                normalizing_constant_dir += (
                    uniform_kernel(
                        vector_x=kernel_mu + t_j * direction,
                    )
                    * t_j
                )
            else:
                raise ValueError(f"Unsupported kernel_type={kernel_type}")
        normalizing_constant_dir *= directional_radius_l / N_samples_dir
        normalizing_constant += normalizing_constant_dir
    normalizing_constant *= 2 * np.pi / N_samples_directions

    return normalizing_constant


def kernel_normalizing_constant_closed(
    hypotheses,
    k,
    N_samples_directions,
    kernel_type="gauss",
    kernel_sigma=None,
    square_size=1,
):
    """Compute the normalizing constant of the kernel (Voronoi cell volume)
    using closed-form formulas for inner integration and Monte-Carlo for the
    outer integration.
    """

    output_dim = hypotheses.shape[-1]

    normalizing_constant = 0
    for _ in range(N_samples_directions):
        normalizing_constant_dir = 0

        direction = sample_direction_d(dimension=output_dim)

        directional_radius_l = directional_radius(
            generator_cell=hypotheses[k, :],
            direction=direction,
            hypotheses=hypotheses,
            index_generator_cell=k,
            square_size=square_size,
        )

        if np.abs(directional_radius_l) < 1e-7:
            while np.abs(directional_radius_l) < 1e-7:
                direction = sample_direction_d(dimension=output_dim)

                directional_radius_l = directional_radius(
                    generator_cell=hypotheses[k, :],
                    direction=direction,
                    hypotheses=hypotheses,
                    index_generator_cell=k,
                    square_size=square_size,
                )

        output_dim = hypotheses.shape[-1]

        if kernel_type == "gauss":
            # Closed-form from Polianskii et al. 2022 after equation (10)
            # (includes the S^1 sphere surface)
            if output_dim == 2:
                normalizing_constant_dir = (2 * np.pi * kernel_sigma**2) * (
                    1 - np.exp(-(directional_radius_l**2) / (2 * kernel_sigma**2))
                )
            else:
                normalizing_constant_dir = (
                    np.power(2 * np.pi, output_dim / 2) * np.power(kernel_sigma, 2)
                ) * gammainc(
                    output_dim / 2, (directional_radius_l**2) / (2 * kernel_sigma**2)
                )

        elif kernel_type == "gauss_normalized":
            if output_dim == 2:
                # NB: In dim = 2, the normalizing cst of the gaussian kernel is 1/(2pi*sigma^2)
                normalizing_constant_dir = 1 - np.exp(
                    -(directional_radius_l**2) / (2 * kernel_sigma**2)
                )
            else:
                normalizing_constant_dir = gammainc(
                    output_dim / 2, (directional_radius_l**2) / (2 * kernel_sigma**2)
                )

        elif kernel_type == "uniform":
            # S^1 sphere surface (2pi) x integral of t dt (equation 10)
            # (includes the S^1 sphere surface)
            normalizing_constant_dir = np.pi * directional_radius_l**2
        else:
            raise ValueError(f"Unsupported kernel_type={kernel_type}")

        normalizing_constant += normalizing_constant_dir
    normalizing_constant /= N_samples_directions
    return normalizing_constant


class VoronoiDE(BaseDensityEstimator):
    def __init__(
        self,
        square_size=None,
        kernel_type=None,
        scaling_factor=None,
        closed_form_vol=None,
        hit_and_run_sampling=None,
        n_directions=1000,
    ):
        super().__init__(
            kernel_type=kernel_type,
            scaling_factor=scaling_factor,
        )
        self.closed_form_vol = closed_form_vol
        self.hit_and_run_sampling = hit_and_run_sampling
        self.square_size = square_size

        # sample set of directions once
        self.n_directions = n_directions

        # Sample each coordinate from a standard normal distribution
        random_directions = np.random.normal(size=(n_directions, self.output_dim))

        # Normalize each vector to have unit length
        norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
        unit_directions = random_directions / norms

        self.directions = unit_directions

    def np_compute_negative_log_likelihood(
        self,
        hyps_pred,
        confs_pred,
        source_activity_target,
        target_position,
        N_samples_directions=10,
        N_samples_dir=50,
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
        output_dim = signature_target.shape[1] - 1
        for i in range(num_targets):
            if signature_target[i, 0] == 0:
                continue
            y = signature_target[i, 1:]  # [2]
            j = np.argmin(
                np.linalg.norm(
                    signature_pred[:, 1:]
                    - y.reshape(1, output_dim).repeat(num_hyps, axis=0),
                    axis=1,
                ),
                axis=0,
            )

            norm_cst = self.kernel_normalizing_constant(
                kernel_mu=signature_pred[j, 1:],
                hypotheses=signature_pred[:, 1:],
                k=j,
                N_samples_directions=N_samples_directions,
                N_samples_dir=N_samples_dir,
            )

            if np.isnan(norm_cst) or norm_cst == 0:
                continue
            prob_pred_y = (
                signature_pred[j, 0]
                * self.kernel_compute(
                    vector_x=y,
                    mu=signature_pred[j, 1:],
                )
                / norm_cst
            )

            nll += -np.log(prob_pred_y + 1e-7)
            N_samples_nll += 1

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
        N_samples_directions,
        N_samples_dir,
    ):
        if self.closed_form_vol:
            return kernel_normalizing_constant_closed(
                hypotheses=hypotheses,
                k=k,
                N_samples_directions=N_samples_directions,
                kernel_type=self.kernel_type,
                kernel_sigma=self.scaling_factor,
                square_size=self.square_size,
            )
        else:
            return kernel_normalizing_constant_mc(
                kernel_mu=kernel_mu,
                hypotheses=hypotheses,
                k=k,
                N_samples_directions=N_samples_directions,
                N_samples_dir=N_samples_dir,
                kernel_type=self.kernel_type,
                kernel_sigma=self.scaling_factor,
                square_size=self.square_size,
            )

    def sample(
        self,
        n_samples,
        hyps_pred,
        confs_pred,
        n_steps_sampling=20,
        max_n_tries=10000,
    ):
        """Samples `n_samples` points from the learned conditional
        distribution given the predicted Voronoi centroids and corresponding
        confidences
        """

        # hyps_pred (num_hypothesis, 2)

        assert np.all(np.abs(hyps_pred) <= self.square_size), "hyps_pred out of square"

        # If we are in KDE mode, the sampling is straightforward
        if self.hparams.kde_mode_nll == True and self.hparams.kde_weighted_nll == True:
            z_arr = np.empty((n_samples, hyps_pred.shape[1]))
            for i in range(n_samples):
                z_arr[i, :] = naive_single_kde_sampling(
                    ###
                    hyps_pred=hyps_pred,
                    confs_pred=confs_pred,
                    scaling_factor=self.scaling_factor,
                    kernel_type=self.kernel_type,
                    max_n_tries=max_n_tries,
                    square_size=self.square_size,
                )

        elif (
            self.hparams.kde_mode_nll == True and self.hparams.kde_weighted_nll == False
        ):
            z_arr = np.empty((n_samples, hyps_pred.shape[1]))
            confs_pred = np.ones_like(confs_pred) / hyps_pred.shape[0]
            for i in range(n_samples):
                z_arr[i, :] = naive_single_kde_sampling(
                    ###
                    hyps_pred=hyps_pred,
                    confs_pred=confs_pred,
                    scaling_factor=self.scaling_factor,
                    kernel_type=self.kernel_type,
                    max_n_tries=max_n_tries,
                    square_size=self.square_size,
                )

        elif (
            self.hparams.kde_mode_nll == False
            and self.hparams.kde_weighted_nll == False
        ):
            multi_reject_sampling = True
            self.hit_and_run_sampling = False

            # Hit-and-run corresponds to the optimized solution proposed in Alg.2
            # of Polianskii et al. 2022
            if self.hit_and_run_sampling:
                # create <p,p> and <dir,p> matrices once
                pp_prods = hyps_pred @ hyps_pred.T
                dirp_prods = self.directions @ hyps_pred.T

                # sample cells for all samples
                p_i_arr = np.random.choice(
                    confs_pred.shape[0],
                    p=confs_pred,
                    size=n_samples,
                )

                # init sampled point z and product <z,p>
                z_arr = hyps_pred[p_i_arr].copy()
                zp_prods = z_arr @ hyps_pred.T

                for _ in range(n_steps_sampling):
                    # sample direction for all samples
                    dir_i_arr = np.random.choice(self.n_directions, size=n_samples)

                    # Update all samples with an MC step
                    for k, (dir_i, p_i, z) in enumerate(zip(dir_i_arr, p_i_arr, z_arr)):
                        z_arr[k], zp_prods[k] = single_mc_step(
                            z=z,
                            chosen_dir_idx=dir_i,
                            chosen_hyp_idx=p_i,
                            hyps_pred=hyps_pred,
                            pp_prods=pp_prods,
                            dirp_prods=dirp_prods,
                            zp_prods=zp_prods[k],
                            directions=self.directions,
                            kernel_type=self.kernel_type,
                            scaling_factor=self.scaling_factor,
                            max_n_tries=max_n_tries,
                            square_size=self.square_size,
                        )

            elif multi_reject_sampling is True:

                return multi_vde_reject_sampling(
                    n_samples=n_samples,
                    hyps_pred=hyps_pred,
                    confs_pred=confs_pred,
                    scaling_factor=self.scaling_factor,
                    kernel_type=self.kernel_type,
                    max_n_tries=max_n_tries,
                )

            else:
                z_arr = np.empty((n_samples, hyps_pred.shape[1]))
                for i in range(n_samples):
                    z_arr[i, :] = naive_single_vde_sampling(
                        ###
                        hyps_pred=hyps_pred,
                        confs_pred=confs_pred,
                        scaling_factor=self.scaling_factor,
                        kernel_type=self.kernel_type,
                        max_n_tries=max_n_tries,
                        square_size=self.square_size,
                    )

        return z_arr

    def batch_compute_negative_log_likelihood(self, predictions, targets):
        # predictions of shape [batch,num_hypothesisxoutput_dim], [batch,num_hypothesisx1]
        # targets of shape [batch,Max_sourcesx(output_dim+1)]
        NLL = 0.0
        batch = predictions[0].shape[0]

        target_position, source_activity_target = (
            targets[0].detach().cpu().numpy(),
            targets[1].detach().cpu().numpy(),
        )

        hyps = predictions[0].detach().cpu().numpy()
        confs = predictions[1].detach().cpu().numpy()

        assert np.allclose(
            np.sum(confs, axis=-2), 1.0, atol=1e-3
        ), "Confs should sum close to 1 in the hypotheses axis"

        N_computations = 0.0
        for batch_elt in range(batch):
            if np.sum(source_activity_target[batch_elt]) == 0:
                continue

            if self.hparams["kde_mode_nll"] == True:
                NLL_elt = self.np_compute_negative_log_likelihood_kde(
                    hyps[batch_elt],
                    confs[batch_elt],
                    source_activity_target[batch_elt],
                    target_position[batch_elt],
                    weighted=self.hparams["kde_weighted_nll"],
                )

            else:
                NLL_elt = self.np_compute_negative_log_likelihood(
                    hyps[batch_elt],
                    confs[batch_elt],
                    source_activity_target[batch_elt],
                    target_position[batch_elt],
                    N_samples_directions=self.hparams.N_samples_directions,
                    N_samples_dir=self.hparams.N_samples_dir,
                )

            if np.isnan(NLL_elt):
                raise ValueError("NLL is NaN")
                # continue
            NLL += NLL_elt
            N_computations += 1

        return NLL / N_computations

    def check_nll_normalization_1d(self, predictions):

        hyps = (
            predictions[0].detach().cpu().numpy()[0, :, :]
        )  # [batch,num_hypothesisxoutput_dim]
        confs = (
            predictions[1].detach().cpu().numpy()[0, :, :]
        )  # [batch,num_hypothesisx1]

        # Sample uniformly the targets in [-3,3]
        # targets of shape [batch,Max_sources,(output_dim+1)]
        N_samples = 1000
        square_size = 5
        target_position = np.random.uniform(
            -square_size, square_size, (N_samples, 1, 1)
        )
        source_activity_target = np.ones((N_samples, 1, 1), dtype=bool)

        LL = 0

        for batch_elt in range(N_samples):
            if self.hparams["kde_mode_nll"] == True:
                NLL_elt = self.np_compute_negative_log_likelihood_kde(
                    hyps,
                    confs,
                    source_activity_target[batch_elt],
                    target_position[batch_elt],
                    weighted=self.hparams["kde_weighted_nll"],
                )

            else:
                NLL_elt = self.np_compute_negative_log_likelihood(
                    hyps,
                    confs,
                    source_activity_target[batch_elt],
                    target_position[batch_elt],
                    N_samples_directions=self.hparams.N_samples_directions,
                    N_samples_dir=self.hparams.N_samples_dir,
                )

            LL += np.exp(-NLL_elt)

        return (LL * 2 * square_size) / N_samples
