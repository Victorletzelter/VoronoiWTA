from argparse import Namespace
from .modules import (
    AbstractLocalizationModule,
    FeatureExtraction,
    VonMisesMixLocalizationOutput,
    LogKappaVonMisesMixLocalizationOutput,
)
import torch
import torch.nn as nn
from typing import Tuple
from src.utils.losses import VonMisesNLLLoss
import numpy as np
from src.utils.utils import compute_angular_distance_torch, logsinh_np
from src.models.density_estimation.vde import (
    uniform_sampling_sphere,
    regular_grid_on_sphere,
)


class MIXTURE_VONMISES_NLLSELDNet(AbstractLocalizationModule):
    def __init__(self, dataset_path: str, cv_fold_idx: int, hparams: Namespace) -> None:
        hparams["num_hypothesis"] = hparams["num_modes"]
        super().__init__(dataset_path, cv_fold_idx, hparams)

        self.loss_function = self.get_loss_function()

        num_steps_per_chunk = int(2 * hparams["chunk_length"] / hparams["frame_length"])
        self.feature_extraction = FeatureExtraction(
            num_steps_per_chunk,
            hparams["num_fft_bins"],
            dropout_rate=hparams["dropout_rate"],
        )

        feature_dim = int(
            hparams["num_fft_bins"] / 4
        )  # See the FeatureExtraction module for the justification of this
        # value for the feature_dim.

        self.gru = nn.GRU(
            feature_dim,
            hparams["hidden_dim"],
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        if hparams["log_kappa_pred"] == False:
            self.localization_output = VonMisesMixLocalizationOutput(
                input_dim=2 * hparams["hidden_dim"], num_modes=hparams["num_modes"]
            )

        if hparams["log_kappa_pred"] == True:

            # raise not implemented error here
            self.localization_output = LogKappaVonMisesMixLocalizationOutput(
                input_dim=2 * hparams["hidden_dim"], num_modes=hparams["num_modes"]
            )

        # In the localization module, the input_dim is to 2 * hparams.hidden_dim if bidirectional=True in the GRU.

    def get_loss_function(self) -> nn.Module:
        return VonMisesNLLLoss(
            self.hparams["max_num_sources"],
            num_modes=self.hparams["num_modes"],
            log_kappa_pred=self.hparams["log_kappa_pred"],
        )

    def forward(
        self, audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        extracted_features = self.feature_extraction(
            audio_features
        )  # extracted_features of shape
        # [batch,T,B/4] where batch is the batch size, T is the number of time steps per chunk, and B is the number of FFT bins.

        output, _ = self.gru(
            extracted_features
        )  # output of shape [batch,T,hparams['hidden_dim']]

        mu_stacked, kappa_stacked, pi_stacked = self.localization_output(output)
        # if log_kappa_pred = True, mu_stacked, logvar_stacked, pi_stacked instead.

        return mu_stacked, kappa_stacked, pi_stacked

    def log_vonmises_kernel_np(self, y, mu_pred, kappa_pred, log_kappa_pred=False):
        # Assuming compute_spherical_distance is another method of your class
        # y batch,3,3,2
        # mu_pred batch,3,3,2
        # kappa_pred
        # batch, Max_sources, num_modes, _ = y.shape

        mu_pred = mu_pred.reshape(-1, 2)  # Shape [batch*num_modes*max_sources,2]
        kappa_pred = kappa_pred.reshape(-1, 1)  # Shape [batch*num_modes*max_sources,1]
        kappa_pred = kappa_pred.squeeze(-1)
        y = y.reshape(-1, 2)  # Shape [batch*num_modes*max_sources,2]

        # Convert to cartesian coordinates
        mu_pred = self.spherical_to_cartesian_np(
            mu_pred[:, 0], mu_pred[:, 1]
        )  # Shape [batch*num_modes*max_sources,3]
        y = self.spherical_to_cartesian_np(
            y[:, 0], y[:, 1]
        )  # Shape [batch*num_modes*max_sources,3]

        if log_kappa_pred == True:
            return (
                kappa_pred
                - np.log(4 * np.pi)
                - logsinh_np(np.exp(kappa_pred))
                + np.exp(kappa_pred) * (y * mu_pred).sum(axis=1)
            )
        else:
            return (
                np.log(kappa_pred)
                - np.log(4 * np.pi)
                - logsinh_np(kappa_pred)
                + kappa_pred * (y * mu_pred).sum(axis=1)
            )

    def spherical_to_cartesian_np(self, azimuth, elevation):
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)
        return np.stack((x, y, z), axis=-1)

    def spherical_to_cartesian(self, azimuth, elevation):
        x = torch.cos(elevation) * torch.cos(azimuth)
        y = torch.cos(elevation) * torch.sin(azimuth)
        z = torch.sin(elevation)
        return torch.stack((x, y, z), dim=-1)

    def batch_compute_negative_log_likelihood(self, predictions, targets):
        return self.loss_function(predictions, targets)[0].detach().cpu().numpy().mean()

    def batch_compute_negative_log_likelihood_np(self, predictions, targets, **kwargs):
        # mu_pred of shape [batch,T,num_modesx2]
        # kappa_pred of shape [batch,T,num_modesx1]
        # pi_pred of shape [batch,T,num_modes]
        # source_activity_target of shape [batch,T,max_num_sources]
        # target_position of shape [batch,T,max_num_sourcesx2]
        # n_samples: Number of samples used to compute the negative log likelihood.

        mu_pred, kappa_pred, pi_pred = predictions
        if len(targets) == 4:
            source_activity_target, target_position, _, _ = targets
        else:
            source_activity_target, target_position = targets

        # Put everything in cpu/numpy
        mu_pred = mu_pred.detach().cpu().numpy()
        kappa_pred = kappa_pred.detach().cpu().numpy()
        pi_pred = pi_pred.detach().cpu().numpy()
        source_activity_target = source_activity_target.detach().cpu().numpy()
        target_position = target_position.detach().cpu().numpy()

        batch, T, num_modes, _ = mu_pred.shape
        max_sources = source_activity_target.shape[2]

        NLL = 0

        for t in range(T):
            mu_pred_t = mu_pred[:, t, :, :]  # [batch,num_modes,2]
            kappa_pred_t = kappa_pred[:, t, :, :]  # [batch,num_modes,2]
            pi_pred_t = pi_pred[:, t, :]  # [batch,num_modes]
            source_activity_target_t = source_activity_target[
                :, t, :
            ]  # [batch,max_sources]
            target_position_t = target_position[:, t, :, :]  # [batch,max_sources,2]

            # Log probabilities of pi
            log_pi_pred_stacked = np.log(pi_pred_t)  # [batch, num_modes, 1]
            log_pi_pred_stacked = log_pi_pred_stacked.transpose(
                0, 2, 1
            )  # [batch,1,num_modes]

            # Expand targets and predictions to match each other's shapes for batch-wise computation
            expanded_direction_of_arrival_target = np.expand_dims(target_position_t, 2)
            expanded_direction_of_arrival_target = np.tile(
                expanded_direction_of_arrival_target, (1, 1, num_modes, 1)
            )  # [batch, max_sources, num_modes, 2]
            expanded_mu_pred_stacked = np.expand_dims(mu_pred_t, 1)
            expanded_mu_pred_stacked = np.tile(
                expanded_mu_pred_stacked, (1, max_sources, 1, 1)
            )  # [batch, max_sources, num_modes, 2]

            expanded_kappa_pred_stacked = np.expand_dims(kappa_pred_t, 1)
            expanded_kappa_pred_stacked = np.tile(
                expanded_kappa_pred_stacked, (1, max_sources, 1, 1)
            )  # [batch, max_sources, num_modes, 1]

            # Call to log_normal_kernel function
            log_vonmises_values = self.log_vonmises_kernel_np(
                y=expanded_direction_of_arrival_target,
                mu_pred=expanded_mu_pred_stacked,
                kappa_pred=expanded_kappa_pred_stacked,
                log_kappa_pred=self.hparams["log_kappa_pred"],
            )

            log_vonmises_values = log_vonmises_values.reshape(
                batch, max_sources, num_modes
            )  # [batch, max_sources, num_modes]

            # Include the log probabilities of pi and compute the log-sum-exp
            total_log_prob = (
                log_vonmises_values + log_pi_pred_stacked
            )  # [batch, max_sources, num_modes]
            log_sum_exp = logsumexp(total_log_prob, axis=2)  # [batch, max_sources]
            log_sum_exp = np.expand_dims(log_sum_exp, -1)  # [batch, max_sources, 1]
            # b = np.log(np.sum(np.exp(total_log_prob), axis=2))
            # assert (log_sum_exp == np.log(np.sum(np.exp(total_log_prob), axis=2))).all()
            # )  # [batch, max_sources, 1]

            # Mask out inactive sources
            active_source_mask = np.expand_dims(
                source_activity_target_t, -1
            )  # [batch, max_sources, 1]
            masked_log_sum_exp = (
                log_sum_exp * active_source_mask
            )  # [batch, max_sources, 1]
            active_source_mask_sum = np.sum(active_source_mask, axis=1)  # [batch, 1]
            active_source_mask_sum = np.tile(
                active_source_mask_sum, (1, max_sources)
            )  # [batch, max_sources]

            mask = active_source_mask_sum > 0
            masked_log_sum_exp = np.squeeze(
                masked_log_sum_exp, axis=-1
            )  # [batch, max_sources]

            masked_log_sum_exp[mask] = (
                masked_log_sum_exp[mask] / active_source_mask_sum[mask]
            )  # [batch, max_sources]

            # Compute mean negative log likelihood
            NLL += -np.sum(masked_log_sum_exp) / batch

        NLL = NLL / T

        return NLL

    def sample(self, n_samples, predictions, batch_and_T_shape=False):

        if batch_and_T_shape is True:
            # mus of shape [batch,T,num_modes, output_dim]
            # kappas of shape [batch,T,num_modes, 1]
            # pis of shape [batch,T,num_modes, 1]

            mus, kappas, pis = predictions

            batch = mus.shape[0]
            T = mus.shape[1]
            num_modes = mus.shape[2]
            output_dim = mus.shape[3]

            assert mus.shape == (
                batch,
                T,
                num_modes,
                output_dim,
            ), "mus shape is not correct"
            assert kappas.shape == (
                batch,
                T,
                num_modes,
                1,
            ), "kappas shape is not correct"
            assert pis.shape == (batch, T, num_modes, 1), "pis shape is not correct"

            if self.hparams["log_kappa_pred"] == True:
                kappas = torch.exp(kappas / 2)

            modes = torch.zeros(
                batch, T, n_samples, dtype=torch.int64, device=mus.device
            )

            # Sample the modes
            for batch_value in range(batch):
                modes[batch_value, :] = torch.multinomial(
                    pis[batch_value, :, :, 0], n_samples, replacement=True
                )  # shape [T,n_samples]

            # modes.unsqueeze(-1).expand(-1,-1,-1,output_dim) # shape [batch,T,n_samples,output_dim]

            mus_selected = torch.gather(
                input=mus,
                dim=2,
                index=modes.unsqueeze(-1).expand(-1, -1, -1, output_dim),
            )
            kappas_selected = torch.gather(
                input=kappas, dim=2, index=modes.unsqueeze(-1).expand(-1, -1, -1, 1)
            )

            return torch.randn_like(mus_selected) * kappas_selected + mus_selected

        elif batch_and_T_shape is False:
            # mus of shape [num_modes, output_dim]
            # kappas of shape [num_modes, 1]
            # pis of shape [num_modes, 1]
            mus, kappas, pis = predictions

            if self.hparams["log_kappa_pred"] == True:
                kappas = torch.exp(kappas / 2)

            # Sample the modes
            modes = torch.multinomial(
                pis, n_samples, replacement=True
            )  # shape [N_samples]

            return torch.randn_like(mus[modes]) * kappas[modes] + mus[modes]

    def kernel(self, vector_x, mu, kappa):
        a = compute_angular_distance_torch(vector_x, mu)
        return (1 / (2 * torch.pi * (kappa) ** 2)) * torch.exp(
            -torch.pow(compute_angular_distance_torch(vector_x, mu), 2.0)
            / (2 * torch.pow(kappa, 2.0))
        )

    def compute_normalizing_constant(self, predictions, n_mc_samples=10):

        mus, kappas, pis = predictions
        batch = mus.shape[0]
        T = mus.shape[1]
        num_modes = mus.shape[2]

        if self.hparams["log_kappa_pred"] == True:
            kappas = torch.exp(kappas / 2)

        # Compute by MC sampling an estimation of the integral of the predicted density over the unit sphere
        # Sample uniform points on the unit sphere
        theta = (
            torch.rand(batch, T, n_mc_samples, device=mus.device) * np.pi - np.pi / 2
        )
        phi = torch.rand(batch, T, n_mc_samples, device=mus.device) * 2 * np.pi - np.pi

        # Concatenate the angles
        angles = torch.stack([theta, phi], dim=-1)  # shape [batch, T, n_mc_samples, 2]

        # Let convert this expression in pytorch
        # (1/(2*np.pi*(kappa)**2))*np.exp(
        # -np.power(compute_angular_distance_vec(vector_x , mu), 2.0)
        # / (2 * np.power(kappa, 2.0)))

        angles = angles.unsqueeze(-3)  # shape [batch, T, 1, n_mc_samples, 2]
        angles = angles.expand(-1, -1, num_modes, -1, -1)
        mus = mus.unsqueeze(-2)  # shape [batch, T, num_modes, 1, 2]
        mus = mus.expand(-1, -1, -1, n_mc_samples, -1)
        kappas = kappas.unsqueeze(-2)
        kappas = kappas.expand(-1, -1, -1, n_mc_samples, -1)

        angles = angles.reshape(-1, 2)
        mus = mus.reshape(-1, 2)
        kappas = kappas.reshape(-1, 1)
        kappas = kappas.squeeze(-1)

        # Compute the kernel values
        kernel_values = self.kernel(
            angles, mus, kappas
        )  # shape [batch, T, n_mc_samples, num_modes]

        kernel_values = kernel_values.reshape(batch, T, n_mc_samples, num_modes)

        # average the kernel values over the MC samples
        kernel_values = torch.mean(kernel_values, dim=-2)  # shape [batch, T, num_modes]
        # compute weighted sum over the modes
        kernel_values = torch.sum(
            kernel_values * pis.squeeze(-1), dim=-1
        )  # shape [batch, T]

        # Multiply by the surface of the unit sphere
        return kernel_values * 4 * np.pi

    def prepare_predictions_emd(self, predictions):

        mu_stacked, kappa_stacked, pi_stacked = predictions

        hyps_DOAs_pred_stacked = self.sample(
            n_samples=self.hparams["N_samples_predicted_dist"],
            predictions=(mu_stacked, kappa_stacked, pi_stacked),
            batch_and_T_shape=True,
        )

        conf_stacked = torch.ones_like(hyps_DOAs_pred_stacked[:, :, :, 0:1])
        conf_stacked = conf_stacked / conf_stacked.sum(dim=-2, keepdim=True)

        return (hyps_DOAs_pred_stacked, conf_stacked)

    def prepare_predictions_oracle(self, predictions):

        mu_stacked, _, _ = predictions

        hyps_DOAs_pred_stacked = mu_stacked

        conf_stacked = torch.ones_like(hyps_DOAs_pred_stacked[:, :, :, 0:1])
        conf_stacked = conf_stacked / conf_stacked.sum(dim=-2, keepdim=True)

        return (hyps_DOAs_pred_stacked, conf_stacked)

    def prepare_predictions_nll(self, predictions):
        # mu_pred, kappa_pred, pi_pred = predictions
        return predictions

    def check_nll_normalization_sphere(self, predictions):

        # predictions: mu_pred, sigma_pred, pi_pred of shape [batch, num_hypothesis, output_dim], [batch, num_hypothesis, 1], [batch, num_hypothesis, 1]

        mus, kappas, pis = predictions
        # Consider only one element of the batch

        N_samples_NLL_computed = 1
        integrals = 0
        T = mus.shape[1]

        for batch_elt_target in range(N_samples_NLL_computed):

            mu_pred = mus[batch_elt_target]
            kappa_pred = kappas[batch_elt_target]
            pi_pred = pis[batch_elt_target]

            N_samples = 10

            target_position = regular_grid_on_sphere(N_samples, N_samples)
            N_samples = target_position.shape[0]
            target_position = np.tile(
                target_position[:, None, :], (1, T, 1)
            )  # shape (N_samples, T, 2)
            target_position = target_position[:, :, None, :]
            target_position = np.tile(target_position, (1, 1, 3, 1))
            source_activity_target = np.ones((N_samples, T, 3), dtype=bool)

            LL = 0

            for batch_elt in range(N_samples):

                predictions = (
                    mu_pred.unsqueeze(0),
                    kappa_pred.unsqueeze(0),
                    pi_pred.unsqueeze(0),
                )
                targets = (
                    torch.tensor(
                        source_activity_target[batch_elt], device=mu_pred.device
                    )
                    .unsqueeze(0)
                    .float(),
                    torch.tensor(target_position[batch_elt], device=mu_pred.device)
                    .unsqueeze(0)
                    .float(),
                )

                NLL_elt = self.batch_compute_negative_log_likelihood(
                    predictions, targets
                )
                LL += np.exp(-NLL_elt)

            integrals += (LL * 4 * np.pi) / N_samples

        return integrals / N_samples_NLL_computed
