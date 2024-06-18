import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .methodsLighting import methodsLighting

from src.utils.losses import mhloss, mhconfloss, nll_loss

from .density_estimation import VoronoiDE
from scipy.special import logsumexp


class GaussMix(methodsLighting):
    def __init__(
        self,
        hparams,
        num_hypothesis,
        hidden_layers,
        log_var_pred,
        restrict_to_square,
        input_dim,
        output_dim,
        square_size=None,
    ):
        """Constructor for Mixture Density Network with Gaussian kernel.

        Args:
            num_hypothesis (int): Number of output hypotheses.
        """
        methodsLighting.__init__(self, hparams)

        self._hparams = hparams
        self.log_var_pred = log_var_pred
        self.num_hypothesis = num_hypothesis
        # self.num_hidden_units = num_hidden_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.restrict_to_square = restrict_to_square

        self.layers = nn.ModuleList()
        self.activation_functions = nn.ModuleList()
        self.final_mu_layers = nn.ModuleDict()
        self.final_sigma_layers = nn.ModuleDict()
        self.final_pi_layers = nn.ModuleDict()

        # Construct the architecture
        self.construct_layers(hidden_layers)

        # Construct the final layers
        self.construct_final_layers(hidden_layers[-1])

    def construct_layers(self, hidden_layers):
        """
        Constructs the sequence of layers based on the input and hidden layers configuration.
        """
        self.layers.append(nn.Linear(self.input_dim, hidden_layers[0]))
        self.activation_functions.append(nn.ReLU())

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.activation_functions.append(nn.ReLU())

    def construct_final_layers(self, last_hidden_size):
        """
        Constructs the hypothesis and confidence layers based on the number of hypotheses.
        """
        for k in range(self.num_hypothesis):
            self.final_mu_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=last_hidden_size, out_features=self.output_dim
            )
            self.final_sigma_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=last_hidden_size, out_features=1
            )
            self.final_pi_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=last_hidden_size, out_features=1
            )

    def forward(self, x):
        """For pass of the multi-hypothesis network with confidence (rMCL).

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis. Shape [batchxself.num_hypothesisx1]
        """
        # Pass input through each layer
        for layer, activation in zip(self.layers, self.activation_functions):
            x = activation(layer(x))

        outputs_pi = []
        outputs_mu = []
        outputs_sigma = []

        for k in range(self.num_hypothesis):
            outputs_pi.append(
                (self.final_pi_layers["mode_" + "{}".format(k)](x))
            )  # Size [batchx1]
            if self.restrict_to_square:
                outputs_mu.append(
                    F.tanh(self.final_mu_layers["mode_" + "{}".format(k)](x))
                )
            else:
                outputs_mu.append(
                    (self.final_mu_layers["mode_" + "{}".format(k)](x))
                )  # Size [batchxoutput_dim]
            outputs_sigma.append(
                self.final_sigma_layers["mode_" + "{}".format(k)](x)
            )  # Size [batchx1]

        pi_stacked = torch.stack(
            outputs_pi, dim=-2
        )  # Shape [batchxself.num_hypothesisx1]
        assert pi_stacked.shape == (x.shape[0], self.num_hypothesis, 1)
        mu_stacked = torch.stack(outputs_mu, dim=-2)
        assert mu_stacked.shape == (x.shape[0], self.num_hypothesis, self.output_dim)
        sigma_stacked = torch.stack(outputs_sigma, dim=-2)
        assert sigma_stacked.shape == (x.shape[0], self.num_hypothesis, 1)

        if self.log_var_pred is False:
            # sigma_stacked = torch.nn.ReLU()(sigma_stacked)
            sigma_stacked = torch.nn.ELU()(sigma_stacked) + 1

        pi_stacked = torch.nn.Softmax(dim=-2)(
            pi_stacked
        )  # Shape [batchxself.num_hypothesisx1]

        return mu_stacked, sigma_stacked, pi_stacked

    def compute_euclidean_square_distance(self, y, mu_pred):
        # Assuming compute_spherical_distance is another method of your class
        # y 128,3,3,output_dim
        # mu_pred 128,3,3,output_dim
        batch, Max_sources, num_modes, _ = y.shape

        mu_pred = mu_pred.reshape(-1, self.output_dim)
        y = y.reshape(-1, self.output_dim)

        diff = np.square(mu_pred - y)  # Shape [batch*max_sources*num_modes,output_dim]

        return np.sum(diff, axis=1).reshape(batch, Max_sources, num_modes)

    def log_normal_kernel(self, y, mu_pred, sigma_pred):
        batch, Max_sources, num_modes, _ = y.shape
        # print(mu_pred.shape)
        dist_square = self.compute_euclidean_square_distance(
            y, mu_pred
        )  # [batch, max_sources, num_modes]
        dist_square = dist_square.reshape(
            batch, Max_sources, num_modes, 1
        )  # [batch, max_sources, num_modes, 1]

        if self.log_var_pred:
            return (
                -(self.output_dim / 2) * np.log(2 * np.pi)
                - (self.output_dim / 2) * sigma_pred
                - dist_square / (2 * np.exp(sigma_pred))
            )
        else:
            return (
                -(self.output_dim / 2) * np.log(2 * np.pi)
                - self.output_dim * np.log(sigma_pred)
                - dist_square / (2 * sigma_pred**2)
            )

    def np_compute_negative_log_likelihood(
        self,
        mu_pred,
        sigma_pred,
        pi_pred,
        source_activity_target,
        target_position,
    ):
        """Negative log-likelihood loss computation.

        Args:
            mu_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,output_dim)
            sigma_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,output_dim)
            pi_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,1)
            source_activity_target torch.tensor): Input tensor of shape (batch,Max_sources)
            target_position (torch.tensor): Input tensor of shape (batch,Max_sources,output_dim)

        Returns:
            loss (torch.tensor)
        """
        batch, num_modes, _ = mu_pred.shape
        max_sources = source_activity_target.shape[1]

        # Log probabilities of pi
        log_pi_pred_stacked = np.log(pi_pred)  # [batch, num_modes, 1]

        # Expand targets and predictions to match each other's shapes for batch-wise computation
        expanded_direction_of_arrival_target = np.expand_dims(target_position, 2)
        expanded_direction_of_arrival_target = np.tile(
            expanded_direction_of_arrival_target, (1, 1, num_modes, 1)
        )  # [batch, max_sources, num_modes, 2]
        expanded_mu_pred_stacked = np.expand_dims(mu_pred, 1)
        expanded_mu_pred_stacked = np.tile(
            expanded_mu_pred_stacked, (1, max_sources, 1, 1)
        )  # [batch, max_sources, num_modes, 2]

        expanded_sigma_pred_stacked = np.expand_dims(sigma_pred, 1)
        expanded_sigma_pred_stacked = np.tile(
            expanded_sigma_pred_stacked, (1, max_sources, 1, 1)
        )  # [batch, max_sources, num_modes, 1]

        # Call to log_normal_kernel function (not converted as per request)
        log_normal_values = self.log_normal_kernel(
            expanded_direction_of_arrival_target,
            expanded_mu_pred_stacked,
            expanded_sigma_pred_stacked,
        )

        # Include the log probabilities of pi and compute the log-sum-exp
        total_log_prob = log_normal_values + np.expand_dims(
            log_pi_pred_stacked, 1
        )  # [batch, max_sources, num_modes, 1]
        log_sum_exp = logsumexp(total_log_prob, axis=2)
        # np.log(
        # np.sum(np.exp(total_log_prob), axis=2)
        # )  # [batch, max_sources, 1]

        # Mask out inactive sources
        # active_source_mask = np.expand_dims(
        #     source_activity_target, -1
        # )  # [batch, max_sources, 1]

        masked_log_sum_exp = log_sum_exp * source_activity_target
        active_source_mask_sum = np.nansum(source_activity_target, axis=1)  # [batch, 1]
        active_source_mask_sum = np.tile(
            active_source_mask_sum, (1, max_sources)
        )  # [batch, max_sources]

        mask = active_source_mask_sum > 0
        masked_log_sum_exp = masked_log_sum_exp.squeeze(-1)

        masked_log_sum_exp[mask] = (
            masked_log_sum_exp[mask] / active_source_mask_sum[mask]
        )  # [batch, max_sources]

        # Compute mean negative log likelihood
        # NLL = -np.nansum(masked_log_sum_exp) / batch
        NLL = -np.sum(masked_log_sum_exp) / batch

        return NLL

    def batch_compute_negative_log_likelihood(self, predictions, targets):
        mu_pred, sigma_pred, pi_pred = predictions
        target_position, source_activity_target = targets

        # Put everything in cpu/numpy
        mu_pred = mu_pred.detach().cpu().numpy()
        sigma_pred = sigma_pred.detach().cpu().numpy()
        pi_pred = pi_pred.detach().cpu().numpy()
        target_position = target_position.detach().cpu().numpy()
        source_activity_target = source_activity_target.detach().cpu().numpy()

        return self.np_compute_negative_log_likelihood(
            mu_pred=mu_pred,
            sigma_pred=sigma_pred,
            pi_pred=pi_pred,
            source_activity_target=source_activity_target,
            target_position=target_position,
        )

    def wta_risk(self, test_loader, device):

        risk_value = torch.tensor(0.0, device=device)

        criterion = mhloss(mode="wta", distance="euclidean-squared")

        for _, data in enumerate(test_loader):
            # Move the input and target tensors to the device

            data_t = data[0].to(device)
            data_target_position = data[1].to(device)
            data_source_activity_target = data[2].to(device)

            # Forward pass
            outputs = self(data_t.float().reshape(-1, 1))

            # Compute the loss
            risk_value += criterion(
                outputs, (data_target_position, data_source_activity_target)
            )

        return risk_value / len(test_loader)

    def sample(self, n_samples, mu_pred, sigma_pred, pi_pred):
        # mu_pred of shape (batch_size, num_modes, output_dim)
        # sigma_pred of shape (batch_size, num_modes, 1)
        # pi_pred of shape (batch_size, num_modes, 1)

        mu_pred = mu_pred.detach().cpu().numpy()
        sigma_pred = sigma_pred.detach().cpu().numpy()
        pi_pred = pi_pred.detach().cpu().numpy()

        rng = np.random.default_rng()
        batch_size = mu_pred.shape[0]
        num_modes = mu_pred.shape[1]

        hyps = np.zeros((batch_size, n_samples, self.output_dim))
        confs = np.ones((batch_size, n_samples, 1))

        for batch_idx in range(batch_size):

            components = rng.choice(
                num_modes, size=n_samples, p=pi_pred[batch_idx, :, 0]
            )
            selected_mus = mu_pred[batch_idx, components, :]  # (n_samples,output_dim)
            selected_sigmas = sigma_pred[
                batch_idx, components, :
            ]  # (n_samples,output_dim)
            if self.log_var_pred is True:
                hyps[batch_idx] = rng.normal(
                    loc=selected_mus, scale=np.exp(selected_sigmas / 2)
                )
            else:
                hyps[batch_idx] = rng.normal(loc=selected_mus, scale=selected_sigmas)

        confs = confs / np.sum(confs, axis=1, keepdims=True)

        return hyps, confs

    def loss(self):
        return nll_loss(log_var_pred=self.log_var_pred)

    def prepare_predictions_emd(self, predictions):

        mu_stacked, sigma_stacked, pi_stacked = predictions

        hyps_stacked, conf_stacked = self.sample(
            n_samples=self.hparams["N_samples_predicted_dist"],
            mu_pred=mu_stacked,
            sigma_pred=sigma_stacked,
            pi_pred=pi_stacked,
        )

        # hyps_stacked = hyps_stacked.detach().cpu().numpy().astype(np.float32) # [batchxTxself.num_hypothesisxoutput_dim]
        # conf_stacked = (conf_stacked.detach().cpu().numpy().astype(np.float32))  # [batchxTxself.num_hypothesisx1]

        return (hyps_stacked.astype(np.float32), conf_stacked.astype(np.float32))

    def prepare_predictions_oracle(self, predictions):

        mu_stacked, _, _ = predictions

        hyps_stacked = mu_stacked

        conf_stacked = torch.ones_like(hyps_stacked[:, :, 0:1])
        conf_stacked = conf_stacked / conf_stacked.sum(dim=-2, keepdim=True)

        return (hyps_stacked, conf_stacked)

    def prepare_predictions_nll(self, predictions):
        # mu_pred, sigma_pred, pi_pred = predictions
        return predictions

    def prepare_predictions_mse(self, predictions):
        # Return the ponderated means by the pis
        mu_stacked, sigma_stacked, pi_stacked = predictions
        return (mu_stacked * pi_stacked).sum(dim=-2)

    def denormalize_predictions(self, predictions, mean_scaler, std_scaler):
        # mean and std scalers are the ones used for the target variable
        # shape [batchself.num_hypothesisxoutput_dim], [batchxself.num_hypothesisx1]
        mus = predictions[0]
        sigmas = predictions[1]
        pis = predictions[2]

        mus = mus * std_scaler + mean_scaler

        if self.log_var_pred is True:
            sigmas = (
                2 * np.log(std_scaler) + sigmas
            )  # log((std_scaler*sigma)^2) = 2*log(std_scaler) + log(sigma^2)
        else:
            sigmas = sigmas * std_scaler

        return (mus, sigmas, pis)

    def denormalize_targets(self, targets, mean_scaler, std_scaler):
        # targets[0] of shape [batch,Max_sources,output_dim]
        # targets[1] of shape [batch,Max_sources,1]

        target_position = targets[0]

        target_position = target_position * std_scaler + mean_scaler

        return target_position, targets[1]

    def check_nll_normalization_1d(self, predictions):

        # predictions: mu_pred, sigma_pred, pi_pred of shape [batch, num_hypothesis, output_dim], [batch, num_hypothesis, 1], [batch, num_hypothesis, 1]

        mus, sigmas, pis = predictions
        # Consider only one element of the batch

        N_samples_NLL_computed = 10
        integrals = 0

        for batch_elt_target in range(N_samples_NLL_computed):

            mu_pred = mus[batch_elt_target]
            sigma_pred = sigmas[batch_elt_target]
            pi_pred = pis[batch_elt_target]

            N_samples = 500
            square_size = 5
            # target_position = np.random.uniform(-square_size,square_size,(N_samples,1, 1))
            target_position = np.linspace(-square_size, square_size, N_samples).reshape(
                N_samples, 1, 1
            )
            source_activity_target = np.ones((N_samples, 1, 1), dtype=bool)

            LL = 0

            for batch_elt in range(N_samples):

                predictions = (
                    mu_pred.unsqueeze(0),
                    sigma_pred.unsqueeze(0),
                    pi_pred.unsqueeze(0),
                )
                targets = (
                    torch.tensor(target_position[batch_elt]).float(),
                    torch.tensor(source_activity_target[batch_elt]).float(),
                )

                NLL_elt = self.batch_compute_negative_log_likelihood(
                    predictions, targets
                )
                LL += np.exp(-NLL_elt)

            integrals += (LL * 2 * square_size) / N_samples

        return integrals / N_samples_NLL_computed
