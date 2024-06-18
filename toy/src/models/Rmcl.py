import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.losses import mhloss, mhconfloss
from src.models.density_estimation import VoronoiDE
from .methodsLighting import methodsLighting


class Rmcl(methodsLighting, VoronoiDE):
    def __init__(
        self,
        hparams,
        num_hypothesis,
        kernel_type,
        scaling_factor,
        closed_form_vol,
        restrict_to_square,
        square_size,
        hit_and_run_sampling,
        hidden_layers,
        input_dim,
        output_dim,
    ):
        """Constructor for the multi-hypothesis network with confidence (rMCL).

        Args:
            num_hypothesis (int): Number of output hypotheses.
        """
        self.output_dim = output_dim
        self.input_dim = input_dim

        methodsLighting.__init__(self, hparams)
        VoronoiDE.__init__(
            self,
            kernel_type=kernel_type,
            scaling_factor=scaling_factor,
            closed_form_vol=closed_form_vol,
            square_size=square_size,
            hit_and_run_sampling=hit_and_run_sampling,
        )

        self.name = "rmcl"
        self._hparams = hparams
        self.num_hypothesis = num_hypothesis
        self.restrict_to_square = restrict_to_square

        # Initialize ModuleList for layers and ModuleList for activation functions
        self.layers = nn.ModuleList()
        self.activation_functions = nn.ModuleList()
        self.final_hyp_layers = nn.ModuleDict()
        self.final_conf_layers = nn.ModuleDict()

        # Construct the architecture
        self.construct_layers(hidden_layers)

        # Construct the final layers
        self.construct_final_layers(hidden_layers[-1])

        # check which device is available
        if torch.cuda.is_available():
            self._hparams["device"] = torch.device("cuda")
        else:
            self._hparams["device"] = torch.device("cpu")

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
            self.final_hyp_layers[f"hyp_{k}"] = nn.Linear(
                last_hidden_size, self.output_dim
            )
            self.final_conf_layers[f"hyp_{k}"] = nn.Linear(last_hidden_size, 1)

    def forward(self, x):
        """For pass of the multi-hypothesis network with confidence (rMCL).

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis. Shape [batchxself.num_hypothesisx1]
        """
        # Pass input through each layer
        for layer, activation in zip(self.layers, self.activation_functions):
            x = activation(layer(x))

        outputs_hyps = []
        confidences = []

        for k in range(self.num_hypothesis):
            if self.restrict_to_square:
                outputs_hyps.append(
                    F.tanh(self.final_hyp_layers[f"hyp_{k}"](x))
                )  # Size [batchxoutput_dim]
            else:
                outputs_hyps.append(
                    (self.final_hyp_layers[f"hyp_{k}"](x))
                )  # Size [batchxoutput_dim]
            confidences.append(
                F.sigmoid(self.final_conf_layers[f"hyp_{k}"](x))
            )  # Size [batchx1])

        hyp_stacked = torch.stack(
            outputs_hyps, dim=-2
        )  # Shape [batchxself.num_hypothesisxoutput_dim]
        assert hyp_stacked.shape == (x.shape[0], self.num_hypothesis, self.output_dim)
        conf_stacked = torch.stack(confidences, dim=-2)  # [batchxself.num_hypothesisx1]
        assert conf_stacked.shape == (x.shape[0], self.num_hypothesis, 1)

        return hyp_stacked, conf_stacked

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

    def loss(self):
        return mhconfloss(
            mode=self._hparams["training_wta_mode"],
            top_n=self._hparams["training_top_n"],
            epsilon=self._hparams["training_epsilon"],
            distance=self._hparams["training_distance"],
            conf_weight=self._hparams["training_conf_weight"],
            rejection_method=self._hparams["training_rejection_method"],
            number_unconfident=self._hparams["training_number_unconfident"],
            output_dim=self.output_dim,
        )

    def prepare_predictions_emd(self, predictions):

        batch = predictions[0].shape[0]
        hyps = predictions[0]
        confs = predictions[1] / predictions[1].sum(dim=-2, keepdim=True)

        # Convert to cpu/numpy
        hyps = hyps.cpu().numpy()
        confs = confs.cpu().numpy()

        if (
            "kernel_mode_emd" in self.hparams.keys()
            and self.hparams["kernel_mode_emd"] is True
        ):

            hyps_pred_stacked = np.zeros(
                (batch, self.hparams["N_samples_predicted_dist"], 2)
            )
            for batch_elt in range(batch):
                # if source_activity_target[batch_elt,t,:].sum() > 0 :
                hyps_pred_stacked[batch_elt, :, :] = self.sample(
                    n_samples=self.hparams["N_samples_predicted_dist"],
                    hyps_pred=hyps[batch_elt, :, :],
                    confs_pred=confs[batch_elt, :, 0],
                )

            conf_stacked = np.ones_like(hyps_pred_stacked[:, :, 0:1])
            conf_stacked = conf_stacked / conf_stacked.sum(axis=-2, keepdims=True)

            predictions_emd = (
                hyps_pred_stacked.astype(np.float32),
                conf_stacked.astype(np.float32),
            )

        else:
            predictions_emd = (hyps.astype(np.float32), confs.astype(np.float32))

        return predictions_emd

    def prepare_predictions_oracle(self, predictions):
        # hyps_stacked, conf_stacked
        return predictions

    def prepare_predictions_nll(self, predictions):
        # hyps_stacked, conf_stacked shape [batchself.num_hypothesisxoutput_dim], [batchxself.num_hypothesisx1]

        hyps = predictions[0]
        confs = predictions[1] / predictions[1].sum(dim=-2, keepdim=True)
        return (hyps, confs)

    def prepare_predictions_mse(self, predictions):

        hyps = predictions[0]  # shape [batchself.num_hypothesisxoutput_dim]
        confs = predictions[1] / predictions[1].sum(
            dim=-2, keepdim=True
        )  # shape [batchself.num_hypothesisx1]

        # Return the ponderated mean of the hypotheses
        return (hyps * confs).sum(dim=-2)  # shape [batchxoutput_dim]

    def denormalize_predictions(self, predictions, mean_scaler, std_scaler):
        # mean and std scalers are the ones used for the target variable
        # shape [batchself.num_hypothesisxoutput_dim], [batchxself.num_hypothesisx1]
        hyps = predictions[0]
        confs = predictions[1]

        hyps = hyps * std_scaler + mean_scaler

        return (hyps, confs)

    def denormalize_targets(self, targets, mean_scaler, std_scaler):
        # targets[0] of shape [batch,Max_sources,output_dim]
        # targets[1] of shape [batch,Max_sources,1]

        target_position = targets[0]

        target_position = target_position * std_scaler + mean_scaler

        return target_position, targets[1]
