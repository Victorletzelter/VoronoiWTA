import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import math

from src.utils.losses import mhloss, mhconfloss

from .methodsLighting import methodsLighting
from .density_estimation import VoronoiDE


class Histogram(methodsLighting, VoronoiDE):
    def __init__(
        self,
        sizes,
        hparams,
        kernel_type,
        scaling_factor,
        closed_form_vol,
        restrict_to_square,
        square_size,
        hit_and_run_sampling,
        hidden_layers,
        input_dim,
        output_dim,
        num_hypothesis,
    ):
        """Constructor for the hisogram like conditional VDE estimator.

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

        self._hparams = hparams
        self.sizes = sizes
        self.restrict_to_square = restrict_to_square

        # Initialize ModuleList for layers and ModuleList for activation functions
        self.layers = nn.ModuleList()
        self.activation_functions = nn.ModuleList()
        self.final_conf_layers = nn.ModuleDict()

        # Construct the architecture
        self.construct_layers(hidden_layers)

        # Construct the final layers
        self.construct_final_layers(hidden_layers[-1])

        self.num_hypothesis = math.prod(self.sizes)

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
        Constructs the confidence layers based on the number of hypotheses.
        """
        for indices in itertools.product(*(range(size) for size in self.sizes)):
            # Construct the layer name based on its indices in the N-dimensional grid
            layer_name = "_".join(
                ["dim" + str(i) + "_" + str(index) for i, index in enumerate(indices)]
            )
            self.final_conf_layers[layer_name] = nn.Linear(
                in_features=last_hidden_size, out_features=1
            )

    def generate_hypotheses_grid(self, batch_size, device):
        """Generate an N-dimensional grid of hypotheses.

        Returns:
            hypotheses_grid (torch.Tensor): Grid of hypotheses.
        """
        # Generate the grid points for each dimension.
        grid_points = []
        for size in self.sizes:
            # Compute the midpoints for each interval in this dimension.
            linspace = torch.linspace(-1, 1, size + 1, device=device)
            midpoints = (linspace[:-1] + linspace[1:]) / 2
            grid_points.append(midpoints)

        # Combine the grid points from all dimensions into a grid.
        # The reshaping and repeating is done to align each dimension's points across the grid.
        grids = []
        for i, points in enumerate(grid_points):
            # Add singleton dimensions ('1') so that this array broadcasts correctly when we combine them.
            shape = [1] * (self.output_dim + 1)  # +1 because of batch size
            shape[i + 1] = len(points)  # i+1 to skip batch size dimension
            grids.append(
                points.view(*shape).repeat(
                    batch_size,
                    *[self.sizes[j] if j != i else 1 for j in range(self.output_dim)]
                )
            )

        # Combine all individual dimensional grids into a single tensor.
        return torch.stack(
            grids, dim=-1
        )  # Shape: [batch_size] + self.sizes + [self.output_dim]

    def generate_hypotheses_2d_grid(self, batch_size, device):
        """Generate a grid of hypotheses.

        Returns:
            hypotheses_grid (torch.Tensor): Grid of hypotheses. Shape [self.num_rowsxself.num_colsx2]
        """

        rows = torch.linspace(-1, 1, self.num_rows + 1, device=device)
        cols = torch.linspace(-1, 1, self.num_cols + 1, device=device)

        new_rows = (rows[:-1] + rows[1:]) / 2
        new_cols = (cols[:-1] + cols[1:]) / 2

        rows = new_rows.view(1, -1, 1, 1).repeat(
            batch_size, 1, self.num_cols, 1
        )  # Shape: [batch_size,num_rows, num_cols, 1]
        cols = new_cols.view(1, 1, -1, 1).repeat(
            batch_size, self.num_rows, 1, 1
        )  # Shape: [batch_size,num_rows, num_cols, 1]

        return torch.cat(
            (rows, cols), dim=-1
        )  # Shape: [batch_size,num_rows, num_cols, 2]

    def forward(self, x):
        """For pass of the multi-hypothesis network with confidence (rMCL).

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis. Shape [batchxself.num_hypothesisx1]
        """
        # Pass input through each layer
        for layer, activation in zip(self.layers, self.activation_functions):
            x = activation(layer(x))

        # outputs_hyps = []
        confidences = []

        # Assuming 'x' is your input to these layers
        for indices in itertools.product(*(range(size) for size in self.sizes)):
            # Construct the layer name based on its indices in the N-dimensional grid
            layer_name = "_".join(
                ["dim" + str(i) + "_" + str(index) for i, index in enumerate(indices)]
            )

            # Apply the layer and sigmoid to 'x', then append to confidences
            confidences.append(F.sigmoid(self.final_conf_layers[layer_name](x)))

        hyp_stacked = self.generate_hypotheses_grid(
            x.shape[0], x.device
        )  # [batchxself.num_rowsxself.num_colsx1]
        conf_stacked = torch.stack(
            confidences, dim=-2
        )  # [batchxself.num_rowsxself.num_colsx1]

        hyp_stacked = hyp_stacked.view(x.shape[0], -1, self.output_dim)
        conf_stacked = conf_stacked.view(x.shape[0], -1, 1)

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
            outputs = self(data_t)  # .float().reshape(-1,1))

            # Compute the loss
            risk_value += criterion(
                outputs, (data_target_position, data_source_activity_target)
            )

        return risk_value / len(test_loader)

    def loss(self):
        return mhconfloss(
            number_unconfident=self._hparams["training_number_unconfident"],
            mode=self._hparams["training_wta_mode"],
            rejection_method=self._hparams["training_rejection_method"],
            epsilon=self._hparams["training_epsilon"],
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

            predictions_emd = (
                hyps.cpu().numpy().astype(np.float32),
                confs.cpu().numpy().astype(np.float32),
            )

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
