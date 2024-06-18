import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from losses import mhloss, mhconfloss, nll_loss

from density_estimation import VoronoiDE


class Smcl(nn.Module):
    def __init__(self, hparams, num_hypothesis, num_hidden_units, restrict_to_square):
        """Constructor for the multi-hypothesis network.

        Args:
            num_hypothesis (int): Number of output hypotheses.
            num_hidden_units (int, optional): _description_. Defaults to 256.
        """
        super(Smcl, self).__init__()
        self._hparams = hparams
        self.num_hypothesis = num_hypothesis
        self.fc1 = nn.Linear(
            1, num_hidden_units
        )  # input layer (1 inputs -> num_hidden_units hidden units)
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(
            num_hidden_units, num_hidden_units
        )  # hidden layer (num_hidden_units hidden units -> num_hidden_units hidden units)
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.final_layers = nn.ModuleDict()

        self.restrict_to_square = restrict_to_square

        for k in range(self.num_hypothesis):
            self.final_layers["hyp_" + "{}".format(k)] = nn.Linear(
                in_features=num_hidden_units, out_features=2
            )

    def forward(self, x):
        """Forward pass of the multi-hypothesis network.

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis (uniform for the classical sMCL model). Shape [batchxself.num_hypothesisx1]
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        outputs_hyps = []

        for k in range(self.num_hypothesis):
            if self.restrict_to_square:
                outputs_hyps.append(
                    F.tanh(self.final_layers["hyp_" + "{}".format(k)](x))
                )
            else:
                outputs_hyps.append(
                    (self.final_layers["hyp_" + "{}".format(k)](x))
                )  # Size [batchxoutput_dim]

        hyp_stacked = torch.stack(
            outputs_hyps, dim=-2
        )  # Shape [batchxself.num_hypothesisxoutput_dim]
        assert hyp_stacked.shape == (x.shape[0], self.num_hypothesis, 2)
        confs = torch.ones_like(hyp_stacked[:, :, 0]).unsqueeze(
            -1
        )  # Shape [batchxself.num_hypothesisx1]
        return hyp_stacked, confs

    def loss(self):
        return mhloss(mode=self._hparams["training_wta_mode"], single_target_loss=True)
