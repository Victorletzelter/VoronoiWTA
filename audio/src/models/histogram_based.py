from argparse import Namespace
from .modules import (
    AbstractLocalizationModule,
    FeatureExtraction,
    GridLocalizationOutput,
)
import torch
import torch.nn as nn
from typing import Tuple
from src.utils.losses import MHCONFSELLoss
import numpy as np

from src.models.density_estimation import VoronoiDE


class HistogramSELDNet(AbstractLocalizationModule, VoronoiDE):
    def __init__(self, dataset_path: str, cv_fold_idx: int, hparams: Namespace) -> None:

        AbstractLocalizationModule.__init__(self, dataset_path, cv_fold_idx, hparams)
        VoronoiDE.__init__(
            self,
            kernel_type=hparams["kernel_type"],
            scaling_factor=hparams["scaling_factor"],
            kde_mode=hparams["kde_mode"],
            kde_weighted=hparams["kde_weighted"],
        )

        self.kernel_type = hparams["kernel_type"]
        self.scaling_factor = hparams["scaling_factor"]
        self.kde_mode = hparams["kde_mode"]
        self.kde_weighted = hparams["kde_weighted"]

        self.closed_form_vol = hparams["closed_form_vol"]
        self.hit_and_run_sampling = hparams["hit_and_run_sampling"]
        self.n_directions = hparams["n_directions"]
        self.square_size = hparams["square_size"]

        self.hparams["num_hypothesis"] = self.hparams["nrows"] * self.hparams["ncols"]

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

        self.localization_output = GridLocalizationOutput(
            input_dim=2 * hparams["hidden_dim"],
            nrows=hparams["nrows"],
            ncols=hparams["ncols"],
        )

        # In the localization module, the input_dim is to 2 * hparams.hidden_dim if bidirectional=True in the GRU.

    def get_loss_function(self) -> nn.Module:
        return MHCONFSELLoss(
            self.hparams["max_num_sources"],
            alpha=self.hparams["alpha"],
            mode="wta",
            top_n=1,
            distance=self.hparams["distance"],
            epsilon=self.hparams["epsilon"],
            conf_weight=1,
            rejection_method="all",
            number_unconfident=self.hparams["number_unconfident"],
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

        scores_pred_grid = self.localization_output(output)

        hyps_pred_grid = self.generate_hypotheses_grid(
            T=scores_pred_grid.shape[1],
            batch_size=scores_pred_grid.shape[0],
            device=scores_pred_grid.device,
        )
        hyps_pred_grid = hyps_pred_grid.view(
            scores_pred_grid.shape[0], scores_pred_grid.shape[1], -1, 2
        )
        scores_pred_grid = scores_pred_grid.view(
            scores_pred_grid.shape[0], scores_pred_grid.shape[1], -1, 1
        )
        meta_data = {}

        return hyps_pred_grid, scores_pred_grid, meta_data

    def generate_hypotheses_grid(self, T=None, batch_size=None, device="cpu"):
        """Generate a grid of hypotheses.

        Returns:
            hypotheses_grid (torch.Tensor): Grid of hypotheses. Shape [self.num_rowsxself.num_colsx2]
        """
        hypothesis = torch.zeros(
            size=(self.hparams["nrows"], self.hparams["ncols"], 2), device=device
        )

        rows = torch.linspace(
            -torch.pi, torch.pi, self.hparams["nrows"] + 1, device=device
        )
        cols = torch.linspace(
            -torch.pi / 2, torch.pi / 2, self.hparams["ncols"] + 1, device=device
        )
        rows = (rows[:-1] + rows[1:]) / 2
        cols = (cols[:-1] + cols[1:]) / 2

        # Reshape for broadcasting
        rows = rows.view(-1, 1, 1).repeat(
            1, self.hparams["ncols"], 1
        )  # Shape: [self.hparams['nrows'], num_cols, 1]
        cols = cols.view(1, -1, 1).repeat(
            self.hparams["nrows"], 1, 1
        )  # Shape: [num_rows, num_cols, 1]

        # Create the hypothesis tensor
        hypothesis = torch.cat((rows, cols), dim=2)  # Shape: [num_rows, num_cols, 2]

        if T is not None and batch_size is not None:
            hypothesis = hypothesis.unsqueeze(0).expand(T, -1, -1, -1)
            hypothesis = hypothesis.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            return hypothesis.reshape(batch_size, T, -1, 2)
        else:
            return hypothesis.reshape(batch_size, T, -1, 2)

    def prepare_predictions_emd(self, predictions):

        batch = predictions[0].shape[0]
        T = predictions[0].shape[1]
        hyps = predictions[0]
        confs = predictions[1] / predictions[1].sum(dim=-2, keepdim=True)

        if (
            "kernel_mode_emd" in self.hparams.keys()
            and self.hparams["kernel_mode_emd"] is True
        ):

            hyps_DOAs_pred_stacked = np.zeros(
                (batch, T, self.hparams["N_samples_predicted_dist"], 2)
            )
            for batch_elt in range(batch):
                for t in range(T):
                    # if source_activity_target[batch_elt,t,:].sum() > 0 :
                    hyps_DOAs_pred_stacked[batch_elt, t, :, :] = self.sample(
                        n_samples=self.hparams["N_samples_predicted_dist"],
                        hyps_pred=hyps[batch_elt, t, :, :],
                        confs_pred=confs[batch_elt, t, :, 0],
                    )

            hyps_DOAs_pred_stacked = torch.from_numpy(hyps_DOAs_pred_stacked).to(
                predictions[0].device
            )
            conf_stacked = torch.ones_like(hyps_DOAs_pred_stacked[:, :, :, 0:1])
            conf_stacked = conf_stacked / conf_stacked.sum(dim=-2, keepdim=True)
            predictions_emd = (hyps_DOAs_pred_stacked, conf_stacked)

        else:
            predictions_emd = predictions

        return predictions_emd

    def prepare_predictions_oracle(self, predictions):

        return predictions

    def prepare_predictions_nll(self, predictions):
        # hyps_stacked, conf_stacked shape [batchxTxself.num_hypothesisxoutput_dim], [batchxTxself.num_hypothesisx1]

        hyps = predictions[0]
        confs = predictions[1] / predictions[1].sum(dim=-2, keepdim=True)
        return (hyps, confs)
