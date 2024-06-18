import abc
from src.data.data_handlers import TUTSoundEvents
from src.metrics import EMD_module, Oracle_module
import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import torch.nn.functional as F
from src.utils.losses import WTARisk
import torch
import torch.nn as nn
from typing import Tuple


class AbstractLocalizationModule(ptl.LightningModule, abc.ABC):
    def __init__(self, dataset_path: str, cv_fold_idx: int, hparams):

        super(AbstractLocalizationModule, self).__init__()

        self.dataset_path = dataset_path
        self.cv_fold_idx = cv_fold_idx

        self._hparams = hparams
        self.max_num_sources = hparams["max_num_sources"]

        if "max_num_overlapping_sources_test" in hparams:
            self.max_num_overlapping_sources_test = hparams[
                "max_num_overlapping_sources_test"
            ]

        else:
            self.max_num_overlapping_sources_test = self.max_num_sources

        self.sigma_classes_rad = [np.deg2rad(x) for x in self._hparams["sigma_classes"]]
        self._hparams["sigma_classes_torch_rad"] = torch.tensor(
            self.sigma_classes_rad, dtype=torch.float32, device="cuda:0"
        )

        conf_mode = "CONF" in self._hparams["name"] or "Grid" in self._hparams["name"]
        grid_mode = "Grid" in self._hparams["name"]
        class_mode = "cls" in self._hparams["name"]

        self.infer_custom_metrics()

        if (
            "MH" in self._hparams["name"]
            or "CONF" in self._hparams["name"]
            or "NLL" in self._hparams["name"]
        ):
            activity_mode = False
        else:
            activity_mode = True

        self.emd_metric_instance = EMD_module(
            data_loading_mode=self.hparams["data_loading_mode"],
            distance=self.hparams["dist_type_eval"],
            num_sources_per_sample_min=self.hparams["num_sources_per_sample_min"],
            rad2deg=True,
            sigma_classes_deg=self.hparams["sigma_classes"],
            sigma_points_mode=self.hparams["sigma_points_mode"],
            class_mode=class_mode,
            N_samples_mog=self.hparams["N_samples_mog"],
            conf_mode=conf_mode,
            grid_mode=grid_mode,
        )

        self.oracle_instance = Oracle_module(
            distance=self.hparams["dist_type_eval"],
            activity_mode=activity_mode,
            rad2deg=True,
            class_mode=class_mode,
            computation_type="dirac",
            print_hyp_idx=False,
            num_sources_per_sample_min=self.hparams["num_sources_per_sample_min"],
            num_sources_per_sample_max=self.hparams["num_sources_per_sample_max"],
        )

    @property
    def hparams(self):
        return self._hparams

    @abc.abstractmethod
    def forward(
        self, audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams["learning_rate"], weight_decay=0.0
        )

        lr_lambda = lambda epoch: self.hparams["learning_rate"] * np.minimum(
            (epoch + 1) ** -0.5,
            (epoch + 1) * (self.hparams["num_epochs_warmup"] ** -1.5),
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def training_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> Dict:
        predictions, targets = self._process_batch(batch)

        if self._hparams.online is True and self._hparams.offline is True:
            raise ValueError("Online and offline cannot be both True")

        if self._hparams.online is True and self._hparams.offline is False:
            # source_activity_target (batch, num_frames_per_chunk, self.max_num_sources)
            # direction_of_arrival_target  (batch, num_frames_per_chunk, self.max_num_sources,2)
            # source_activity_target_classes (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes)
            # direction_of_arrival_target_classes (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes,2)
            # source_activity, direction_of_arrival, targets[2], targets[3] = targets

            sigma_classes = self._hparams["sigma_classes_torch_rad"].view(
                1, 1, 1, -1, 1
            )
            sigma_classes = sigma_classes.expand(size=targets[3].shape)

            # Apply mask and add noise
            mask_active = (
                targets[2].unsqueeze(-1).expand(size=targets[3].shape) > 0
            )  # shape (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes,2)
            normal_dist = torch.randn(size=targets[3].shape, device=targets[3].device)
            normal_dist = torch.where(
                mask_active, normal_dist, torch.zeros_like(normal_dist)
            )

            targets[3] += normal_dist * sigma_classes
            targets[1] = targets[3].sum(dim=-2)  # sum over the classes

        loss, meta_data = self.loss_function(predictions, targets)

        output = {"loss": loss}
        self.log_dict(output)
        self.log_dict(meta_data)

        return output

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> Dict:

        with torch.no_grad():
            predictions, targets = self._process_batch(batch)

            if (
                self._hparams.online is True
                and self._hparams.offline is False
                and self._hparams.online_val is True
            ):

                sigma_classes = self._hparams["sigma_classes_torch_rad"].view(
                    1, 1, 1, -1, 1
                )
                sigma_classes = sigma_classes.expand(size=targets[3].shape)

                # Apply mask and add noise
                mask_active = (
                    targets[2].unsqueeze(-1).expand(size=targets[3].shape) > 0
                )  # shape (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes,2)
                normal_dist = torch.randn(
                    size=targets[3].shape, device=targets[3].device
                )
                normal_dist = torch.where(
                    mask_active, normal_dist, torch.zeros_like(normal_dist)
                )

                targets[3] += normal_dist * sigma_classes
                targets[3] = torch.where(
                    mask_active, targets[3], torch.zeros_like(targets[3])
                )
                targets[1] = targets[3].sum(dim=-2)  # sum over the classes

            loss, meta_data = self.loss_function(predictions, targets)

        output = {"val_loss": loss}
        self.log_dict(output)

        return output

    def validation_epoch_end(self, outputs: list) -> None:
        average_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log_dict({"val_loss": average_loss, "learning_rate": learning_rate})

        return average_loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        dataset_idx: int = 0,
        save_preds: bool = True,
    ) -> Dict:

        predictions, targets = self._process_batch(batch)

        ##################### IF WE ARE IN SYNTHETIC PERTURBATION MODE AT TEST TIME
        if self._hparams.online is True and self._hparams.offline is False:
            # source_activity_target (batch, num_frames_per_chunk, self.max_num_sources)
            # direction_of_arrival_target  (batch, num_frames_per_chunk, self.max_num_sources,2)
            # source_activity_target_classes (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes)
            # direction_of_arrival_target_classes (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes,2)
            # source_activity, direction_of_arrival, targets[2], targets[3] = targets
            # tuple with the same size as targets
            # Initialize an empty list instead of a tuple
            targets_sampled = [None] * len(targets)

            # Fill the list with cloned elements from `targets`
            for i in range(len(targets)):
                targets_sampled[i] = targets[i].clone()

            sigma_classes = self._hparams["sigma_classes_torch_rad"].view(
                1, 1, 1, -1, 1
            )
            sigma_classes = sigma_classes.expand(size=targets[3].shape)

            # Apply mask and add noise
            mask_active = (
                targets[2].unsqueeze(-1).expand(size=targets[3].shape) > 0
            )  # shape (batch, num_frames_per_chunk, self.max_num_sources, self.num_unique_classes,2)
            normal_dist = torch.randn(size=targets[3].shape, device=targets[3].device)
            normal_dist = torch.where(
                mask_active, normal_dist, torch.zeros_like(normal_dist)
            )

            targets_sampled[3] += normal_dist * sigma_classes
            targets_sampled[1] = targets_sampled[3].sum(dim=-2)  # sum over the classes
        else:
            targets_sampled = targets  # If we are not in the synthetic perturbation mode, we take the original targets.
        #####################

        predictions_emd = self.prepare_predictions_emd(predictions=predictions)
        predictions_oracle = self.prepare_predictions_oracle(predictions=predictions)
        predictions_nll = self.prepare_predictions_nll(predictions=predictions)

        # We consider the (single) sampled targets, except for the EMD here, which requires the full target distribution.
        emd = self.emd_metric_instance.compute(
            predictions=predictions_emd, targets=targets
        )
        oracle = self.oracle_instance.compute(
            predictions=predictions_oracle,
            targets=targets_sampled,
            dataset_idx=dataset_idx,
            batch_idx=batch_idx,
        )

        risk = self.compute_risk(
            predictions=predictions_oracle, targets=targets_sampled
        )

        if (
            "check_normalization_nll" in self.hparams
            and self.hparams["check_normalization_nll"] is True
        ):
            integral = self.check_nll_normalization_sphere(predictions=predictions_nll)

        nll = self.batch_compute_negative_log_likelihood(
            predictions=predictions_nll, targets=targets_sampled
        )

        output = {
            "test_oracle_doa_error" + "_" + self.hparams["dist_type_eval"][0:4]: oracle,
            "test_emd_metric" + "_" + self.hparams["dist_type_eval"][0:4]: emd[0],
            "test_std_emd_metric" + "_" + self.hparams["dist_type_eval"][0:4]: emd[1],
            "test_wta_risk": risk[0],
            "nll": nll,
        }

        self.log_dict(output)

        return output

    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        audio_features, targets = batch
        predictions = self.forward(audio_features)

        return predictions, targets

    def train_dataloader(self) -> DataLoader:
        train_dataset = TUTSoundEvents(
            self.dataset_path,
            split="train",
            tmp_dir=self.hparams["tmp_dir"],
            test_fold_idx=self.cv_fold_idx,
            sequence_duration=self.hparams["sequence_duration"],
            chunk_length=self.hparams["chunk_length"],
            frame_length=self.hparams["frame_length"],
            num_fft_bins=self.hparams["num_fft_bins"],
            max_num_sources=self.hparams["max_num_sources"],
        )

        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = TUTSoundEvents(
            self.dataset_path,
            split="valid",
            tmp_dir=self.hparams["tmp_dir"],
            test_fold_idx=self.cv_fold_idx,
            sequence_duration=self.hparams["sequence_duration"],
            chunk_length=self.hparams["chunk_length"],
            frame_length=self.hparams["frame_length"],
            num_fft_bins=self.hparams["num_fft_bins"],
            max_num_sources=self.hparams["max_num_sources"],
        )

        return DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )

    def test_dataloader(self) -> List[DataLoader]:
        # During testing, a whole sequence is packed into one batch. The batch size set for training and validation
        # is ignored in this case.
        num_chunks_per_sequence = int(
            self.hparams["sequence_duration"] / self.hparams["chunk_length"]
        )

        test_loaders = []

        for num_overlapping_sources in range(
            1, min(self.max_num_overlapping_sources_test, 3) + 1
        ):
            test_dataset = TUTSoundEvents(
                self.dataset_path,
                split="test",
                tmp_dir=self.hparams["tmp_dir"],
                test_fold_idx=self.cv_fold_idx,
                sequence_duration=self.hparams["sequence_duration"],
                chunk_length=self.hparams["chunk_length"],
                frame_length=self.hparams["frame_length"],
                num_fft_bins=self.hparams["num_fft_bins"],
                max_num_sources=self.hparams["max_num_sources"],
                num_overlapping_sources=num_overlapping_sources,
            )

            test_loaders.append(
                DataLoader(
                    test_dataset,
                    shuffle=False,
                    batch_size=num_chunks_per_sequence,
                    num_workers=self.hparams["num_workers"],
                )
            )

        return test_loaders

    def compute_risk(self, predictions, targets):

        predictions = (predictions[0], None)
        # Compute the risk
        return WTARisk(
            max_num_sources=targets[0].shape[2],
            distance="spherical-squared",
            rad2deg=False,
        )(predictions=predictions, targets=targets)

    def infer_custom_metrics(self):

        self.custom_metrics = (
            "MH" in self.hparams["name"] or "Histogram" in self.hparams["name"]
        )

        self.custom_metrics_nll = False
        self.custom_metrics_grid = False

        if "NLL" in self.hparams["name"]:
            self.custom_metrics_nll = True


class FeatureExtraction(nn.Module):
    """CNN-based feature extraction originally proposed in [1].

    Args:
        num_steps_per_chunk: Number of time steps per chunk, which is required for correct layer normalization.

        num_fft_bins: Number of FFT bins used for spectrogram computation.

        dropout_rate: Dropout rate.

    References:
        [1] Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen, "Sound event localization and
            detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected
            Topics in Signal Processing (JSTSP 2018)
    """

    def __init__(
        self, num_steps_per_chunk: int, num_fft_bins: int, dropout_rate: float = 0.0
    ) -> None:
        """Initialization of CNNs-based layers for features extraction.

        Args:
            num_steps_per_chunk (int): Number of steps in each chunk.
            num_fft_bins (int): Number of frequencies calculated at each FFT computation.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(FeatureExtraction, self).__init__()
        # As the number of audio channels in the raw data is four, this number doubles after frequency features extraction (amplitude and phase).
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                8, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
        )

        self.layer_norm = nn.LayerNorm(
            [num_steps_per_chunk, int(num_fft_bins / 4)]
        )  # Layer normalization used
        # Statistics are calculated over the last two dimensions of the input.s

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Feature extraction forward pass.

        Args:
            audio_features (torch.Tensor): Input tensor with dimensions [batchx2*CxTxB], where batch is the batch size, T is the number of
        time steps per chunk, B is the number of FFT bins and C is the number of audio channels.

        Returns:
            torch.Tensor: Extracted features with dimension [batchxTxB/4].
        """
        output = self.conv_layer1(audio_features)  # Output shape [batchx64xTxB/8]
        output = self.conv_layer2(output)  # Output shape [batchx64xTxB/64]
        output = self.conv_layer3(output)  # Output shape [batchx64xTxB/256]
        output = output.permute(0, 2, 1, 3)  # Output shape [batchxTx64xB/256]
        batch_size, num_frames, _, _ = output.shape
        output = output.contiguous().view(
            batch_size, num_frames, -1
        )  # Output shape [batchxTxB/4]

        return self.layer_norm(output)


class LocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.

    Args:
        input_dim: Input dimension.

        max_num_sources: Maximum number of sound sources that should be represented by the module.
    """

    def __init__(self, input_dim: int, max_num_sources: int):
        super(LocalizationOutput, self).__init__()

        self.source_activity_output = nn.Linear(input_dim, max_num_sources)
        self.azimuth_output = nn.Linear(input_dim, max_num_sources)
        self.elevation_output = nn.Linear(input_dim, max_num_sources)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batchxTxD], where batch is the batch size, T is the number of time steps per
                      chunk and D is the input dimension.
        :return: Tuple containing the source activity tensor of size [batchxTxS] and the direction-of-arrival tensor with
                 dimensions [batchxTxSx2], where S is the maximum number of sources.
        """
        source_activity = self.source_activity_output(input)

        azimuth = self.azimuth_output(input)
        elevation = self.elevation_output(input)
        direction_of_arrival = torch.cat(
            (azimuth.unsqueeze(-1), elevation.unsqueeze(-1)), dim=-1
        )

        return source_activity, direction_of_arrival


class MHLocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.
    Args:
        input_dim: Input dimension at each time step.
        num_hypothesis: Number of hypothesis in the model.
    """

    def __init__(self, input_dim: int, num_hypothesis: int, output_dim: int = 2):
        super(MHLocalizationOutput, self).__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.num_hypothesis = num_hypothesis
        self.output_dim = output_dim

        self.doa_layers = nn.ModuleDict()

        for k in range(self.num_hypothesis):
            self.doa_layers["hyp_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=output_dim, device="cuda:0"
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batchxTxD], where batch is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: Stacked direciton of arrival hypothesis with shape [batchxTxself.num_hypothesisxoutput_dim]
        """
        directions_of_arrival = []
        batch, T = input.shape[0], input.shape[1]
        input_reshaped = input.reshape(
            -1, input.shape[-1]
        )  # Of shape [batch*T x input_dim]

        for k in range(self.num_hypothesis):
            directions_of_arrival.append(
                (self.doa_layers["hyp_" + "{}".format(k)](input_reshaped)).reshape(
                    batch, T, -1
                )
            )  # Size [batchxTxoutput_dim]

        hyp_stacked = torch.stack(
            directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_hypothesisxoutput_dim]

        ### OR
        # directions_of_arrival = self.doa_layers(input) # Size [batchxTx(2*self.num_hypothesis)]
        # hyps_splitted = torch.split(directions_of_arrival, [2 for i in range(num_hypothesis)], 2) #num_hypothesis-uples of elements of shape [batch,T,2]
        # hyps_stacked = torch.stack([h for h in hyps_splitted], dim=2) #Tuples of elements of shape [batch,T,num_hypothesis,2]

        return hyp_stacked


class MHCONFLocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.
    Args:
        input_dim: Input dimension at each time step.
        num_hypothesis: Number of hypothesis in the model.
    """

    def __init__(self, input_dim: int, num_hypothesis: int, output_dim: int = 2):
        super(MHCONFLocalizationOutput, self).__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.num_hypothesis = num_hypothesis
        self.output_dim = output_dim

        self.doa_layers = nn.ModuleDict()
        self.doa_conf_layers = nn.ModuleDict()

        for k in range(self.num_hypothesis):
            self.doa_layers["hyp_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=output_dim, device="cuda:0"
            )
            self.doa_conf_layers["hyp_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=1, device="cuda:0"
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batchxTxD], where batch is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: Stacked direciton of arrival hypothesis with shape [batchxTxself.num_hypothesisxoutput_dim]
        """
        directions_of_arrival = []
        associated_confidences = []
        batch, T = input.shape[0], input.shape[1]
        input_reshaped = input.reshape(
            -1, input.shape[-1]
        )  # Of shape [batch*T x input_dim]

        for k in range(self.num_hypothesis):
            directions_of_arrival.append(
                (self.doa_layers["hyp_" + "{}".format(k)](input_reshaped)).reshape(
                    batch, T, -1
                )
            )  # Size [batchxTxoutput_dim]
            associated_confidences.append(
                (
                    F.sigmoid(
                        self.doa_conf_layers["hyp_" + "{}".format(k)](input_reshaped)
                    )
                ).reshape(batch, T, -1)
            )  # Size [batchxTx1]

        hyp_stacked = torch.stack(
            directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_hypothesisxoutput_dim]
        conf_stacked = torch.stack(
            associated_confidences, dim=-2
        )  # Shape [batchxTxself.num_hypothesisx1]

        ### OR
        # directions_of_arrival = self.doa_layers(input) # Size [batchxTx(2*self.num_hypothesis)]
        # hyps_splitted = torch.split(directions_of_arrival, [2 for i in range(num_hypothesis)], 2) #num_hypothesis-uples of elements of shape [batch,T,2]
        # hyps_stacked = torch.stack([h for h in hyps_splitted], dim=2) #Tuples of elements of shape [batch,T,num_hypothesis,2]

        return hyp_stacked, conf_stacked


class VonMisesMixLocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.
    Args:
        input_dim: Input dimension at each time step.
        num_hypothesis: Number of hypothesis in the model.
    """

    def __init__(self, input_dim: int, num_modes: int, output_dim: int = 2):
        super().__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.num_modes = num_modes
        self.output_dim = output_dim

        self.mu_doa_layers = nn.ModuleDict()
        self.kappa_doa_layers = nn.ModuleDict()
        self.pi_doa_layers = nn.ModuleDict()

        for k in range(self.num_modes):
            self.mu_doa_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=output_dim, device="cuda:0"
            )
            self.kappa_doa_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=1, device="cuda:0"
            )
            self.pi_doa_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=1, device="cuda:0"
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batchxTxD], where batch is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: Stacked direciton of arrival hypothesis with shape [batchxTxself.num_modesxoutput_dim]
        """
        mu_directions_of_arrival = []
        kappa_directions_of_arrival = []
        pi_directions_of_arrival = []
        batch, T = input.shape[0], input.shape[1]
        input_reshaped = input.reshape(
            -1, input.shape[-1]
        )  # Of shape [batch*T x input_dim]

        for k in range(self.num_modes):
            mu_directions_of_arrival.append(
                (self.mu_doa_layers["mode_" + "{}".format(k)](input_reshaped)).reshape(
                    batch, T, -1
                )
            )  # Size [batchxTxoutput_dim]
            kappa_directions_of_arrival.append(
                (
                    torch.nn.ReLU()(
                        self.kappa_doa_layers["mode_" + "{}".format(k)](input_reshaped)
                    )
                ).reshape(batch, T, -1)
            )  # Size [batchxTx1]
            pi_directions_of_arrival.append(
                (self.pi_doa_layers["mode_" + "{}".format(k)](input_reshaped)).reshape(
                    batch, T, -1
                )
            )  # Size [batchxTx1]

        mu_stacked = torch.stack(
            mu_directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_modesxoutput_dim]
        kappa_stacked = torch.stack(
            kappa_directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_modesx1]
        pi_stacked = torch.stack(
            pi_directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_modesx1]
        pi_stacked = torch.nn.Softmax(dim=-2)(
            pi_stacked
        )  # Shape [batchxTxself.num_modesx1]

        return mu_stacked, kappa_stacked, pi_stacked


class LogKappaVonMisesMixLocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.
    Args:
        input_dim: Input dimension at each time step.
        num_hypothesis: Number of hypothesis in the model.
    """

    def __init__(self, input_dim: int, num_modes: int, output_dim: int = 2):
        super().__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.num_modes = num_modes
        self.output_dim = output_dim

        self.mu_doa_layers = nn.ModuleDict()
        self.log_kappa_doa_layers = nn.ModuleDict()
        self.pi_doa_layers = nn.ModuleDict()

        for k in range(self.num_modes):
            self.mu_doa_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=output_dim, device="cuda:0"
            )
            self.log_kappa_doa_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=1, device="cuda:0"
            )
            self.pi_doa_layers["mode_" + "{}".format(k)] = nn.Linear(
                in_features=input_dim, out_features=1, device="cuda:0"
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward pass.

        :param input: Input tensor with dimensions [batchxTxD], where batch is the batch size, T is the number of time steps per
                    chunk and D is the input dimension.
        :return: Stacked direciton of arrival hypothesis with shape [batchxTxself.num_modesxoutput_dim]
        """
        mu_directions_of_arrival = []
        log_kappa_directions_of_arrival = []
        pi_directions_of_arrival = []
        batch, T = input.shape[0], input.shape[1]
        input_reshaped = input.reshape(
            -1, input.shape[-1]
        )  # Of shape [batch*T x input_dim]

        for k in range(self.num_modes):
            mu_directions_of_arrival.append(
                (self.mu_doa_layers["mode_" + "{}".format(k)](input_reshaped)).reshape(
                    batch, T, -1
                )
            )  # Size [batchxTxoutput_dim]
            log_kappa_directions_of_arrival.append(
                (
                    (
                        self.log_kappa_doa_layers["mode_" + "{}".format(k)](
                            input_reshaped
                        )
                    )
                ).reshape(batch, T, -1)
            )  # Size [batchxTx1]
            pi_directions_of_arrival.append(
                (self.pi_doa_layers["mode_" + "{}".format(k)](input_reshaped)).reshape(
                    batch, T, -1
                )
            )  # Size [batchxTx1]

        mu_stacked = torch.stack(
            mu_directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_modesxoutput_dim]
        log_kappa_stacked = torch.stack(
            log_kappa_directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_modesx1]
        pi_stacked = torch.stack(
            pi_directions_of_arrival, dim=-2
        )  # Shape [batchxTxself.num_modesx1]
        pi_stacked = torch.nn.Softmax(dim=-2)(
            pi_stacked
        )  # Shape [batchxTxself.num_modesx1]

        return mu_stacked, log_kappa_stacked, pi_stacked


class GridLocalizationOutput(nn.Module):
    """Implements a module that outputs source activity and direction-of-arrival for sound event localization. An input
    of fixed dimension is passed through a fully-connected layer and then split into a source activity vector with
    sigmoid output activations and corresponding azimuth and elevation vectors, which are subsequently combined to a
    direction-of-arrival output tensor.
    Args:
        input_dim: Input dimension at each time step.
        num_hypothesis: Number of hypothesis in the model.
    """

    def __init__(
        self, input_dim: int, output_dim: int = 2, nrows: int = 4, ncols: int = 4
    ):
        super(GridLocalizationOutput, self).__init__()

        self.source_activity_output_layers = {}
        self.azimuth_output_layers = {}
        self.elevation_output_layers = {}
        self.output_dim = output_dim
        self.nrows = nrows
        self.ncols = ncols

        self.doa_layers = nn.ModuleDict()

        for row in range(self.nrows):
            for col in range(self.ncols):
                self.doa_layers["row_" + "{}_col_{}".format(row, col)] = nn.Linear(
                    in_features=input_dim, out_features=1, device="cuda:0"
                )
