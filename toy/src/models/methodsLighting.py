from typing import Any, Dict, Tuple
from lightning import LightningModule
import torch.optim as optim
import torch
import numpy as np
from src.metrics import EMD_module, Oracle_module
from src.utils.losses import mhloss, mhconfloss

from src.utils.eval_utils import generate_plot_adapted, gss
import os

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class methodsLighting(LightningModule):
    """A `LightningModule`, which implements 8 key methods:

    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.

    """

    def __init__(
        self,
        hparams,
    ) -> None:
        """Initialize a `Lightning module`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self._hparams = hparams

        self.emd_metric_instance = EMD_module(
            distance=self.hparams["dist_type_eval"],
        )

        self.oracle_instance = Oracle_module(
            distance=self.hparams["dist_type_eval"],
            computation_type="dirac",
        )

        if self.hparams["compute_mse"] is True:
            self.mse_accumulator = 0
            self.n_samples_mse = 0
            self.rmse_accumulator = 0

        if self.hparams["compute_nll"] is True:
            self.nll_accumulator = 0
            self.n_samples_nll = 0

    def loss(self) -> torch.Tensor:
        """Compute the loss function.

        :param outputs: The model's predictions.
        :param targets: The target labels.
        :return: The loss value.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = getattr(optim, self._hparams["optimizer"])(
            self.trainer.model.parameters(), lr=self._hparams["learning_rate"]
        )

        if "scheduler" in self.hparams and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer}

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, data_target_position, data_source_activity_target = batch
        targets = (data_target_position, data_source_activity_target)

        # Forward pass
        outputs = self(x.float())  # .reshape(-1, self.input_dim).float())

        # Compute the loss
        loss = self.loss()(predictions=outputs, targets=targets)

        return loss, outputs, targets

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        output = {"train_loss": loss}
        self.log_dict(output)

        return loss

    def on_train_epoch_start(self) -> None:
        "Lightning hook that is called when a training epoch starts."

        if (
            "plot_mode_training" in self.hparams
            and "plot_training_frequency" in self.hparams
        ):
            if (
                self.current_epoch % self.hparams["plot_training_frequency"] == 0
                and self.hparams["plot_mode"] is True
                and (
                    self.hparams["name"] == "rmcl"
                    or "histogram" in self.hparams["name"]
                )
            ):
                # we check if the folder with plots exists
                if not os.path.exists(self.trainer.default_root_dir + "/plots"):
                    os.makedirs(os.path.join(self.trainer.default_root_dir, "plots"))

                plot_title = self.hparams["name"] + "_epoch_" + str(self.current_epoch)

                generate_plot_adapted(
                    self,
                    dataset_ms=self.trainer.datamodule.dataset_ms_class,
                    dataset_ss=self.trainer.datamodule.dataset_ss_class,
                    path_plot=os.path.join(self.trainer.default_root_dir, "plots"),
                    model_type="rMCL",
                    list_x_values=[0.1, 0.6, 0.9],
                    n_samples_gt_dist=3000,
                    num_hypothesis=self.num_hypothesis,
                    save_mode=True,
                    device="cpu",
                    plot_title=plot_title,
                    plot_voronoi=False,
                    plot_title_bool=True,
                )

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        output = {"val_loss": loss}
        self.log_dict(output)

        if (
            "compute_risk_val" in self.hparams
            and self.hparams["compute_risk_val"] is True
        ):
            risk = self.compute_risk(predictions=preds, targets=targets)
            self.log_dict({"val_risk": risk})

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # The loss should be reset to the original one in training_epoch_start in the case of awta.
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        if (
            "denormalize_predictions" in self.hparams
            and self.hparams["denormalize_predictions"] is True
        ):

            self.mean_train = self.trainer.datamodule.uci_dataset_train.scaler_y.mean_[
                0
            ]
            self.std_train = np.sqrt(
                self.trainer.datamodule.uci_dataset_train.scaler_y.var_[0]
            )

            ### Adjust the square size accordingly if needed for nll computation, here, it is large on purpose.
            self.square_size = np.abs(self.mean_train) + 8 * self.std_train

            preds = self.denormalize_predictions(
                preds, mean_scaler=self.mean_train, std_scaler=self.std_train
            )
            targets = self.denormalize_targets(
                targets, mean_scaler=self.mean_train, std_scaler=self.std_train
            )
            # The test set was normalized using the train set statistics. We denormalize it.

        # update and log metrics
        predictions_oracle = self.prepare_predictions_oracle(predictions=preds)

        oracle = self.oracle_instance.compute(
            predictions=predictions_oracle,
            targets=targets,
        )

        if self.hparams["compute_risk"] is True:
            risk = self.compute_risk(predictions=predictions_oracle, targets=targets)
        else:
            risk = torch.tensor(float("nan"))

        if self.hparams["compute_mse"] is True:
            predictions_mse = self.prepare_predictions_mse(predictions=preds)
            mse, rmse = self.compute_mse(predictions=predictions_mse, targets=targets)
            self.mse_accumulator += mse * batch[0].shape[0]
            self.rmse_accumulator += (rmse) * batch[0].shape[0]
            self.n_samples_mse += batch[0].shape[0]
        else:
            mse = torch.tensor(float("nan"))

        if self.hparams["compute_emd"] is True:
            predictions_emd = self.prepare_predictions_emd(predictions=preds)
            targets_emd = self.trainer.datamodule.prepare_targets_emd(
                inputs=batch[0],
                targets=targets,
                n_gt_samples_per_frame=self.hparams["n_gt_samples_per_frame"],
            )

            emd = self.emd_metric_instance.compute(
                predictions=predictions_emd, targets=targets_emd
            )
        else:
            emd = torch.tensor(float("nan"))

        if (self.output_dim == 2 or self.output_dim == 1) and self.hparams[
            "compute_nll"
        ] is True:

            predictions_nll = self.prepare_predictions_nll(predictions=preds)

            if (
                "check_nll_normalization" in self.hparams
                and self.output_dim == 1
                and self.hparams["check_nll_normalization"] is True
            ):
                integral_predicted_density = self.check_nll_normalization_1d(
                    predictions=predictions_nll
                )

            nll = self.batch_compute_negative_log_likelihood(
                predictions=predictions_nll, targets=targets
            )

            self.nll_accumulator += nll * batch[0].shape[0]
            self.n_samples_nll += batch[0].shape[0]

        else:
            # raise a warning and set the nll to nan
            # print('The Voronoi WTA is only implemented in the 2D and 1D case')
            nll = torch.tensor(float("nan"))

        output = {
            "test_loss": loss,
            "test_emd": emd,
            "test_oracle": oracle,
            "test_risk": risk,
            "test_nll": nll,
            "test_mse": mse,
        }

        self.log_dict(output)

    def compute_mse(self, predictions, targets):
        # Compute the emd
        # Assumes predictions of shape [batch, output_dim]
        # targets[0]: data_target_position of shape [batch, Max_sources, output_dim]
        # targets[1]: data_source_activity_target of shape [batch, Max_sources, 1]
        # Assumes one target is active here.
        targets = targets[0][:, 0, :]  # shape [batch, output_dim]
        square_diff = (predictions - targets) ** 2
        return torch.mean(square_diff.sum(dim=-1)), torch.mean(
            square_diff.sum(dim=-1) ** 0.5
        )

    def compute_risk(self, predictions, targets):

        predictions = (predictions[0], None)
        # Compute the risk
        return mhloss(
            distance="euclidean-squared",
            output_dim=self.output_dim,
        )(predictions=predictions, targets=targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        if self.hparams["compute_mse"] is True:
            # Compute the mean MSE over all test batches
            total_mse = self.mse_accumulator / self.n_samples_mse
            rmse = total_mse**0.5

            self.log_dict({"test_accumulated_rmse": rmse})

            instance_based_rmse = self.rmse_accumulator / self.n_samples_mse

            self.log_dict({"test_instance_based_rmse": instance_based_rmse})

        if self.hparams["compute_nll"] is True and self.hparams["compute_mse"] is True:
            # Compute the mean NLL over all test batches
            total_nll = self.nll_accumulator / self.n_samples_nll
            self.log_dict({"test_accumulated_nll": total_nll})

        if "plot_mode" in self.hparams:
            plot_mode = self.hparams["plot_mode"]
        else:
            plot_mode = True
        if "dataset_name" in self.trainer.datamodule._hparams:
            plot_mode = False

        if plot_mode is True and (
            self.hparams["name"] == "rmcl" or "histogram" in self.hparams["name"]
        ):
            generate_plot_adapted(
                self,
                dataset_ms=self.trainer.datamodule.dataset_ms_class,
                dataset_ss=self.trainer.datamodule.dataset_ss_class,
                path_plot=self.trainer.default_root_dir,
                model_type="rMCL",
                list_x_values=[0.01, 0.6, 0.9],
                n_samples_gt_dist=3000,
                num_hypothesis=self.num_hypothesis,
                save_mode=True,
                device="cpu",
                plot_title=self.hparams["name"] + "_preds",
            )

        if plot_mode is True and self.hparams["name"] == "gauss_mix":
            generate_plot_adapted(
                self,
                dataset_ms=self.trainer.datamodule.dataset_ms_class,
                dataset_ss=self.trainer.datamodule.dataset_ss_class,
                path_plot=self.trainer.default_root_dir,
                model_type="MDN",
                list_x_values=[0.01, 0.6, 0.9],
                n_samples_gt_dist=3000,
                num_hypothesis=self.num_hypothesis,
                log_var_pred=self.log_var_pred,
                save_mode=True,
                device="cpu",
                plot_title=self.hparams["name"] + "_preds",
            )

    def on_fit_end(self):

        if (
            self.hparams.name != "gauss_mix"
            and "h_optimization" in self.hparams
            and self.hparams["h_optimization"] is True
        ):

            # Disable the logger temporarily
            original_loggers = self.trainer.logger
            original_compute_emd = self.hparams.compute_emd
            original_compute_risk = self.hparams.compute_risk
            original_compute_mse = self.hparams.compute_mse

            if "batch_size_h_opt" in self.hparams:
                original_batch_size = self.trainer.datamodule.hparams["batch_size"]
                self.trainer.datamodule.hparams["batch_size"] = (
                    self.hparams.batch_size_h_opt
                )

            if "limit_val_batches_h_opt" in self.hparams:
                original_limit_test_batches = self.trainer.limit_test_batches
                self.trainer.limit_test_batches = self.hparams.limit_val_batches_h_opt

            self.trainer.logger = None
            self.hparams.compute_emd = False
            self.hparams.compute_risk = False
            self.hparams.compute_mse = False

            def f(h):
                self.scaling_factor = h
                dic_metrics = self.trainer.test(
                    model=self,
                    dataloaders=self.trainer.datamodule.val_dataloader(),
                    verbose=False,
                )
                return dic_metrics[0]["test_nll"]

            if "h_min" in self.hparams:
                h_min = self.hparams.h_min
            else:
                h_min = 0.1

            if (
                "denormalize_predictions" in self.hparams
                and self.hparams["denormalize_predictions"] is True
            ):
                h_max = np.sqrt(
                    self.trainer.datamodule.uci_dataset_train.scaler_y.var_[0]
                )
            elif "h_max" in self.hparams:
                h_max = self.hparams.h_max
            else:
                h_max = 2

            h_tol = 1e-1

            if "h_tol" in self.hparams:
                h_tol = self.hparams.h_tol
                log.info("Using h_tol {} from hparams".format(h_tol))
            else:
                h_tol = 1e-1

            h_opt, f_hopt = gss(f, h_min, h_max, tol=h_tol)

            self.scaling_factor = h_opt
            log.info(
                "Optimal scaling factor found: {}, with val nll {} and h_max {}".format(
                    np.round(self.scaling_factor, 2),
                    np.round(f_hopt, 2),
                    np.round(h_max, 2),
                )
            )

            # Restore the original attributes
            self.trainer.logger = original_loggers
            self.hparams.compute_emd = original_compute_emd
            self.hparams.compute_risk = original_compute_risk
            self.hparams.compute_mse = original_compute_mse
            self.mse_accumulator = 0
            self.rmse_accumulator = 0
            self.n_samples_mse = 0
            self.nll_accumulator = 0
            self.n_samples_nll = 0

            if "limit_val_batches_h_opt" in self.hparams:
                self.trainer.limit_test_batches = original_limit_test_batches

            if "batch_size_h_opt" in self.hparams:
                self.trainer.datamodule.hparams["batch_size"] = original_batch_size
