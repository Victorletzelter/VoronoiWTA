import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import re
import torch.nn.functional as F

log = pylogger.get_pylogger(__name__)


def wrap_to_spherical_coordinates(azimuth, elevation):
    """
    Wraps the values of the input azimuth and elevation modulo 2pi and pi, respectively, so that they lie between -pi and pi and -pi/2 and pi/2, respectively.
    """

    # assuming azimuth and elevation are in radians, and the tensors are of order 1.
    # n = np.azimuth.shape[0]
    # assert azimuth.shape == elevation.shape == (n,)
    azimuth_wrapped = np.empty_like(azimuth)
    elevation_wrapped = np.empty_like(elevation)

    for i in range(len(azimuth)):
        azimuth_wrapped[i] = wrap_to_pi(azimuth[i])
        elevation_wrapped[i] = wrap_to_pi(elevation[i])

        if elevation_wrapped[i] > np.pi / 2:
            elevation_wrapped[i] = np.pi - elevation_wrapped[i]
            azimuth_wrapped[i] = wrap_to_pi(azimuth_wrapped[i] + np.pi)

        elif elevation_wrapped[i] < -np.pi / 2:
            elevation_wrapped[i] = -np.pi - elevation_wrapped[i]
            azimuth_wrapped[i] = wrap_to_pi(azimuth_wrapped[i] + np.pi)

    return azimuth_wrapped, elevation_wrapped


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def compute_spherical_distance(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        assert RuntimeError("Input tensors require a dimension of two.")

    sine_term = torch.sin(y_pred[:, 1]) * torch.sin(y_true[:, 1])
    cosine_term = (
        torch.cos(y_pred[:, 1])
        * torch.cos(y_true[:, 1])
        * torch.cos(y_true[:, 0] - y_pred[:, 0])
    )

    return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))


### Useful functions


def sample_hyps_gaussian_mixture(mus, sigmas, pis, N_samples=20):
    # mus of shape [num_modes, output_dim]
    # sigmas of shape [num_modes, 1]
    # pis of shape [num_modes, 1]

    # Sample the modes
    modes = torch.multinomial(pis, N_samples, replacement=True)  # shape [N_samples]

    return torch.randn_like(mus[modes]) * sigmas[modes] + mus[modes]


def wrap_to_pi(arr: np.ndarray) -> np.ndarray:
    """
    Wraps the values of the input array modulo 2pi, so that they lie between -pi and pi.

    Args:
        arr (np.ndarray): Input array with values to be wrapped.

    Returns:
        np.ndarray: Wrapped array with values between -pi and pi.
    """
    wrapped_arr = np.mod(arr + np.pi, 2 * np.pi) - np.pi
    return wrapped_arr


def wrap_to_pi_2(arr: np.ndarray) -> np.ndarray:
    """
    Wraps the values of the input array modulo pi, so that they lie between -pi/2 and pi/2.

    Args:
        arr (np.ndarray): Input array with values to be wrapped.

    Returns:
        np.ndarray: Wrapped array with values between -pi/2 and pi/2.
    """
    wrapped_arr = np.mod(arr + np.pi / 2, np.pi) - np.pi / 2
    return wrapped_arr


def plot_prediction(
    key_scoring,
    mh_gt_source_activity_classes,
    mh_source_activity,
    mh_predictions,
    num_hypothesis,
    sample_idx_values,
    T_values,
    mh_confidences,
    mh_gt_doa,
    normalize=None,
    plot_mh=True,
    compute_emd=False,
    compute_oracle=False,
    sigma_points_mode=True,
    N_samples_mog=10,
    sigma_classes=None,
):

    if sigma_classes is None:
        # self.log.warning('No sigma classes provided. We use the default one.')
        classes = [
            "speech",
            "phone",
            "keyboard",
            "doorslam",
            "laughter",
            "keysDrop",
            "pageturn",
            "drawer",
            "cough",
            "clearthroat",
            "knock",
        ]
        print("No sigma classes provided. We use the default one.")
        sigma_classes = [
            5 + 5 * i for i in range(len(classes))
        ]  # [5,10,15,20,25,30,35,40,45,50,55]

    if normalize == "classical":
        mh_confidences_considered = mh_confidences[key_scoring] / mh_confidences[
            key_scoring
        ].sum(
            axis=2, keepdims=True
        )  # Shape [batchxTxself.num_hypothesisx1]
    elif normalize == "softmax":
        mh_confidences_considered = np.exp(mh_confidences[key_scoring]) / np.sum(
            np.exp(mh_confidences[key_scoring]), axis=2, keepdims=True
        )
    else:
        mh_confidences_considered = mh_confidences[key_scoring].copy()

    mh_predictions_considered = mh_predictions[key_scoring].copy()

    cmap = plt.get_cmap("Blues")
    Max_sources = 3
    # Calculate the grid dimensions
    grid_rows = len(sample_idx_values)
    grid_cols = len(T_values)
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(int(5.5 * grid_cols), int(5.5 * grid_rows)),
        sharex=True,
        sharey=True,
    )
    # Set the x and y limits for all plots
    x_min, x_max, y_min, y_max = -np.pi, np.pi, -np.pi / 2, np.pi / 2

    classes = [
        "speech",
        "phone",
        "keyboard",
        "doorslam",
        "laughter",
        "keysDrop",
        "pageturn",
        "drawer",
        "cough",
        "clearthroat",
        "knock",
    ]
    sigma_classes = [
        5 + 5 * i for i in range(len(classes))
    ]  # [5,10,15,20,25,30,35,40,45,50,55]

    for r, sample_idx in enumerate(sample_idx_values):
        for c, T in enumerate(T_values):
            List_active_sources_idx = []
            for i in range(Max_sources):
                if mh_source_activity[key_scoring][sample_idx, T, i] == 1:
                    List_active_sources_idx.append(i)
            if len(sample_idx_values) == 1 and len(T_values) == 1:
                ax = axes
            elif len(sample_idx_values) == 1:
                ax = axes[c]
            elif len(T_values) == 1:
                ax = axes[r]
            else:
                ax = axes[r, c]

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(
                axis="x", labelsize=25
            )  # Set font size for x-axis tick label
            ax.tick_params(
                axis="y", labelsize=25
            )  # Set font size for x-axis tick labels

            if plot_mh == True:

                for i in List_active_sources_idx:
                    active_class_idx = (
                        mh_gt_source_activity_classes[key_scoring][sample_idx, T, i, :]
                        > 0
                    )  # shape [num_classes]
                    sigma = sigma_classes[
                        list(active_class_idx).index(1.0)
                    ]  # shape [num_active]
                    x = mh_gt_doa[key_scoring][sample_idx, T, i, 0]
                    y = mh_gt_doa[key_scoring][sample_idx, T, i, 1]
                    sigma_x = sigma
                    sigma_y = sigma
                    sigma_x = np.deg2rad(sigma_x)
                    sigma_y = np.deg2rad(sigma_y)
                    # ellipse = Ellipse((x, y), width=2*sigma_x, height=2*sigma_y, edgecolor='green', facecolor='none')
                    ##
                    # Generate points for the ellipse
                    theta = np.linspace(
                        0, 2 * np.pi, 100
                    )  # 100 points around the ellipse
                    ellipse_x = x + sigma_x * np.cos(theta)
                    ellipse_y = y + sigma_y * np.sin(theta)

                    # Wrap the points
                    wrapped_x, wrapped_y = wrap_to_spherical_coordinates(
                        ellipse_x, ellipse_y
                    )

                    # Plotting
                    ax.scatter(
                        wrapped_x, wrapped_y, s=2, c="green"
                    )  # Plot wrapped ellipse points
                    ##
                    # ax.add_patch(ellipse)

                ax.scatter(
                    [
                        mh_gt_doa[key_scoring][sample_idx, T, i, 0]
                        for i in List_active_sources_idx
                    ],
                    [
                        mh_gt_doa[key_scoring][sample_idx, T, i, 1]
                        for i in List_active_sources_idx
                    ],
                    color="green",
                    marker="*",
                    edgecolors="black",
                    linewidth=0.5,
                    s=300,
                )

            # ###Multi-hypothesis prediction
            colors = [
                cmap(conf) for conf in mh_confidences_considered[sample_idx, T, :, 0]
            ]

            ### WRAP to -pi, pi
            if plot_mh == True:

                mh_predictions_considered[sample_idx, T, :, 0] = wrap_to_pi(
                    mh_predictions_considered[sample_idx, T, :, 0]
                )
                mh_predictions_considered[sample_idx, T, :, 1] = wrap_to_pi_2(
                    mh_predictions_considered[sample_idx, T, :, 1]
                )

                ax.scatter(
                    [
                        mh_predictions_considered[sample_idx, T, k, 0]
                        for k in range(num_hypothesis)
                    ],
                    [
                        mh_predictions_considered[sample_idx, T, k, 1]
                        for k in range(num_hypothesis)
                    ],
                    c=colors,
                    cmap=cmap,
                    edgecolors="black",
                    linewidth=0.5,
                    s=100,
                )

            ax.set_title(f"t={int((T*0.5/25)*10**3)} ms", fontsize=30)

    cbar_ax = fig.add_axes(
        [0.92, 0.1, 0.02, 0.8]
    )  # Adjust the position and size as needed
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=np.min(
                [
                    mh_confidences_considered[sample, t, :, 0]
                    for sample in sample_idx_values
                    for t in T_values
                ]
            ),
            vmax=np.max(
                [
                    mh_confidences_considered[sample, t, :, 0]
                    for sample in sample_idx_values
                    for t in T_values
                ]
            ),
        ),
    )
    sm.set_array(mh_confidences[key_scoring])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(labelsize=15)
    # plt.show()
    # plt.close()

    return fig


def plot_prediction_step(
    step_value,
    key_scoring,
    mh_gt_source_activity_classes,
    mh_source_activity,
    mh_predictions,
    num_hypothesis,
    sample_idx_values,
    T_values,
    mh_confidences,
    mh_gt_doa,
    normalize=None,
    plot_mh=True,
    compute_emd=False,
    compute_oracle=False,
    sigma_points_mode=True,
    N_samples_mog=10,
    sigma_classes=None,
):

    if sigma_classes is None:
        # self.log.warning('No sigma classes provided. We use the default one.')
        print("No sigma classes provided. We use the default one.")
        sigma_classes = [
            5 + 5 * i for i in range(len(classes))
        ]  # [5,10,15,20,25,30,35,40,45,50,55]

    if normalize == "classical":
        mh_confidences_considered = mh_confidences[key_scoring] / mh_confidences[
            key_scoring
        ].sum(
            axis=2, keepdims=True
        )  # Shape [batchxTxself.num_hypothesisx1]
    elif normalize == "softmax":
        mh_confidences_considered = np.exp(mh_confidences[key_scoring]) / np.sum(
            np.exp(mh_confidences[key_scoring]), axis=2, keepdims=True
        )
    else:
        mh_confidences_considered = mh_confidences[key_scoring].copy()

    mh_predictions_considered = mh_predictions[key_scoring].copy()

    cmap = plt.get_cmap("Blues")
    Max_sources = 3
    # Calculate the grid dimensions
    grid_rows = len(sample_idx_values)
    grid_cols = len(T_values) + 1
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(int(5.5 * (grid_cols)), int(5.5 * grid_rows)),
        sharex=True,
        sharey=True,
    )
    # Set the x and y limits for all plots
    x_min, x_max, y_min, y_max = -np.pi, np.pi, -np.pi / 2, np.pi / 2

    classes = [
        "speech",
        "phone",
        "keyboard",
        "doorslam",
        "laughter",
        "keysDrop",
        "pageturn",
        "drawer",
        "cough",
        "clearthroat",
        "knock",
    ]
    sigma_classes = [
        5 + 5 * i for i in range(len(classes))
    ]  # [5,10,15,20,25,30,35,40,45,50,55]

    # Iterate through all subplots
    for i in range(grid_rows):
        for j in range(grid_cols):
            ax = axes[i, j]

            # Set the same x and y limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Check if it's the first column
            if j == 0:
                # Disable the grid for this subplot
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            # else:
            #     # Enable grid for other subplots if desired
            #     ax.grid(True)

    # Add text for the entire first column
    # Adjust the x and y coordinates as needed to position the text
    fig.text(
        0.05,
        0.5,
        "Step = {}".format(step_value),
        va="center",
        rotation="horizontal",
        fontsize=50,
    )

    for r, sample_idx in enumerate(sample_idx_values):
        for c, T in enumerate(T_values):
            c += 1
            List_active_sources_idx = []
            for i in range(Max_sources):
                if mh_source_activity[key_scoring][sample_idx, T, i] == 1:
                    List_active_sources_idx.append(i)
            if len(sample_idx_values) == 1 and len(T_values) == 1:
                ax = axes
            # elif len(sample_idx_values)==1 :
            #     ax = axes[c]
            # elif len(T_values)==1  :
            #     ax = axes[r]
            else:
                ax = axes[r, c]

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(
                axis="x", labelsize=25
            )  # Set font size for x-axis tick label
            ax.tick_params(
                axis="y", labelsize=25
            )  # Set font size for x-axis tick labels

            x_ticks = np.linspace(-2 * np.pi / 3, 2 * np.pi / 3, 3)
            x_ticklabels = [f"{tick:.0f}°" for tick in np.degrees(x_ticks)]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)

            # Modify the y-axis ticks to display degrees
            y_ticks = np.linspace(-np.pi / 3, np.pi / 3, 3)
            y_ticklabels = [f"{tick:.1f}°" for tick in np.degrees(y_ticks)]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels)

            if plot_mh == True:

                for i in List_active_sources_idx:
                    active_class_idx = (
                        mh_gt_source_activity_classes[key_scoring][sample_idx, T, i, :]
                        > 0
                    )  # shape [num_classes]
                    sigma = sigma_classes[
                        list(active_class_idx).index(1.0)
                    ]  # shape [num_active]
                    x = mh_gt_doa[key_scoring][sample_idx, T, i, 0]
                    y = mh_gt_doa[key_scoring][sample_idx, T, i, 1]
                    sigma_x = np.deg2rad(sigma)
                    sigma_y = np.deg2rad(sigma)

                    # Generate points for the ellipse
                    theta = np.linspace(
                        0, 2 * np.pi, 100
                    )  # 100 points around the ellipse
                    ellipse_x = x + sigma_x * np.cos(theta)
                    ellipse_y = y + sigma_y * np.sin(theta)

                    # Wrap the points
                    wrapped_x, wrapped_y = wrap_to_spherical_coordinates(
                        ellipse_x, ellipse_y
                    )

                    # Plotting
                    ax.scatter(
                        wrapped_x, wrapped_y, s=2, c="green"
                    )  # Plot wrapped ellipse points
                    #################################
                    # ax.add_patch(ellipse)

                ax.scatter(
                    [
                        mh_gt_doa[key_scoring][sample_idx, T, i, 0]
                        for i in List_active_sources_idx
                    ],
                    [
                        mh_gt_doa[key_scoring][sample_idx, T, i, 1]
                        for i in List_active_sources_idx
                    ],
                    color="green",
                    marker="*",
                    edgecolors="black",
                    linewidth=0.5,
                    s=300,
                )

            # ###Multi-hypothesis prediction
            colors = [
                cmap(conf) for conf in mh_confidences_considered[sample_idx, T, :, 0]
            ]

            ### WRAP to -pi, pi
            if plot_mh == True:

                (
                    mh_predictions_considered[sample_idx, T, :, 0],
                    mh_predictions_considered[sample_idx, T, :, 1],
                ) = wrap_to_spherical_coordinates(
                    mh_predictions_considered[sample_idx, T, :, 0],
                    mh_predictions_considered[sample_idx, T, :, 1],
                )

                ax.scatter(
                    [
                        mh_predictions_considered[sample_idx, T, k, 0]
                        for k in range(num_hypothesis)
                    ],
                    [
                        mh_predictions_considered[sample_idx, T, k, 1]
                        for k in range(num_hypothesis)
                    ],
                    c=colors,
                    cmap=cmap,
                    edgecolors="black",
                    linewidth=0.5,
                    s=100,
                )

            ax.set_title(f"t={int((T*0.5/25)*10**3)} ms", fontsize=30)

    for r, sample_idx in enumerate(sample_idx_values):
        ax = axes[r, 0]
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])

    cbar_ax = fig.add_axes(
        [0.92, 0.1, 0.02, 0.8]
    )  # Adjust the position and size as needed
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=np.min(
                [
                    mh_confidences_considered[sample, t, :, 0]
                    for sample in sample_idx_values
                    for t in T_values
                ]
            ),
            vmax=np.max(
                [
                    mh_confidences_considered[sample, t, :, 0]
                    for sample in sample_idx_values
                    for t in T_values
                ]
            ),
        ),
    )
    sm.set_array(mh_confidences[key_scoring])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(labelsize=30)
    cbar.ax.yaxis.set_major_locator(
        plt.MaxNLocator(nbins=10)
    )  # Reducing the number of ticks
    cbar.solids.set_rasterized(True)

    return fig


def get_step_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


def sort_steps(file_list):
    # Function to extract the step number from the filename
    # Sorting the list based on the step number
    sorted_list = sorted(file_list, key=get_step_number)
    return sorted_list


def sigmoid(T):
    return 1 / (1 + np.exp(-T))


def logsinh_np(x):
    """Compute log(sinh(x)), stably for large x.

    Parameters
    ----------
    x : float or numpy.array
        argument to evaluate at, must be positive

    Returns
    -------
    float or numpy.array
        log(sinh(x))
    """
    if np.any(x < 0):
        raise ValueError("logsinh only valid for positive arguments")
    return x + np.log(1 - np.exp(-2 * x)) - np.log(2)


def logsinh_torch(x):
    """Compute log(sinh(x)), stably for large x using PyTorch.

    Parameters
    ----------
    x : torch.Tensor
        argument to evaluate at, must be positive

    Returns
    -------
    torch.Tensor
        log(sinh(x))
    """
    if torch.any(x < 0):
        raise ValueError("logsinh only valid for positive arguments")
    return x + torch.log(1 - torch.exp(-2 * x)) - torch.log(torch.tensor(2.0))


def compute_angular_distance(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """
    if np.ndim(x) != 1:
        raise ValueError("First DoA must be a single value.")

    return np.arccos(
        np.sin(x[1]) * np.sin(y[1]) + np.cos(x[1]) * np.cos(y[1]) * np.cos(y[0] - x[0])
    )


def compute_angular_distance_vec(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """

    return np.arccos(
        np.sin(x[:, 1]) * np.sin(y[:, 1])
        + np.cos(x[:, 1]) * np.cos(y[:, 1]) * np.cos(y[:, 0] - x[:, 0])
    )


def compute_angular_distance_torch(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """

    return torch.acos(
        torch.sin(x[:, 1]) * torch.sin(y[:, 1])
        + torch.cos(x[:, 1]) * torch.cos(y[:, 1]) * torch.cos(y[:, 0] - x[:, 0])
    )


def compute_spherical_distance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (Tensor): Tensor of predicted azimuth and elevation angles.
        y_true (Tensor): Tensor of ground-truth azimuth and elevation angles.

    Returns:
        Tensor: Tensor of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        assert RuntimeError("Input tensors require a dimension of two.")

    sine_term = torch.sin(y_pred[:, 1]) * torch.sin(y_true[:, 1])
    cosine_term = (
        torch.cos(y_pred[:, 1])
        * torch.cos(y_true[:, 1])
        * torch.cos(y_true[:, 0] - y_pred[:, 0])
    )

    return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))
