from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

import os
import logging
import sys

# File copied from https://github.com/ashleve/lightning-hydra-template/blob/main/src/eval.py.

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:

    # create a console output log file in the output directory newly created

    # # set the log level
    # log.setLevel("DEBUG")

    # # create a file handler with the specified log file path
    # file_handler = logging.FileHandler(filename=os.path.join(str(cfg.paths.output_dir),'eval.log'))

    # # set the log level for the file handler
    # file_handler.setLevel(logging.DEBUG)

    # # create a formatter and add it to the file handler
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)

    # # add the file handler to the log
    # log.addHandler(file_handler)

    # # same for the terminal output
    sys.stdout = open(
        os.path.join(str(cfg.paths.output_dir), "terminal_output.txt"), "w"
    )
    sys.stderr = sys.stdout

    evaluate(cfg)

    # copy the log file to the output dir
    if os.path.exists(
        os.path.join(str(cfg.paths.work_dir), str(cfg.logger.mlflow.run_name) + ".log")
    ):
        os.system(
            f"sudo mv {os.path.join(str(cfg.paths.work_dir), str(cfg.logger.mlflow.run_name)+'.log')} {str(cfg.paths.output_dir)}"
        )


if __name__ == "__main__":
    main()
