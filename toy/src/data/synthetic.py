from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.dataset import *


class SyntheticData(LightningDataModule):
    """
    Generator of datasets
    """

    def __init__(self, batch_size, num_workers, hparams) -> None:
        """

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.hparams["batch_size"] = batch_size
        self.hparams["num_workers"] = num_workers

        for key in hparams:
            self.hparams[key] = hparams[key]

        if self.hparams["name"] == "rotating-two-moons":

            self.dataset_ss_class = rotating_two_moons
            self.dataset_ms_class = MultiSources_rotating_two_moons

        elif self.hparams["name"] == "changing-damier":

            self.dataset_ss_class = changing_damier
            self.dataset_ms_class = MultiSources_changing_damier

        elif self.hparams["name"] == "single_gaussian-not-centered":

            self.dataset_ss_class = single_gauss_not_centered
            self.dataset_ms_class = MultiSources_single_gauss_not_centered

        elif self.hparams["name"] == "mixture-uni-to-gaussians-v2":

            self.dataset_ss_class = mixture_uni_to_gaussians_v2
            self.dataset_ms_class = MultiSources_mixture_uni_to_gaussians_v2

        else:
            raise ValueError("The dataset name is not correct")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        train_dataset = self.dataset_ms_class(n_samples=self.hparams["n_samples_train"])

        # Create data loader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
        )

        return train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.

        """
        val_dataset = self.dataset_ms_class(n_samples=self.hparams["n_samples_val"])

        # Create the data loader
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )

        return val_loader

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        test_dataset = self.dataset_ms_class(n_samples=self.hparams["n_samples_val"])

        # Create the data loader
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )

        return test_loader

    def prepare_targets_emd(self, inputs, targets, n_gt_samples_per_frame=1000):
        # inputs of shape [batch,Max_sources,input_dim]
        # target_position, mask_activity = (
        #     targets[0],
        #     targets[1],
        # )  # Shape [batch,Max_sources,output_dim],[batch,Max_sources,1]

        # targets of shape [batch,Max_sources*n_gt_samples_per_frame,output_dim]

        list_samples_gen = []
        list_mask_activity_gen = []
        output_dim = targets[0].shape[-1]
        # convert to cpu/numpy
        # inputs = inputs.cpu().numpy()

        assert inputs.shape[-1] == 1, "Only input dim of size 1 is managed here"

        test_dataset = self.dataset_ms_class(n_samples=n_gt_samples_per_frame)

        for elt_in_batch in range(targets[0].shape[0]):
            samples_gen, mask_activity_gen = test_dataset.generate_dataset_distribution(
                t=inputs[elt_in_batch, 0].cpu().numpy(),
                n_samples=n_gt_samples_per_frame,
            )

            samples_gen = samples_gen.reshape(1, -1, output_dim)
            mask_activity_gen = mask_activity_gen.reshape(1, -1, 1).astype(float)

            list_samples_gen.append(
                torch.tensor(samples_gen, device=inputs.device)
            )  # sample_gen of shape [1,Max*sources,n_gt_samples_per_frame,output_dim]
            list_mask_activity_gen.append(
                torch.tensor(mask_activity_gen, device=inputs.device)
            )

        # Concatenate the list of samples elements in the batch axis
        samples_gen = torch.cat(list_samples_gen, dim=0)
        mask_activity_gen = torch.cat(list_mask_activity_gen, dim=0)

        assert samples_gen.shape == (
            inputs.shape[0],
            n_gt_samples_per_frame * targets[0].shape[1],
            targets[0].shape[2],
        )
        assert mask_activity_gen.shape == (
            inputs.shape[0],
            n_gt_samples_per_frame * targets[1].shape[1],
            1,
        )

        return samples_gen, mask_activity_gen
