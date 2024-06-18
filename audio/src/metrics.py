import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from src.utils.utils import compute_spherical_distance
import sys
import h5py
import os
import cv2


class Metric:
    def __init__(self):
        pass

    def compute(self, predictions, targets):
        pass


class Oracle_module(Metric):
    def __init__(
        self,
        distance,
        activity_mode,
        rad2deg,
        class_mode,
        computation_type="dirac",
        print_hyp_idx=False,
        num_sources_per_sample_min=0,
        num_sources_per_sample_max=3,
    ):

        super(Oracle_module, self).__init__()

        self.distance = distance

        if self.distance == "spherical-squared":
            self.distance = "spherical"

        self.activity_mode = activity_mode
        self.rad2deg = rad2deg
        self.print_hyp_idx = print_hyp_idx
        self.class_mode = class_mode
        self.num_sources_per_sample_min = num_sources_per_sample_min
        self.num_sources_per_sample_max = num_sources_per_sample_max
        self.computation_type = computation_type

    def compute(self, predictions, targets, dataset_idx=0, batch_idx=0):

        if self.computation_type == "dirac":
            return self.oracle_doa_error(predictions, targets, dataset_idx, batch_idx)

        else:
            # Not implemented Error
            raise NotImplementedError

    def oracle_doa_error(self, predictions, targets, dataset_idx=0, batch_idx=0):
        """The oracle DOA error compute the minimum distance between the hypothesis predicted and the
        ground truth, for each ground truth.

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxTxself.num_hypothesisx2]
            targets (torch.Tensor,torch.Tensor): Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            distance (str, optional): Distance to use. Defaults to 'euclidean'.
            dataset_idx (int, optional): Dataset index. Defaults to 0.
            batch_idx (int, optional): Batch index. Defaults to 0.

        Return:
            oracle_doa_error (torch.tensor)
        """
        if self.activity_mode:
            hyps_DOAs_pred_stacked, act_pred_stacked = predictions
        else:
            hyps_DOAs_pred_stacked = predictions[0]

        source_activity_target = targets[0]
        direction_of_arrival_target = targets[1]

        # if self.class_mode :
        # direction_of_arrival_target = direction_of_arrival_target.sum(dim=-2)
        # source_activity_target = source_activity_target.sum(dim=-1)

        batch, T, num_hyps, _ = hyps_DOAs_pred_stacked.shape
        Max_sources = source_activity_target.shape[2]
        doa_error_matrix_new = np.zeros((T, 1))
        count_number_actives_predictions = 0  # This variable counts the number of predictions (for element in the batch and each time step) that are active

        for t in range(T):

            hyps_stacked_t = hyps_DOAs_pred_stacked[
                :, t, :, :
            ]  # Shape [batch,num_hyps,2]
            source_activity_target_t = source_activity_target[
                :, t, :
            ]  # Shape [batch,Max_sources]
            direction_of_arrival_target_t = direction_of_arrival_target[
                :, t, :, :
            ]  # Shape [batch,Max_sources,2]

            filling_value = 10000  # Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
            # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings.

            # 1st padding related to the inactive sources, not considered in the error calculation (with high error values)
            mask_inactive_sources = source_activity_target_t == 0
            mask_inactive_sources_target = mask_inactive_sources.unsqueeze(-1)

            if self.activity_mode:
                mask_inactives_source_predicted = (
                    act_pred_stacked[:, t, :, :] == 0
                )  # [batch,num_hyps,1], num_hyps = Max_sources in this case
                mask_inactive_sources = mask_inactive_sources_target
                hyps_stacked_t[
                    mask_inactives_source_predicted.expand_as(hyps_stacked_t)
                ] = filling_value  # We fill the inactive sources with a large value

            if mask_inactive_sources.dim() == 2:
                mask_inactive_sources = mask_inactive_sources.unsqueeze(-1)
            mask_inactive_sources = mask_inactive_sources.expand_as(
                direction_of_arrival_target_t
            )
            direction_of_arrival_target_t[mask_inactive_sources] = (
                filling_value  # Shape [batch,Max_sources,2]
            )

            # The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the
            # tensors gts[batch,i,num_hypothesis,2] contains duplicates of direction_of_arrival_target_t along the num_hypothesis
            # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,2] may contain inactive sources, and therefore
            # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
            gts = direction_of_arrival_target_t.unsqueeze(2).repeat(
                1, 1, num_hyps, 1
            )  # Shape [batch,Max_sources,num_hypothesis,2]

            # assert gts.shape==(batch,Max_sources,num_hyps,2)

            # We duplicate the hyps_stacked with a new dimension of shape Max_sources
            hyps_stacked_t_duplicated = hyps_stacked_t.unsqueeze(1).repeat(
                1, Max_sources, 1, 1
            )  # Shape [batch,Max_sources,num_hypothesis,2]

            # assert hyps_stacked_t_duplicated.shape==(batch,Max_sources,num_hyps,2)

            epsilon = 0.05
            eps = 0.001

            if self.distance == "euclidean":
                #### With euclidean distance
                diff = torch.square(
                    hyps_stacked_t_duplicated - gts
                )  # Shape [batch,Max_sources,num_hypothesis,2]
                channels_sum = torch.sum(
                    diff, dim=3
                )  # Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
                dist_matrix = torch.sqrt(
                    channels_sum + eps
                )  # Distance matrix [batch,Max_sources,num_hypothesis]

                wta_dist_matrix, idx_selected = torch.min(
                    dist_matrix, dim=2
                )  # wta_dist_matrix of shape [batch,Max_sources]
                mask = (
                    wta_dist_matrix <= filling_value / 2
                )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with
                wta_dist_matrix = (
                    wta_dist_matrix * mask
                )  # [batch,Max_sources], we select only the active sources.

            elif self.distance == "spherical":

                dist_matrix_euclidean = torch.sqrt(
                    torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
                )

                ### With spherical distance
                hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                    -1, 2
                )  # Shape [batch*num_hyps*Max_sources,2]
                gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
                diff = compute_spherical_distance(hyps_stacked_t_duplicated, gts)
                dist_matrix = diff.view(
                    batch, Max_sources, num_hyps
                )  # Shape [batch,Max_sources,num_hyps]

            wta_dist_matrix, idx_selected = torch.min(
                dist_matrix, dim=2
            )  # wta_dist_matrix of shape [batch,Max_sources]
            if self.distance == "spherical":
                eucl_wta_dist_matrix, _ = torch.min(
                    dist_matrix_euclidean, dim=2
                )  # wta_dist_matrix of shape [batch,Max_sources] for mask purpose
                mask = (
                    eucl_wta_dist_matrix <= filling_value / 2
                )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with
            else:
                mask = wta_dist_matrix <= filling_value / 2
            wta_dist_matrix = (
                wta_dist_matrix * mask
            )  # [batch,Max_sources], we select only the active sources.
            count_non_zeros = torch.sum(
                mask != 0
            )  # We count the number of actives sources for the computation of the mean (below).

            if self.print_hyp_idx:
                idx_selected[~mask] = (
                    -1
                )  # We look at the hypothesis selected for each active source only.
                output_folder = sys.stdout.name.split("terminal_output.txt")[-2]

                if os.path.exists(os.path.join(output_folder, "idx_selected")):
                    with h5py.File(
                        os.path.join(output_folder, "idx_selected"), "a"
                    ) as f:
                        f.create_dataset(
                            "idx_selected_dataset_{}_batch_{}_{}".format(
                                dataset_idx, batch_idx, t
                            ),
                            data=idx_selected.cpu().numpy(),
                        )

                else:
                    with h5py.File(
                        os.path.join(output_folder, "idx_selected"), "w"
                    ) as f:
                        f.create_dataset(
                            "idx_selected_dataset_{}_batch_{}_{}".format(
                                dataset_idx, batch_idx, t
                            ),
                            data=idx_selected.cpu().numpy(),
                        )

            num_sources_per_sample = torch.sum(
                source_activity_target_t.float(), dim=1, keepdim=True
            ).repeat(
                1, Max_sources
            )  # [batch,Max_sources]
            mask_where = torch.logical_and(
                num_sources_per_sample > self.num_sources_per_sample_min,
                num_sources_per_sample <= self.num_sources_per_sample_max,
            )
            count_number_actives_predictions += torch.sum(
                torch.sum(mask * mask_where != 0, dim=1) != 0
            )

            if count_non_zeros > 0:

                num_sources_per_sample = torch.sum(
                    source_activity_target_t.float(), dim=1, keepdim=True
                ).repeat(
                    1, Max_sources
                )  # [batch,Max_sources]
                # assert num_sources_per_sample.shape == (batch,Max_sources)
                # assert torch.sum(num_sources_per_sample)/Max_sources == torch.sum(source_activity_target_t)
                wta_dist_matrix = torch.where(
                    torch.logical_and(
                        num_sources_per_sample > self.num_sources_per_sample_min,
                        num_sources_per_sample <= self.num_sources_per_sample_max,
                    ),
                    wta_dist_matrix / num_sources_per_sample,
                    torch.zeros_like(num_sources_per_sample),
                )  # We divide by the number of active sources to get the mean error per sample
                oracle_err_new = torch.sum(
                    wta_dist_matrix
                )  # We compute the mean of the diff.
                if oracle_err_new == 0:
                    doa_error_matrix_new[t, 0] = np.nan
                elif self.rad2deg is True:
                    doa_error_matrix_new[t, 0] = np.rad2deg(
                        oracle_err_new.detach().cpu().numpy()
                    )
                else:
                    doa_error_matrix_new[t, 0] = oracle_err_new.detach().cpu().numpy()

            else:
                doa_error_matrix_new[t, 0] = np.nan

        return (
            torch.tensor(np.nansum(doa_error_matrix_new, dtype=np.float32))
            / count_number_actives_predictions
        )


class EMD_module(Metric):
    def __init__(
        self,
        data_loading_mode,
        distance,
        rad2deg,
        sigma_classes_deg,
        sigma_points_mode=None,
        class_mode=False,
        num_sources_per_sample_min=0,
        N_samples_mog=15,
        conf_mode=True,
        grid_mode=False,
    ):

        super(EMD_module, self).__init__()

        self.data_loading_mode = data_loading_mode
        self.distance = distance
        self.rad2deg = rad2deg
        self.class_mode = class_mode
        self.num_sources_per_sample_min = num_sources_per_sample_min
        self.N_samples_mog = N_samples_mog
        self.conf_mode = conf_mode
        self.grid_mode = grid_mode

        if self.distance == "spherical-squared":
            self.distance = "spherical"

        if self.distance not in ["euclidean", "spherical"]:
            raise ValueError("The distance must be either euclidean or spherical.")

        if self.data_loading_mode not in ["normal", "noisy-classes"]:
            raise ValueError(
                "The data loading mode must be either normal or noisy-class"
            )

        if data_loading_mode == "noisy-classes":
            self.sigma_classes = sigma_classes_deg
            self.sigma_points_mode = sigma_points_mode

    def compute(self, predictions, targets):

        if self.data_loading_mode == "normal":
            emd = self.emd_metric_normal(predictions, targets)

        elif self.data_loading_mode == "noisy-classes":
            emd = self.emd_metric_class(predictions, targets)

        return emd

    def emd_metric_normal(self, predictions, targets):
        """Compute the EMD (or Wasserstein metric) metric between the multihypothesis predictions, viewed as a mixture of diracs, and the ground truth,
        also viewed as a mixture of diracs with number of modes equal to the number of sources.

        Args:
            (in conf_mode) predictions (torch.Tensor,torch.Tensor): Shape [batchxTxself.num_hypothesisx2],[batchxTxself.num_hypothesisx1]
            targets (torch.Tensor,torch.Tensor): Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            conf_mode (bool): If True, the predictions are in the form (hypothesis_DOAs, hypothesis_confidences), otherwise the predictions are in the form (hypothesis_DOAs). Default: True.
            distance (str): If 'euclidean', the distance between the diracs is computed using the euclidean distance. Default: 'euclidean'.
            dataset_idx (int): Index of the dataset. Default: 0.
            batch_idx (int): Index of the batch. Default: 0.

        Return:
        emd distance (torch.tensor)"""
        if self.conf_mode is True:
            if self.class_mode:
                hyps_DOAs_pred_stacked, conf_stacked, hyp_output_classes = (
                    predictions  # Shape ([batchxTxself.num_hypothesisx2],[batchxTxself.num_hypothesisx1],[batchxTxself.num_hypothesisxnum_classes])
                )
            else:
                if len(predictions) == 3:
                    hyps_DOAs_pred_stacked, conf_stacked, _ = (
                        predictions  # Shape ([batchxTxself.num_hypothesisx2],[batchxTxself.num_hypothesisx1])
                    )
                else:
                    hyps_DOAs_pred_stacked, conf_stacked = (
                        predictions  # Shape ([batchxTxself.num_hypothesisx2],[batchxTxself.num_hypothesisx1])
                    )
        else:
            if len(predictions) == 2:
                hyps_DOAs_pred_stacked, _ = (
                    predictions  # Shape [batchxTxself.num_hypothesisx2]
                )
            else:
                hyps_DOAs_pred_stacked, _, _ = (
                    predictions  # Shape [batchxTxself.num_hypothesisx2]
                )
            conf_stacked = torch.ones_like(
                hyps_DOAs_pred_stacked[:, :, :, :1]
            )  # Shape [batchxTxself.num_hypothesisx1]

        if len(targets) == 3:
            source_activity_target, direction_of_arrival_target, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        elif len(targets) == 2:
            source_activity_target, direction_of_arrival_target = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        elif len(targets) == 4:
            source_activity_target, direction_of_arrival_target, _, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )

        if self.class_mode:
            direction_of_arrival_target = direction_of_arrival_target.sum(
                dim=-2
            )  # [batch,T,Max_sources,num_cls,2] -> [batch,T,Max_sources,2]
            source_activity_target = source_activity_target.sum(
                dim=-1
            )  # [batch,T,Max_sources,num_cls] -> [batch,T,Max_sources]

        # Convert tensor to numpy arrays and ensure they are float 32
        source_activity_target = (
            source_activity_target.detach().cpu().numpy().astype(np.float32)
        )
        conf_stacked = (
            conf_stacked.detach().cpu().numpy().astype(np.float32)
        )  # Shape [batchxTxself.num_hypothesisx1]

        direction_of_arrival_target = (
            direction_of_arrival_target.detach().cpu().numpy().astype(np.float32)
        )
        hyps_DOAs_pred_stacked = (
            hyps_DOAs_pred_stacked.detach().cpu().numpy().astype(np.float32)
        )

        batch, T, num_hyps, _ = hyps_DOAs_pred_stacked.shape
        Max_sources = source_activity_target.shape[2]
        emd_matrix = np.zeros((T, batch))

        conf_sum = conf_stacked.sum(
            axis=2, keepdims=True
        )  # [batchxTxnum_hypothesisx1] (constant in the num_hypothesis axis)
        conf_stacked_normalized = np.divide(
            conf_stacked,
            conf_sum,
            out=np.full_like(conf_stacked, np.nan),
            where=conf_sum != 0,
        )
        source_activity_target_sum = source_activity_target.sum(
            axis=2, keepdims=True
        )  # Shape [batchxTxMax_sources] (constant in the Max_sources axis)
        source_activity_target_normalized = np.divide(
            source_activity_target,
            source_activity_target_sum,
            out=np.full_like(source_activity_target, np.nan),
            where=source_activity_target_sum != 0,
        )

        signature_source = np.concatenate(
            (conf_stacked_normalized, hyps_DOAs_pred_stacked), axis=3
        )  # [batchxTxself.num_hypothesisx3]
        signature_target = np.concatenate(
            (
                source_activity_target_normalized[:, :, :, None],
                direction_of_arrival_target,
            ),
            axis=3,
        )  # [batchxTxMax_sourcesx3]

        for t in range(T):
            for number_in_batch in range(batch):
                if (
                    source_activity_target_sum[number_in_batch, t].sum()
                    <= self.num_sources_per_sample_min
                ):
                    emd_matrix[t, number_in_batch] = np.nan
                elif conf_sum[number_in_batch, t].sum() == 0:
                    emd_matrix[t, number_in_batch] = np.nan
                else:
                    if self.distance == "euclidean":
                        #### With euclidean distance
                        emd = cv2.EMD(
                            signature_source[number_in_batch, t],
                            signature_target[number_in_batch, t],
                            cv2.DIST_L2,
                        )
                        emd_matrix[t, number_in_batch] = emd[0]
                    elif self.distance == "spherical":
                        #### With spherical distance
                        # cost_matrix = create_cost_matrix(signature_source[number_in_batch, t], signature_target[number_in_batch, t], compute_spherical_distance_np)
                        cost_matrix = compute_accelerated_angular_distance(
                            signature_source[number_in_batch, t][:, 1:],
                            signature_target[number_in_batch, t][:, 1:],
                        )
                        emd = cv2.EMD(
                            signature_source[number_in_batch, t],
                            signature_target[number_in_batch, t],
                            cv2.DIST_USER,
                            cost_matrix,
                        )
                        if self.rad2deg is True:
                            emd_matrix[t, number_in_batch] = np.rad2deg(emd[0])
                        else:
                            emd_matrix[t, number_in_batch] = emd[0]

        return torch.tensor(np.nanmean(emd_matrix, dtype=np.float32)), torch.tensor(
            np.nanstd(emd_matrix, dtype=np.float32)
        )

    def emd_metric_class(self, predictions, targets):
        """Compute the EMD (or Wasserstein metric) metric between the multihypothesis predictions, viewed as a mixture of diracs, and the ground truth,
        also viewed as a mixture of diracs with number of modes equal to the number of sources.

        Args:
            (in conf_mode) predictions (torch.Tensor,torch.Tensor): Shape [batchxTxself.num_hypothesisx2],[batchxTxself.num_hypothesisx1]
            targets (torch.Tensor,torch.Tensor): Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            conf_mode (bool): If True, the predictions are in the form (hypothesis_DOAs, hypothesis_confidences), otherwise the predictions are in the form (hypothesis_DOAs). Default: True.
            distance (str): If 'euclidean', the distance between the diracs is computed using the euclidean distance. Default: 'euclidean'.
            dataset_idx (int): Index of the dataset. Default: 0.
            batch_idx (int): Index of the batch. Default: 0.

        Return:
        emd distance (torch.tensor)"""
        if self.sigma_classes is None:
            raise ValueError("The sigma of the classes must be specified.")
        if self.sigma_points_mode is None:
            raise ValueError("The evaluation mode must be specified.")

        hyps_DOAs_pred_stacked, conf_stacked = predictions[0], predictions[1]

        assert len(targets) == 4, "The EMD is not in the correct mode"
        (
            source_activity_target,
            direction_of_arrival_target,
            source_activity_target_classes,
            direction_of_arrival_target_classes,
        ) = targets

        # Convert tensor to numpy arrays and ensure they are float 32
        source_activity_target = (
            source_activity_target.detach().cpu().numpy().astype(np.float32)
        )
        conf_stacked = (
            conf_stacked.detach().cpu().numpy().astype(np.float32)
        )  # Shape [batchxTxself.num_hypothesisx1]

        direction_of_arrival_target = (
            direction_of_arrival_target.detach().cpu().numpy().astype(np.float32)
        )
        direction_of_arrival_target_classes = (
            direction_of_arrival_target_classes.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        source_activity_target_classes = (
            source_activity_target_classes.detach().cpu().numpy().astype(np.float32)
        )
        hyps_DOAs_pred_stacked = (
            hyps_DOAs_pred_stacked.detach().cpu().numpy().astype(np.float32)
        )

        batch, T, num_hyps, _ = hyps_DOAs_pred_stacked.shape
        Max_sources = source_activity_target.shape[2]
        emd_matrix = np.zeros((T, batch))

        conf_sum = conf_stacked.sum(
            axis=2, keepdims=True
        )  # [batchxTxnum_hypothesisx1] (constant in the num_hypothesis axis)
        conf_stacked_normalized = np.divide(
            conf_stacked,
            conf_sum,
            out=np.full_like(conf_stacked, np.nan),
            where=conf_sum != 0,
        )
        source_activity_target_sum = source_activity_target.sum(
            axis=2, keepdims=True
        )  # Shape [batchxTxMax_sources] (constant in the Max_sources axis)
        source_activity_target_normalized = np.divide(
            source_activity_target,
            source_activity_target_sum,
            out=np.full_like(source_activity_target, np.nan),
            where=source_activity_target_sum != 0,
        )

        signature_source = np.concatenate(
            (conf_stacked_normalized, hyps_DOAs_pred_stacked), axis=3
        )  # [batchxTxself.num_hypothesisx3]

        sigma_classes_rad = np.array(
            [np.deg2rad(sigma) for sigma in self.sigma_classes]
        )

        if self.sigma_points_mode is False:
            samples_targets = sample_gaussian_mixture_classes(
                mus=direction_of_arrival_target_classes,
                source_activity_target_classes=source_activity_target_classes,
                num_samples=self.N_samples_mog,
                sigma_classes=sigma_classes_rad,
            ).astype(np.float32)
            # samples_target of shape [batch,T,N_samples_mog,2]
            is_frame_not_empty = (
                source_activity_target_normalized[:, :, :].sum(axis=-1) > 0
            )  # shape (batch, T)
            is_frame_not_empty = np.expand_dims(
                is_frame_not_empty, axis=-1
            )  # shape (batch, T,1)
            is_frame_not_empty = np.expand_dims(
                np.repeat(is_frame_not_empty, repeats=self.N_samples_mog, axis=-1),
                axis=-1,
            ).astype(np.float32)
            # is_frame_not empty shape (batch, T, N_samples_mog, 1)
            source_activity_target_reshaped = np.where(
                is_frame_not_empty, is_frame_not_empty, np.nan
            )  # [batch,T,N_samples_mog,1]
            source_activity_target_normalized = np.divide(
                source_activity_target_reshaped,
                source_activity_target_reshaped.sum(axis=2, keepdims=True),
                out=np.full_like(source_activity_target_reshaped, np.nan),
                where=source_activity_target_reshaped.sum(axis=2, keepdims=True) > 0,
            )
            signature_target = np.concatenate(
                (source_activity_target_normalized, samples_targets), axis=3
            )  # [batch,T,N_samples_mog,3]
            # num_sources_per_sample_max = N_samples_mog*3
        else:
            samples_targets = sample_sigma_points_classes(
                mus=direction_of_arrival_target_classes,
                source_activity_target_classes=source_activity_target_classes,
                sigma_classes=sigma_classes_rad,
            ).astype(np.float32)
            # samples_target of shape [batch,T,Max_sources,5,2]
            samples_targets = samples_targets.reshape(batch, T, Max_sources * 5, 2)
            source_activity_target_reshaped = np.repeat(
                source_activity_target_normalized, repeats=5, axis=-1
            )  # batch, T, Max_sources, 5
            source_activity_target_reshaped = source_activity_target_reshaped.reshape(
                batch, T, Max_sources * 5, 1
            )  # batch, T, Max_sources*5,1
            source_activity_target_sum = source_activity_target_reshaped.sum(
                axis=2, keepdims=True
            )
            source_activity_target_normalized = np.divide(
                source_activity_target_reshaped,
                source_activity_target_sum,
                out=np.full_like(source_activity_target_reshaped, np.nan),
                where=source_activity_target_sum > 0,
            )
            signature_target = np.concatenate(
                (source_activity_target_normalized, samples_targets), axis=3
            )  # [batch,T,Max_sources*5,3]
            assert signature_target.shape == (batch, T, Max_sources * 5, 3)
            # N_samples_mog = Max_sources*5
            # num_sources_per_sample_max = num_sources_per_sample_max*5

        for t in range(T):
            for number_in_batch in range(batch):
                # assert np.isnan(source_activity_target_sum[number_in_batch, t].sum()), "check num source max setting"
                if (
                    np.isnan(source_activity_target_reshaped[number_in_batch, t].sum())
                    or source_activity_target_sum[number_in_batch, t].sum()
                    <= self.num_sources_per_sample_min
                ):
                    emd_matrix[t, number_in_batch] = np.nan
                elif conf_sum[number_in_batch, t].sum() == 0:
                    emd_matrix[t, number_in_batch] = np.nan
                else:
                    if self.distance == "euclidean":
                        #### With euclidean distance
                        emd = cv2.EMD(
                            signature_source[number_in_batch, t],
                            signature_target[number_in_batch, t],
                            cv2.DIST_L2,
                        )
                        emd_matrix[t, number_in_batch] = emd[0]
                    elif self.distance == "spherical":
                        #### With spherical distance
                        cost_matrix = compute_accelerated_angular_distance(
                            signature_source[number_in_batch, t][:, 1:],
                            signature_target[number_in_batch, t][:, 1:],
                        )
                        emd = cv2.EMD(
                            signature_source[number_in_batch, t],
                            signature_target[number_in_batch, t],
                            cv2.DIST_USER,
                            cost_matrix,
                        )
                        if self.rad2deg is True:
                            emd_matrix[t, number_in_batch] = np.rad2deg(emd[0])
                        else:
                            emd_matrix[t, number_in_batch] = emd[0]

        return torch.tensor(np.nanmean(emd_matrix, dtype=np.float32)), torch.tensor(
            np.nanstd(emd_matrix, dtype=np.float32)
        )


def compute_accelerated_angular_distance(x, y):
    """
    Computes the angular distance between each pair of points in x and y.

    :param x: Array of shape (n, 2) representing n points in spherical coordinates.
    :param y: Array of shape (m, 2) representing m points in spherical coordinates.
    :return: Array of shape (n, m) containing angular distances.
    """
    sin_x1 = np.sin(x[:, 1])[:, np.newaxis]
    sin_y1 = np.sin(y[:, 1])
    cos_x1 = np.cos(x[:, 1])[:, np.newaxis]
    cos_y1 = np.cos(y[:, 1])
    cos_diff_x0_y0 = np.cos(y[:, 0] - x[:, 0][:, np.newaxis])

    cos_angle = np.clip(sin_x1 * sin_y1 + cos_x1 * cos_y1 * cos_diff_x0_y0, -1.0, 1.0)

    return np.arccos(cos_angle)


def create_cost_matrix(signature1, signature2, distance_func):
    num_rows = signature1.shape[0]
    num_cols = signature2.shape[0]

    cost_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            cost_matrix[i, j] = distance_func(
                signature1[i, 1:].reshape(1, -1), signature2[j, 1:].reshape(1, -1)
            )

    return cost_matrix


def compute_spherical_distance_np(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (np.ndarray): Numpy array of predicted azimuth and elevation angles.
        y_true (np.ndarray): Numpy array of ground-truth azimuth and elevation angles.

    Returns:
        np.ndarray: Numpy array of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        raise RuntimeError("Input arrays require a dimension of two.")

    sine_term = np.sin(y_pred[:, 1]) * np.sin(y_true[:, 1])
    cosine_term = (
        np.cos(y_pred[:, 1])
        * np.cos(y_true[:, 1])
        * np.cos(y_true[:, 0] - y_pred[:, 0])
    )

    return np.arccos(np.clip(sine_term + cosine_term, a_min=-1, a_max=1))


def sample_sigma_points(mus, distances, source_activity, sigma=None):
    if sigma is None:
        raise ValueError("The noise level must be specified")
    batch_size, T, max_gaussians, _ = mus.shape
    Max_sources = 3
    samples = np.zeros(
        (batch_size, T, Max_sources, 5, 2)
    )  # Initialize with zero values

    for i in range(batch_size):
        for t in range(T):
            active_gaussians = np.where(source_activity[i, t] > 0)[0]
            num_active = len(active_gaussians)
            if num_active == 0:
                continue
            sigma_y = sigma / (distances[i, t, active_gaussians])  # shape [num_active]
            sigma_x = sigma / (
                (
                    distances[i, t, active_gaussians]
                    * np.cos(mus[i, t, active_gaussians, 1])
                )
            )  # shape [num_active]
            sigmas_x = np.stack(
                (sigma_x, np.repeat(np.array([0]), repeats=num_active)), axis=1
            )  # [num_active,2]
            sigmas_y = np.stack(
                (np.repeat(np.array([0]), repeats=num_active), sigma_y), axis=1
            )  # [num_active,2]
            assert sigmas_x.shape == (num_active, 2)
            samples[i, t, active_gaussians, 0] = mus[i, t, active_gaussians]
            samples[i, t, active_gaussians, 1] = mus[i, t, active_gaussians] + sigmas_x
            samples[i, t, active_gaussians, 2] = mus[i, t, active_gaussians] - sigmas_x
            samples[i, t, active_gaussians, 3] = mus[i, t, active_gaussians] + sigmas_y
            samples[i, t, active_gaussians, 4] = mus[i, t, active_gaussians] - sigmas_y

    return samples


def sample_gaussian_mixture_classes(
    mus,
    source_activity_target_classes,
    num_samples,
    sigma_classes=None,
    dataset="ansim",
):
    """Sample from a Gaussian mixture of classes.

    inputs:
    mus : shape [batch,T,max_gaussians,num_classes,2]
    source_activity_target_classes : shape [batch,T,max_gaussians,num_classes]
    sigma_classes : shape [num_classes]
    num_samples : int

    output:

    samples : shape [batch,T,num_samples,2]"""

    if sigma_classes is None:
        raise ValueError("The sigma_classes must be specified")

    # Source activity target classes of shape [batch,T,max_gaussians,num_classes]
    # mus of shape [batch,T,max_gaussians,num_classes,2]

    batch_size, T, max_gaussians, num_classes, _ = mus.shape
    samples = np.zeros((batch_size, T, num_samples, 2))  # Initialize with zero values

    if dataset == "ansim" or dataset == "resim" or dataset == "mansyn":
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
        # sigma_classes = np.array([np.deg2rad(5+5*i) for i in range(len(classes))])

    else:
        raise NotImplementedError

    for i in range(batch_size):
        for t in range(T):
            active_gaussians_index, classes_numbers = np.where(
                source_activity_target_classes[i, t] > 0
            )
            num_active = len(active_gaussians_index)
            if num_active == 0:
                continue
            gauss_indices = np.random.choice(
                active_gaussians_index, size=num_samples, replace=True
            )
            a = np.where(source_activity_target_classes[i, t, gauss_indices] > 0)[
                1
            ]  # Class number associated with the selected index
            sigma = sigma_classes[a]  # shape [num_samples]
            sigma_y = sigma
            sigma_x = sigma  # shape [num_active]

            sigmas = np.stack((sigma_x, sigma_y), axis=1)  # [num_samples,2]
            samples[i, t] = np.random.normal(
                mus[i, t, gauss_indices, :, :].sum(axis=-2), sigmas
            )

    return samples


def sample_sigma_points_classes(
    mus, source_activity_target_classes, sigma_classes=None, dataset="ansim"
):
    """Sample sigma points from the Gaussian mixture of classes.

    inputs:
    mus: shape [batch,T,max_sources,num_classes,2]
    source_activity_target_classes: shape [batch,T,max_sources,num_classes]
    sigma_classes : shape [num_classes]

    output:
    samples: shape [batch,T,max_sources,5,2]"""

    if sigma_classes is None:
        raise ValueError("The sigma classes must be specified")
    batch_size, T, max_gaussians, num_classes, _ = mus.shape
    Max_sources = 3
    samples = np.zeros(
        (batch_size, T, Max_sources, 5, 2)
    )  # Initialize with zero values

    if dataset == "ansim" or dataset == "resim" or dataset == "mansyn":
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
        # sigma_classes = np.array([np.deg2rad(5+5*i) for i in range(len(classes))])

    else:
        raise NotImplementedError

    for i in range(batch_size):
        for t in range(T):
            active_idx, active_gaussians = np.where(
                source_activity_target_classes[i, t] > 0
            )  # self.max_num_sources, self.num_unique_classes]
            num_active = len(active_gaussians)
            if num_active == 0:
                continue
            sigma = sigma_classes[active_gaussians]  # shape [num_active]

            sigma_y = sigma
            sigma_x = sigma  # shape [num_active]
            sigmas_x = np.stack(
                (sigma_x, np.repeat(np.array([0]), repeats=num_active)), axis=1
            )  # [num_active,2]
            sigmas_y = np.stack(
                (np.repeat(np.array([0]), repeats=num_active), sigma_y), axis=1
            )  # [num_active,2]
            assert sigmas_x.shape == (num_active, 2)
            samples[i, t, active_idx, 0, :] = mus[i, t, :, active_gaussians, :].sum(
                axis=-2
            )
            samples[i, t, active_idx, 1, :] = (
                mus[i, t, :, active_gaussians, :].sum(axis=-2) + sigmas_x
            )
            samples[i, t, active_idx, 2, :] = (
                mus[i, t, :, active_gaussians, :].sum(axis=-2) - sigmas_x
            )
            samples[i, t, active_idx, 3, :] = (
                mus[i, t, :, active_gaussians, :].sum(axis=-2) + sigmas_y
            )
            samples[i, t, active_idx, 4, :] = (
                mus[i, t, :, active_gaussians, :].sum(axis=-2) - sigmas_y
            )

    return samples  #
