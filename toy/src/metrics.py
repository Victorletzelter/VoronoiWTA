import cv2
import numpy as np
import torch


class Metric:
    def __init__(self):
        pass

    def compute(self, predictions, targets):
        pass


class Oracle_module(Metric):
    def __init__(
        self,
        distance,
        computation_type="dirac",
    ):

        super(Oracle_module, self).__init__()

        self.distance = distance
        self.computation_type = computation_type

    def compute(self, predictions, targets):

        if self.computation_type == "dirac":
            return self.oracle_error(predictions, targets)

        else:
            # Not implemented Error
            raise NotImplementedError

    def oracle_error(self, predictions, targets):
        """The oracle DOA error compute the minimum distance between the hypothesis predicted and the
        ground truth, for each ground truth.

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxTxself.num_hypothesisx2]
            targets (torch.Tensor,torch.Tensor): Shape [batch,T,Max_sources],[batch,T,Max_sources,output_dim]

        Return:
            oracle_doa_error (torch.tensor)
        """
        direction_of_arrival_target, source_activity_target = targets[0], targets[1]

        # assert direction_of_arrival_target.shape[-1] != 1, "There is an issue of shape"

        hyps_stacked = predictions[0]

        batch, num_hyps, _ = hyps_stacked.shape
        Max_sources = source_activity_target.shape[2]
        doa_error_matrix_new = np.zeros((1, 1))
        count_number_actives_predictions = 0  # This variable counts the number of predictions (for element in the batch and each time step) that are active

        # hyps_stacked of shape [batch,num_hyps,2]
        # source_activity_target of shape [batch,Max_sources]
        # direction_of_arrival_target of shape [batch,Max_sources,output_dim]

        filling_value = 10000  # Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings.

        # 1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target == 0

        mask_inactive_sources = mask_inactive_sources.expand_as(
            direction_of_arrival_target
        )

        direction_of_arrival_target[mask_inactive_sources] = (
            filling_value  # Shape [batch,Max_sources,output_dim]
        )

        # The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the
        # tensors gts[batch,i,num_hypothesis,output_dim] contains duplicates of direction_of_arrival_target_t along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,output_dim] may contain inactive sources, and therefore
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts = direction_of_arrival_target.unsqueeze(2).repeat(
            1, 1, num_hyps, 1
        )  # Shape [batch,Max_sources,num_hypothesis,output_dim]

        # We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_stacked_duplicated = hyps_stacked.unsqueeze(1).repeat(
            1, Max_sources, 1, 1
        )  # Shape [batch,Max_sources,num_hypothesis,output_dim]

        if self.distance == "euclidean":
            #### With euclidean distance
            diff = torch.square(
                hyps_stacked_duplicated - gts
            )  # Shape [batch,Max_sources,num_hypothesis,output_dim]
            channels_sum = torch.sum(
                diff, dim=-1
            )  # Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = torch.sqrt(
                channels_sum
            )  # Distance matrix [batch,Max_sources,num_hypothesis]

        wta_dist_matrix, _ = torch.min(
            dist_matrix, dim=2
        )  # wta_dist_matrix of shape [batch,Max_sources]
        mask = wta_dist_matrix <= filling_value / 2
        wta_dist_matrix = (
            wta_dist_matrix * mask
        )  # [batch,Max_sources], we select only the active sources.
        count_non_zeros = torch.sum(
            mask != 0
        )  # We count the number of actives sources for the computation of the mean (below).

        count_number_actives_predictions += torch.sum(torch.sum(mask != 0, dim=1) != 0)

        if count_non_zeros > 0:

            num_sources_per_sample = torch.sum(
                source_activity_target.float()[:, :, 0], dim=1, keepdim=True
            ).repeat(1, Max_sources)[
                :,
                :,
            ]  # [batch,Max_sources]
            # assert num_sources_per_sample.shape == (batch,Max_sources)
            # assert torch.sum(num_sources_per_sample)/Max_sources == torch.sum(source_activity_target_t)
            wta_dist_matrix = torch.where(
                num_sources_per_sample > 0,
                wta_dist_matrix / num_sources_per_sample,
                torch.zeros_like(num_sources_per_sample),
            )  # We divide by the number of active sources to get the mean error per sample
            oracle_err_new = torch.sum(
                wta_dist_matrix
            )  # We compute the mean of the diff.
            if oracle_err_new == 0:
                doa_error_matrix_new[0, 0] = np.nan

            else:
                doa_error_matrix_new[0, 0] = oracle_err_new.detach().cpu().numpy()

        else:
            doa_error_matrix_new[0, 0] = np.nan

        return (
            torch.tensor(np.nansum(doa_error_matrix_new, dtype=np.float32))
            / count_number_actives_predictions
        )


class EMD_module(Metric):
    def __init__(self, distance, conf_mode=True):

        super(EMD_module, self).__init__()

        self.distance = distance
        self.conf_mode = conf_mode

        if self.distance not in ["euclidean"]:
            raise ValueError("Only the euclidean version is implemented")

    def compute(self, predictions, targets):

        return self.emd_metric(predictions, targets)

    def emd_metric(self, predictions, targets):
        """Compute the EMD (or Wasserstein metric) metric between the multihypothesis predictions, viewed as a mixture of diracs, and the ground truth,
        also viewed as a mixture of diracs with number of modes equal to the number of sources.

        Args:
            (in conf_mode) predictions (torch.Tensor,torch.Tensor): Shape [batchxself.num_hypothesisx2],[batchxself.num_hypothesisx1]
            targets (torch.Tensor,torch.Tensor): Shape [batch,Max_sources],[batch,Max_sources,output_dim]
            conf_mode (bool): If True, the predictions are in the form (hypothesis_DOAs, hypothesis_confidences), otherwise the predictions are in the form (hypothesis_DOAs). Default: True.
            distance (str): If 'euclidean', the distance between the diracs is computed using the euclidean distance. Default: 'euclidean'.
            dataset_idx (int): Index of the dataset. Default: 0.
            batch_idx (int): Index of the batch. Default: 0.

        Return:
        emd distance (torch.tensor)"""

        if self.conf_mode is True:
            hyps_stacked, conf_stacked = predictions[0], predictions[1]
        else:
            hyps_stacked = predictions[0]
            conf_stacked = torch.ones_like(hyps_stacked[:, :, :1])

        direction_of_arrival_target, source_activity_target = (
            targets[0],
            targets[1],
        )  # Shape [batch,Max_sources,output_dim],[batch,Max_sources,1]

        # # Convert tensor to numpy arrays and ensure they are float 32
        # hyps_stacked = hyps_stacked.detach().cpu().numpy().astype(np.float32) # [batchxself.num_hypothesisx2]
        # conf_stacked = (
        #     conf_stacked.detach().cpu().numpy().astype(np.float32)
        # )  # [batchxTxself.num_hypothesisx1]

        direction_of_arrival_target = (
            direction_of_arrival_target.detach().cpu().numpy().astype(np.float32)
        )
        source_activity_target = (
            source_activity_target.detach().cpu().numpy().astype(np.float32)
        )

        batch = hyps_stacked.shape[0]
        emd_matrix = np.zeros((batch))

        conf_sum = conf_stacked.sum(
            axis=1, keepdims=True
        )  # [batchxnum_hypothesisx1] (constant in the num_hypothesis axis)
        conf_stacked_normalized = np.divide(
            conf_stacked,
            conf_sum,
            out=np.full_like(conf_stacked, np.nan),
            where=conf_sum != 0,
        )
        source_activity_target_sum = source_activity_target.sum(
            axis=1, keepdims=True
        )  # Shape [batchxMax_sources] (constant in the Max_sources axis)
        source_activity_target_normalized = np.divide(
            source_activity_target,
            source_activity_target_sum,
            out=np.full_like(source_activity_target, np.nan),
            where=source_activity_target_sum != 0,
        )

        signature_source = np.concatenate(
            (conf_stacked_normalized, hyps_stacked), axis=2
        )  # [batchxself.num_hypothesisx3]
        signature_target = np.concatenate(
            (
                source_activity_target_normalized,
                direction_of_arrival_target,
            ),
            axis=2,
        )  # [batchxMax_sourcesx3]

        for number_in_batch in range(batch):
            if conf_sum[number_in_batch].sum() == 0:
                emd_matrix[number_in_batch] = np.nan
            else:
                if self.distance == "euclidean":
                    #### With euclidean distance
                    emd = cv2.EMD(
                        signature_source[number_in_batch],
                        signature_target[number_in_batch],
                        cv2.DIST_L2,
                    )
                    emd_matrix[number_in_batch] = emd[0]

        return torch.tensor(np.nanmean(emd_matrix, dtype=np.float32))
