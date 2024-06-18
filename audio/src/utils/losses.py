import torch
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Tuple

from src.utils.utils import compute_spherical_distance


class WTARisk(_Loss):
    """Custom sound event localization (SEL) loss function, which returns the sum of the binary cross-entropy loss
    regarding the estimated number of sources at each time-step and the minimum direction-of-arrival mean squared error
    loss, calculated according to all possible combinations of active sources."""

    __constants__ = ["reduction"]

    def __init__(
        self,
        max_num_sources: int,
        size_average=None,
        reduce=None,
        reduction="mean",
        mode="wta",
        top_n=1,
        distance="spherical-squared",
        epsilon=0.05,
        single_target_loss=False,
        rad2deg=False,
    ) -> None:
        super(WTARisk, self).__init__(size_average, reduce, reduction)

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.single_target_loss = single_target_loss
        self.rad2deg = rad2deg

    def compute_spherical_distance(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError("Input tensors require a dimension of two.")

        sine_term = torch.sin(y_pred[:, 1]) * torch.sin(y_true[:, 1])
        cosine_term = (
            torch.cos(y_pred[:, 1])
            * torch.cos(y_true[:, 1])
            * torch.cos(y_true[:, 0] - y_pred[:, 0])
        )

        if self.rad2deg is True:
            return (
                torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))
                * 180
                / np.pi
            )
        else:
            return torch.acos(
                F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1)
            )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for the Multi-hypothesis Sound Event Localization Loss.

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxTxself.num_hypothesisx2]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

        Returns:
            loss (torch.tensor)
            meta_data (dict)
        """
        hyps_DOAs_pred_stacked, _ = predictions  # Shape [batchxTxself.num_hypothesisx2]

        if len(targets) == 3:
            source_activity_target, direction_of_arrival_target, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        elif len(targets) == 4:
            source_activity_target, direction_of_arrival_target, _, _ = targets
        else:
            source_activity_target, direction_of_arrival_target = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )

        T = source_activity_target.shape[1]
        losses = torch.tensor(0.0)

        Number_active_frames = 0

        for t in range(T):

            source_activity_target_t = source_activity_target[:, t, :].detach()
            direction_of_arrival_target_t = direction_of_arrival_target[
                :, t, :, :
            ].detach()
            hyps_stacked_t = hyps_DOAs_pred_stacked[:, t, :, :]

            loss_t = self.draft_make_sampling_loss_ambiguous_gts(
                hyps_stacked_t=hyps_stacked_t,
                source_activity_target_t=source_activity_target_t,
                direction_of_arrival_target_t=direction_of_arrival_target_t,
                mode=self.mode,
                top_n=self.top_n,
                distance=self.distance,
                epsilon=self.epsilon,
                single_target_loss=self.single_target_loss,
            )
            if loss_t > 0:
                Number_active_frames += 1

            losses = torch.add(losses, loss_t)

        if Number_active_frames > 0:
            losses = losses / Number_active_frames

        meta_data = {"MHLoss": losses}

        return losses, meta_data

    def draft_make_sampling_loss_ambiguous_gts(
        self,
        hyps_stacked_t,
        source_activity_target_t,
        direction_of_arrival_target_t,
        mode="wta",
        top_n=1,
        distance="euclidean",
        epsilon=0.05,
        single_target_loss=False,
    ):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target_t torch.tensor): Input tensor of shape (batch,Max_sources)
            direction_of_arrival_target_t (torch.tensor): Input tensor of shape (batch,Max_sources,2)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """

        # hyps_stacked_t of shape [batch,num_hyps,2]
        # source_activity_target_t of shape [batch,Max_sources]
        # direction_of_arrival_target_t of shape [batch,Max_sources,2]

        filling_value = 1000  # Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings.
        num_hyps = hyps_stacked_t.shape[1]
        batch = source_activity_target_t.shape[0]
        Max_sources = source_activity_target_t.shape[1]

        # 1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target_t == 0
        mask_inactive_sources = mask_inactive_sources.unsqueeze(-1).expand_as(
            direction_of_arrival_target_t
        )
        direction_of_arrival_target_t[mask_inactive_sources] = (
            filling_value  # Shape [batch,Max_sources,2]
        )

        # We can check whether the operation is performed correctly
        # assert (source_activity_target_t.sum(axis=1).all()==(direction_of_arrival_target_t[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target_t.sum(axis=1).all()==(direction_of_arrival_target_t[:,:,1]!=filling_value).sum(axis=1).all())

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

        eps = 0.001

        if distance == "euclidean":
            #### With euclidean distance
            diff = torch.square(
                hyps_stacked_t_duplicated - gts
            )  # Shape [batch,Max_sources,num_hypothesis,2]
            channels_sum = torch.sum(
                diff, dim=3
            )  # Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]

            assert channels_sum.shape == (batch, Max_sources, num_hyps)
            assert (channels_sum >= 0).all()

            dist_matrix = torch.sqrt(
                channels_sum + eps
            )  # Distance matrix [batch,Max_sources,num_hypothesis]

            assert dist_matrix.shape == (batch, Max_sources, num_hyps)

        elif distance == "spherical":

            dist_matrix_euclidean = torch.sqrt(
                torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
            )  # We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation.

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                -1, 2
            )  # Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
            diff = self.compute_spherical_distance(hyps_stacked_t_duplicated, gts)
            dist_matrix = diff.view(
                batch, Max_sources, num_hyps
            )  # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value / 2] = (
                filling_value  # We fill the parts corresponding to false gts.
            )

        elif distance == "spherical-squared":

            dist_matrix_euclidean = torch.sqrt(
                torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
            )  # We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation.

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                -1, 2
            )  # Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
            diff = torch.square(
                self.compute_spherical_distance(hyps_stacked_t_duplicated, gts)
            )
            dist_matrix = diff.view(
                batch, Max_sources, num_hyps
            )  # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value / 2] = (
                filling_value  # We fill the parts corresponding to false gts.
            )

        sum_losses = torch.tensor(0.0)

        if mode == "wta":

            if single_target_loss == True:
                wta_dist_matrix, idx_selected = torch.min(
                    dist_matrix, dim=2
                )  # wta_dist_matrix of shape [batch,Max_sources]
                wta_dist_matrix, idx_source_selected = torch.min(wta_dist_matrix, dim=1)
                wta_dist_matrix = wta_dist_matrix.unsqueeze(-1)  # [batch,1]

                assert wta_dist_matrix.shape == (batch, 1)
                if distance == "spherical" or distance == "spherical-squared":
                    eucl_wta_dist_matrix, _ = torch.min(
                        dist_matrix_euclidean, dim=2
                    )  # wta_dist_matrix of shape [batch,Max_sources] for mask purpose
                    eucl_wta_dist_matrix, _ = torch.min(eucl_wta_dist_matrix, dim=1)
                    eucl_wta_dist_matrix = eucl_wta_dist_matrix.unsqueeze(
                        -1
                    )  # [batch,1]
                    mask = (
                        eucl_wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                else:
                    mask = (
                        wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                wta_dist_matrix = (
                    wta_dist_matrix * mask
                )  # [batch,1], we select only the active sources.

                count_non_zeros = torch.sum(
                    mask != 0
                )  # We count the number of actives sources for the computation of the mean (below).

            else:
                wta_dist_matrix, idx_selected = torch.min(
                    dist_matrix, dim=2
                )  # wta_dist_matrix of shape [batch,Max_sources]
                if distance == "spherical" or distance == "spherical-squared":
                    eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                    mask = (
                        eucl_wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                else:
                    mask = (
                        wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                wta_dist_matrix = (
                    wta_dist_matrix * mask
                )  # [batch,Max_sources], we select only the active sources.
                count_non_zeros = torch.sum(
                    mask != 0
                )  # We count the number of actives sources for the computation of the mean (below).

            if count_non_zeros > 0:
                loss = (
                    torch.sum(wta_dist_matrix) / count_non_zeros
                )  # We compute the mean of the diff.
            else:
                loss = torch.tensor(0.0)

            sum_losses = torch.add(sum_losses, loss)

        N = torch.sum(source_activity_target_t.sum(axis=1) > 0)  # Shape [batch]

        if N > 0:
            return sum_losses / N
        else:
            return sum_losses


class MHSELLoss(_Loss):
    """Custom sound event localization (SEL) loss function, which returns the sum of the binary cross-entropy loss
    regarding the estimated number of sources at each time-step and the minimum direction-of-arrival mean squared error
    loss, calculated according to all possible combinations of active sources."""

    __constants__ = ["reduction"]

    def __init__(
        self,
        max_num_sources: int,
        size_average=None,
        reduce=None,
        reduction="mean",
        mode="wta",
        top_n=1,
        distance="euclidean",
        epsilon=0.05,
        single_target_loss=False,
        rad2deg=False,
    ) -> None:
        super(MHSELLoss, self).__init__(size_average, reduce, reduction)

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.single_target_loss = single_target_loss
        self.rad2deg = rad2deg

    def compute_spherical_distance(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
            assert RuntimeError("Input tensors require a dimension of two.")

        sine_term = torch.sin(y_pred[:, 1]) * torch.sin(y_true[:, 1])
        cosine_term = (
            torch.cos(y_pred[:, 1])
            * torch.cos(y_true[:, 1])
            * torch.cos(y_true[:, 0] - y_pred[:, 0])
        )

        if self.rad2deg is True:
            return (
                torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))
                * 180
                / np.pi
            )
        else:
            return torch.acos(
                F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1)
            )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for the Multi-hypothesis Sound Event Localization Loss.

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxTxself.num_hypothesisx2]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

        Returns:
            loss (torch.tensor)
            meta_data (dict)
        """
        hyps_DOAs_pred_stacked, _ = predictions  # Shape [batchxTxself.num_hypothesisx2]

        if len(targets) == 3:
            source_activity_target, direction_of_arrival_target, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        elif len(targets) == 4:
            source_activity_target, direction_of_arrival_target, _, _ = targets
        else:
            source_activity_target, direction_of_arrival_target = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )

        T = source_activity_target.shape[1]
        losses = torch.tensor(0.0)

        for t in range(T):

            source_activity_target_t = source_activity_target[:, t, :].detach()
            direction_of_arrival_target_t = direction_of_arrival_target[
                :, t, :, :
            ].detach()
            hyps_stacked_t = hyps_DOAs_pred_stacked[:, t, :, :]

            loss_t = self.draft_make_sampling_loss_ambiguous_gts(
                hyps_stacked_t=hyps_stacked_t,
                source_activity_target_t=source_activity_target_t,
                direction_of_arrival_target_t=direction_of_arrival_target_t,
                mode=self.mode,
                top_n=self.top_n,
                distance=self.distance,
                epsilon=self.epsilon,
                single_target_loss=self.single_target_loss,
            )
            losses = torch.add(losses, loss_t)

        losses = losses / T

        meta_data = {"MHLoss": losses}

        return losses, meta_data

    def draft_make_sampling_loss_ambiguous_gts(
        self,
        hyps_stacked_t,
        source_activity_target_t,
        direction_of_arrival_target_t,
        mode="wta",
        top_n=1,
        distance="euclidean",
        epsilon=0.05,
        single_target_loss=False,
    ):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target_t torch.tensor): Input tensor of shape (batch,Max_sources)
            direction_of_arrival_target_t (torch.tensor): Input tensor of shape (batch,Max_sources,2)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """

        # hyps_stacked_t of shape [batch,num_hyps,2]
        # source_activity_target_t of shape [batch,Max_sources]
        # direction_of_arrival_target_t of shape [batch,Max_sources,2]

        filling_value = 1000  # Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings.
        num_hyps = hyps_stacked_t.shape[1]
        batch = source_activity_target_t.shape[0]
        Max_sources = source_activity_target_t.shape[1]

        # 1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target_t == 0
        mask_inactive_sources = mask_inactive_sources.unsqueeze(-1).expand_as(
            direction_of_arrival_target_t
        )
        direction_of_arrival_target_t[mask_inactive_sources] = (
            filling_value  # Shape [batch,Max_sources,2]
        )

        # We can check whether the operation is performed correctly
        # assert (source_activity_target_t.sum(axis=1).all()==(direction_of_arrival_target_t[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target_t.sum(axis=1).all()==(direction_of_arrival_target_t[:,:,1]!=filling_value).sum(axis=1).all())

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

        eps = 0.001

        if distance == "euclidean":
            #### With euclidean distance
            diff = torch.square(
                hyps_stacked_t_duplicated - gts
            )  # Shape [batch,Max_sources,num_hypothesis,2]
            channels_sum = torch.sum(
                diff, dim=3
            )  # Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]

            assert channels_sum.shape == (batch, Max_sources, num_hyps)
            assert (channels_sum >= 0).all()

            dist_matrix = torch.sqrt(
                channels_sum + eps
            )  # Distance matrix [batch,Max_sources,num_hypothesis]

            assert dist_matrix.shape == (batch, Max_sources, num_hyps)

        elif distance == "spherical":

            dist_matrix_euclidean = torch.sqrt(
                torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
            )  # We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation.

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                -1, 2
            )  # Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
            diff = self.compute_spherical_distance(hyps_stacked_t_duplicated, gts)
            dist_matrix = diff.view(
                batch, Max_sources, num_hyps
            )  # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value / 2] = (
                filling_value  # We fill the parts corresponding to false gts.
            )

        elif distance == "spherical-squared":

            dist_matrix_euclidean = torch.sqrt(
                torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
            )  # We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation.

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                -1, 2
            )  # Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
            diff = torch.square(
                self.compute_spherical_distance(hyps_stacked_t_duplicated, gts)
            )
            dist_matrix = diff.view(
                batch, Max_sources, num_hyps
            )  # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value / 2] = (
                filling_value  # We fill the parts corresponding to false gts.
            )

        sum_losses = torch.tensor(0.0)

        if mode == "wta":

            if single_target_loss == True:
                wta_dist_matrix, idx_selected = torch.min(
                    dist_matrix, dim=2
                )  # wta_dist_matrix of shape [batch,Max_sources]
                wta_dist_matrix, idx_source_selected = torch.min(wta_dist_matrix, dim=1)
                wta_dist_matrix = wta_dist_matrix.unsqueeze(-1)  # [batch,1]

                assert wta_dist_matrix.shape == (batch, 1)
                if distance == "spherical":
                    eucl_wta_dist_matrix, _ = torch.min(
                        dist_matrix_euclidean, dim=2
                    )  # wta_dist_matrix of shape [batch,Max_sources] for mask purpose
                    eucl_wta_dist_matrix, _ = torch.min(eucl_wta_dist_matrix, dim=1)
                    eucl_wta_dist_matrix = eucl_wta_dist_matrix.unsqueeze(
                        -1
                    )  # [batch,1]
                    mask = (
                        eucl_wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                else:
                    mask = (
                        wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                wta_dist_matrix = (
                    wta_dist_matrix * mask
                )  # [batch,1], we select only the active sources.

                count_non_zeros = torch.sum(
                    mask != 0
                )  # We count the number of actives sources for the computation of the mean (below).

            else:
                wta_dist_matrix, idx_selected = torch.min(
                    dist_matrix, dim=2
                )  # wta_dist_matrix of shape [batch,Max_sources]
                if distance == "spherical" or distance == "spherical-squared":
                    eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                    mask = (
                        eucl_wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                else:
                    mask = (
                        wta_dist_matrix <= filling_value / 2
                    )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
                wta_dist_matrix = (
                    wta_dist_matrix * mask
                )  # [batch,Max_sources], we select only the active sources.
                count_non_zeros = torch.sum(
                    mask != 0
                )  # We count the number of actives sources for the computation of the mean (below).

            if count_non_zeros > 0:
                loss = (
                    torch.sum(wta_dist_matrix) / count_non_zeros
                )  # We compute the mean of the diff.
            else:
                loss = torch.tensor(0.0)

            sum_losses = torch.add(sum_losses, loss)

        elif mode == "wta-relaxed":

            # We compute the loss for the "best" hypothesis.

            wta_dist_matrix, idx_selected = torch.min(
                dist_matrix, dim=2
            )  # wta_dist_matrix of shape [batch,Max_sources], idx_selected of shape [batch,Max_sources].

            assert wta_dist_matrix.shape == (batch, Max_sources)
            assert idx_selected.shape == (batch, Max_sources)

            if distance == "spherical" or distance == "spherical-squared":
                eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                mask = eucl_wta_dist_matrix <= filling_value / 2
            else:
                mask = (
                    wta_dist_matrix <= filling_value / 2
                )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = (
                wta_dist_matrix * mask
            )  # Shape [batch,Max_sources] ; we select only the active sources.
            count_non_zeros_1 = torch.sum(
                mask != 0
            )  # We count the number of actives sources as a sum over the batch for the computation of the mean (below).

            if count_non_zeros_1 > 0:
                loss0 = torch.multiply(
                    torch.sum(wta_dist_matrix) / count_non_zeros_1, 1 - epsilon
                )  # Scalar (average with coefficient)
            else:
                loss0 = torch.tensor(0.0)

            # We then the find the other hypothesis, and compute the epsilon weighted loss for them

            if distance == "spherical" or distance == "spherical-squared":
                large_mask = dist_matrix_euclidean <= filling_value
            else:
                # At first, we remove hypothesis corresponding to "fake" ground-truth.
                large_mask = (
                    dist_matrix <= filling_value
                )  # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = (
                dist_matrix * large_mask
            )  # Shape [batch,Max_sources,num_hypothesis].

            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(
                dist_matrix, dtype=bool
            )  # Shape [batch,Max_sources,num_hypothesis]
            mask_selected.scatter_(
                2, idx_selected.unsqueeze(-1), 1
            )  # idx_selected new shape: [batch,Max_sources,1].
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed.
            # Shape of mask_selected: [batch,Max_sources,num_hypothesis]

            ### Uncomment the loop to check that the assignement was performed correctly.
            # for i in range(batch) :
            #     for j in range(Max_sources) :
            #         assert mask_selected[i,j,idx_selected[i,j]]==1
            #         for k in range(idx_selected[i,j]) :
            #             assert mask_selected[i,j,k]==0
            #         for k in range(min(num_hyps,idx_selected[i,j]+1),num_hyps) :
            #             assert mask_selected[i,j,k]==0

            assert mask_selected.shape == (batch, Max_sources, num_hyps)

            mask_selected = (
                ~mask_selected
            )  # Shape [batch,Max_sources,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = (
                dist_matrix * mask_selected
            )  # Shape [batch,Max_sources,num_hypothesis]

            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix != 0)

            if count_non_zeros_2 > 0:
                loss = torch.multiply(
                    torch.sum(dist_matrix) / count_non_zeros_2, epsilon
                )  # Scalar for each hyp
            else:
                loss = torch.tensor(0.0)

            sum_losses = torch.add(sum_losses, loss)
            sum_losses = torch.add(sum_losses, loss0)

        elif mode == "wta-top-n" and top_n > 1:

            # dist_matrix.shape == (batch,Max_sources,num_hyps)
            # wta_dist_matrix of shape [batch,Max_sources]

            dist_matrix = torch.multiply(
                dist_matrix, -1
            )  # Shape (batch,Max_sources,num_hyps)
            top_k, indices = torch.topk(
                input=dist_matrix, k=top_n, dim=-1
            )  # top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n)
            dist_matrix_min = torch.multiply(top_k, -1)

            if distance == "spherical" or distance == "spherical-squared":
                dist_matrix_euclidean = torch.multiply(
                    dist_matrix_euclidean, -1
                )  # Shape (batch,Max_sources,num_hyps)
                top_k, _ = torch.topk(
                    input=dist_matrix_euclidean, k=top_n, dim=-1
                )  # top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n)
                dist_matrix_min_euclidean = torch.multiply(top_k, -1)
                mask = dist_matrix_min_euclidean <= filling_value / 2
            else:
                mask = (
                    dist_matrix_min <= filling_value / 2
                )  # We create a mask of shape [batch,Max_sources,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values.
            assert (
                mask[:, :, 0].all() == mask[:, :, -1].all()
            )  # This mask related should be constant in the third dimension.

            dist_matrix_min = (
                dist_matrix_min * mask
            )  # [batch,Max_sources,top_n], we select only the active sources.
            assert dist_matrix_min.shape == (batch, Max_sources, top_n)

            count_non_zeros = torch.sum(
                mask[:, :, 0] != 0
            )  # We count the number of entries (in the first two dimensions) for which the mask is different from zero.

            for i in range(top_n):

                assert count_non_zeros == torch.sum(
                    mask[:, :, i] != 0
                )  # We count the number of entries for which the mask is different from zero.

                if count_non_zeros > 0:
                    loss = torch.multiply(
                        torch.sum(dist_matrix_min[:, :, i]) / count_non_zeros, 1.0
                    )
                else:
                    loss = torch.tensor(0.0)

                sum_losses = torch.add(sum_losses, loss)

            sum_losses = sum_losses / top_n

        return sum_losses


class MHCONFSELLoss(_Loss):
    """Custom sound event localization (SEL) loss function, which returns the sum of the binary cross-entropy loss
    regarding the estimated number of sources at each time-step and the minimum direction-of-arrival mean squared error
    loss, calculated according to all possible combinations of active sources."""

    __constants__ = ["reduction"]

    def __init__(
        self,
        max_num_sources: int,
        size_average=None,
        reduce=None,
        reduction="mean",
        mode="wta",
        top_n=1,
        distance="euclidean",
        epsilon=0.05,
        conf_weight=1,
        rejection_method="uniform_negative",
        number_unconfident=3,
    ) -> None:
        super(MHCONFSELLoss, self).__init__(size_average, reduce, reduction)

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.conf_weight = conf_weight
        self.rejection_method = rejection_method
        self.number_unconfident = number_unconfident

    def compute_spherical_distance(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
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

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for the Multi-hypothesis Sound Event Localization Loss.

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxTxself.num_hypothesisx2]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

        Returns:
            loss (torch.tensor)
            meta_data (dict)
        """
        hyps_DOAs_pred_stacked, conf_pred_stacked, _ = (
            predictions  # Shape ([batchxTxself.num_hypothesisx2],[batchxTxself.num_hypothesisx1])
        )
        if len(targets) == 4:
            source_activity_target, direction_of_arrival_target, _, _ = targets
        elif len(targets) == 2:
            source_activity_target, direction_of_arrival_target = targets
        else:
            source_activity_target, direction_of_arrival_target, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        T = source_activity_target.shape[1]

        losses = torch.tensor(0.0, device=hyps_DOAs_pred_stacked.device)
        confidence_loss = torch.tensor(0.0, device=hyps_DOAs_pred_stacked.device)

        for t in range(T):

            source_activity_target_t = source_activity_target[:, t, :].detach()
            direction_of_arrival_target_t = direction_of_arrival_target[
                :, t, :, :
            ].detach()
            hyps_stacked_t = hyps_DOAs_pred_stacked[:, t, :, :]
            conf_pred_stacked_t = conf_pred_stacked[:, t, :, :]

            loss_t, confidence_loss_t = self.sampling_conf_loss_ambiguous_gts(
                hyps_stacked_t=hyps_stacked_t,
                conf_pred_stacked_t=conf_pred_stacked_t,
                source_activity_target_t=source_activity_target_t,
                direction_of_arrival_target_t=direction_of_arrival_target_t,
                mode=self.mode,
                top_n=self.top_n,
                distance=self.distance,
                epsilon=self.epsilon,
                conf_weight=self.conf_weight,
                rejection_method=self.rejection_method,
                number_unconfident=self.number_unconfident,
            )
            losses = torch.add(losses, loss_t)
            confidence_loss = torch.add(confidence_loss, confidence_loss_t)

        losses = losses / T

        # meta_data = {
        #     'MHLoss':losses,
        # }

        meta_data = {
            "WTAOnly": T * losses - confidence_loss,
            "ConfidenceLossOnly": confidence_loss,
        }

        return losses, meta_data

    def sampling_conf_loss_ambiguous_gts(
        self,
        hyps_stacked_t,
        conf_pred_stacked_t,
        source_activity_target_t,
        direction_of_arrival_target_t,
        mode="wta",
        top_n=1,
        distance="euclidean",
        epsilon=0.05,
        conf_weight=1.0,
        rejection_method="uniform_negative",
        number_unconfident=3,
    ):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target_t torch.tensor): Input tensor of shape (batch,Max_sources)
            conf_pred_stacked_t (torch.tensor): Input tensor of shape (batch,num_hyps,1)
            direction_of_arrival_target_t (torch.tensor): Input tensor of shape (batch,Max_sources,2)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """

        # hyps_stacked_t of shape [batch,num_hyps,2]
        # source_activity_target_t of shape [batch,Max_sources]
        # direction_of_arrival_target_t of shape [batch,Max_sources,2]

        filling_value = 1000  # Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings.
        num_hyps = hyps_stacked_t.shape[1]
        batch = source_activity_target_t.shape[0]
        Max_sources = source_activity_target_t.shape[1]

        # assert num_hyps >= number_unconfident, "The number of hypothesis is too small comparing to the number of unconfident hypothesis selected in the scoring" # We check that the number of hypothesis is higher than the number of "negative" hypothesis sampled.

        # 1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target_t == 0
        mask_inactive_sources = mask_inactive_sources.unsqueeze(-1).expand_as(
            direction_of_arrival_target_t
        )
        direction_of_arrival_target_t[mask_inactive_sources] = (
            filling_value  # Shape [batch,Max_sources,2]
        )

        # We can check whether the operation is performed correctly
        # assert (source_activity_target_t.sum(axis=1).all()==(direction_of_arrival_target_t[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target_t.sum(axis=1).all()==(direction_of_arrival_target_t[:,:,1]!=filling_value).sum(axis=1).all())

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

        eps = 0.001

        ### Management of the confidence part
        conf_pred_stacked_t = torch.squeeze(
            conf_pred_stacked_t, dim=-1
        )  # (batch,num_hyps), predicted confidence scores for each hypothesis.
        gt_conf_stacked_t = torch.zeros_like(
            conf_pred_stacked_t
        )  # (batch,num_hyps), will contain the ground-truth of the confidence scores.

        # assert gt_conf_stacked_t.shape == (batch,num_hyps)

        if distance == "euclidean":
            #### With euclidean distance
            diff = torch.square(
                hyps_stacked_t_duplicated - gts
            )  # Shape [batch,Max_sources,num_hyps,2]
            channels_sum = torch.sum(
                diff, dim=3
            )  # Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]

            # assert channels_sum.shape == (batch,Max_sources,num_hyps)
            # assert (channels_sum>=0).all()

            dist_matrix = torch.sqrt(
                channels_sum + eps
            )  # Distance matrix [batch,Max_sources,num_hyps]

            # assert dist_matrix.shape == (batch,Max_sources,num_hyps)

        elif distance == "spherical":

            dist_matrix_euclidean = torch.sqrt(
                torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
            )  # We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation.

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                -1, 2
            )  # Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
            diff = self.compute_spherical_distance(hyps_stacked_t_duplicated, gts)
            diff = diff.view(
                batch, Max_sources, num_hyps
            )  # Shape [batch,Max_sources,num_hyps]
            dist_matrix = diff  # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value / 2] = (
                filling_value  # We fill the parts corresponding to false gts.
            )

        elif distance == "spherical-squared":

            dist_matrix_euclidean = torch.sqrt(
                torch.sum(torch.square(hyps_stacked_t_duplicated - gts), dim=3)
            )  # We also compute the euclidean distance matrix, to use it as a mask for the spherical distance computation.

            ### With spherical distance
            hyps_stacked_t_duplicated = hyps_stacked_t_duplicated.view(
                -1, 2
            )  # Shape [batch*num_hyps*Max_sources,2]
            gts = gts.view(-1, 2)  # Shape [batch*num_hyps*Max_sources,2]
            diff = torch.square(
                compute_spherical_distance(hyps_stacked_t_duplicated, gts)
            )
            dist_matrix = diff.view(
                batch, Max_sources, num_hyps
            )  # Shape [batch,Max_sources,num_hyps]
            dist_matrix[dist_matrix_euclidean >= filling_value / 2] = (
                filling_value  # We fill the parts corresponding to false gts.
            )

        sum_losses = torch.tensor(0.0)

        if mode == "wta" or mode == "wta-diffusion":

            # We select the best hypothesis for each source
            wta_dist_matrix, idx_selected = torch.min(
                dist_matrix, dim=2
            )  # wta_dist_matrix of shape [batch,Max_sources]
            if distance == "spherical" or distance == "spherical-squared":
                eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                mask = (
                    eucl_wta_dist_matrix <= filling_value / 2
                )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values.
            else:
                mask = (
                    wta_dist_matrix <= filling_value / 2
                )  # We create a mask of shape [batch,Max_sources] for only selecting the actives sources, i.e. those which were not filled with fake values.

            wta_dist_matrix = (
                wta_dist_matrix * mask
            )  # [batch,Max_sources], we select only the active sources.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device="cuda:0")[:, None].expand(
                -1, Max_sources
            )  # Shape (batch, Max_sources)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = (
                1  # Shape (batch,num_hyps)
            )

            ### Above lines are equivalent to
            # for batch_idx in range(batch) :
            #     for j in range(Max_sources) :
            #         gt_conf_stacked_t[batch_idx,idx_selected[batch_idx,j]] = 1  #Shape [batch,num_hyps], we set the confidences of the selected hypotheses.
            ###

            count_non_zeros = torch.sum(
                mask != 0
            )  # We count the number of actives sources for the computation of the mean (below).

            if count_non_zeros > 0:
                loss = (
                    torch.sum(wta_dist_matrix) / count_non_zeros
                )  # We compute the mean of the diff.

                selected_confidence_mask = (
                    gt_conf_stacked_t == 1
                )  # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.
                unselected_mask = (
                    ~selected_confidence_mask
                )  # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.

                if rejection_method == "uniform_negative":
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack(
                        [
                            torch.multinomial(
                                unselected_mask[b_idx].float(),
                                number_unconfident,
                                replacement=False,
                            )
                            for b_idx in range(batch)
                        ]
                    )  # (batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, number_unconfident
                    )  # (batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    # gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness.

                elif rejection_method == "wrong_confidence":
                    # We select the negative hypothesis with the highest confidence score predicted.

                    # Create a tensor with a given negative value for the already selected hypotheses (specified in selected_confidence_mask)
                    conf_pred_stacked_t_masked = (
                        conf_pred_stacked_t.clone().detach()
                    )  # (batch,num_hyps)
                    conf_pred_stacked_t_masked[selected_confidence_mask] = (
                        -1
                    )  # (batch,num_hyps)

                    # Get the indices of the highest confidence unselected hypotheses
                    _, ranking_unselected_confidence_pred = torch.sort(
                        conf_pred_stacked_t_masked, dim=-1, descending=True
                    )  # (batch,num_hyps)

                    # Get the top 'number_unconfident' indices
                    unconfident_indices = ranking_unselected_confidence_pred[
                        :, :number_unconfident
                    ]  # (batch,num_unconfident)

                    # Update the selected_confidence_mask using the unconfident_indices
                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, number_unconfident
                    )  # (batch,num_unconfident)

                    # assert selected_confidence_mask[batch_indices, unconfident_indices].all() == False, "The negative hypothesis should not be already selected."

                    selected_confidence_mask[batch_indices, unconfident_indices] = (
                        True  # (batch,num_hyps)
                    )

                elif (
                    rejection_method == "worst_hypothesis_selection"
                ):  # We select the worst hypothesis for each source as unconfident (negative) hypothesis.
                    if distance == "spherical" or distance == "spherical-squared":
                        mask_false_sources = dist_matrix_euclidean <= filling_value / 2
                    else:
                        mask_false_sources = dist_matrix <= filling_value / 2
                    dist_matrix = (
                        dist_matrix * mask_false_sources
                    )  # We set to 0 the distances of the fake sources (as the maximum errors will be computed).
                    _, worst_idx = torch.max(
                        dist_matrix, dim=2
                    )  # Worst hypothesis indexes of shape [batch,Max_sources]

                    # assert worst_idx.shape == (batch, Max_sources)

                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, Max_sources
                    )  # (batch,Max_sources)
                    unconfident_indices = worst_idx

                    # assert selected_confidence_mask[batch_indices, unconfident_indices].all() == False, "The negative hypothesis should not be already selected."

                    selected_confidence_mask[batch_indices, unconfident_indices] = (
                        True  # (batch,num_hyps)
                    )

                elif rejection_method == "all":

                    selected_confidence_mask = torch.ones_like(
                        selected_confidence_mask
                    ).bool()  # (batch,num_hyps)

                # assert conf_pred_stacked_t.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification.

                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                # if rejection_method in ['uniform_negative','wrong_confidence'] :
                #     assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."
                # elif rejection_method=='worst_hypothesis_selection': # In this case, the worst hypothesis can be the same for several sources.
                #     assert selected_confidence_mask.sum() <= batch*Max_sources+torch.sum(gt_conf_stacked_t==1) and selected_confidence_mask.sum() >= batch + torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(
                    conf_pred_stacked_t[selected_confidence_mask],
                    gt_conf_stacked_t[selected_confidence_mask],
                )

            else:
                loss = torch.tensor(0.0)
                confidence_loss = torch.tensor(0.0)

            sum_losses = torch.add(sum_losses, loss)
            sum_losses = torch.add(sum_losses, conf_weight * confidence_loss)

        elif mode == "wta-relaxed":

            # We compute the loss for the "best" hypothesis but also for the others with weight epsilon.

            wta_dist_matrix, idx_selected = torch.min(
                dist_matrix, dim=2
            )  # wta_dist_matrix of shape [batch,Max_sources], idx_selected of shape [batch,Max_sources].

            # assert wta_dist_matrix.shape == (batch,Max_sources)
            # assert idx_selected.shape == (batch,Max_sources)

            if distance == "spherical" or distance == "spherical-squared":
                eucl_wta_dist_matrix, _ = torch.min(dist_matrix_euclidean, dim=2)
                mask = eucl_wta_dist_matrix <= filling_value / 2
            else:
                mask = (
                    wta_dist_matrix <= filling_value / 2
                )  # We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = (
                wta_dist_matrix * mask
            )  # Shape [batch,Max_sources] ; we select only the active sources.
            count_non_zeros_1 = torch.sum(
                mask != 0
            )  # We count the number of actives sources as a sum over the batch for the computation of the mean (below).

            ### Confidence management
            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device="cuda:0")[:, None].expand(
                -1, Max_sources
            )  # Shape (batch, Max_sources)

            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = (
                1  # Shape (batch,num_hyps)
            )
            ###

            if count_non_zeros_1 > 0:
                loss0 = torch.multiply(
                    torch.sum(wta_dist_matrix) / count_non_zeros_1, 1 - epsilon
                )  # Scalar (average with coefficient)

                selected_confidence_mask = (
                    gt_conf_stacked_t == 1
                )  # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.
                unselected_mask = (
                    ~selected_confidence_mask
                )  # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.

                if rejection_method == "uniform_negative":
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack(
                        [
                            torch.multinomial(
                                unselected_mask[b_idx].float(),
                                number_unconfident,
                                replacement=False,
                            )
                            for b_idx in range(batch)
                        ]
                    )  # (batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, number_unconfident
                    )  # (batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = (
                        0  # (Useless) Line added for the sake of completness.
                    )

                elif rejection_method == "wrong_confidence":
                    # We select the negative hypothesis with the highest confidence score predicted.

                    # Create a tensor with a given negative value for the already selected hypotheses (specified in selected_confidence_mask)
                    conf_pred_stacked_t_masked = (
                        conf_pred_stacked_t.clone().detach()
                    )  # (batch,num_hyps)
                    conf_pred_stacked_t_masked[selected_confidence_mask] = (
                        -1
                    )  # (batch,num_hyps)

                    # Get the indices of the highest confidence unselected hypotheses
                    _, ranking_unselected_confidence_pred = torch.sort(
                        conf_pred_stacked_t_masked, dim=-1, descending=True
                    )  # (batch,num_hyps)

                    # Get the top 'number_unconfident' indices
                    unconfident_indices = ranking_unselected_confidence_pred[
                        :, :number_unconfident
                    ]  # (batch,num_unconfident)

                    # Update the selected_confidence_mask using the unconfident_indices
                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, number_unconfident
                    )  # (batch,num_unconfident)

                    # assert selected_confidence_mask[batch_indices, unconfident_indices].all() == False, "The negative hypothesis should not be already selected."

                    selected_confidence_mask[batch_indices, unconfident_indices] = (
                        True  # (batch,num_hyps)
                    )

                elif (
                    rejection_method == "worst_hypothesis_selection"
                ):  # We select the worst hypothesis for each source as unconfident (negative) hypothesis.

                    if distance == "spherical" or distance == "spherical-squared":
                        mask_false_sources = dist_matrix_euclidean <= filling_value / 2
                    else:
                        mask_false_sources = dist_matrix <= filling_value / 2
                    dist_matrix = (
                        dist_matrix * mask_false_sources
                    )  # We set to 0 the distances of the fake sources (as the maximum errors will be computed).
                    _, worst_idx = torch.max(
                        dist_matrix, dim=2
                    )  # Worst hypothesis indexes of shape [batch,Max_sources]

                    # assert worst_idx.shape == (batch, Max_sources)

                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, Max_sources
                    )  # (batch,Max_sources)
                    unconfident_indices = worst_idx

                    # assert selected_confidence_mask[batch_indices, unconfident_indices].all() == False, "The negative hypothesis should not be already selected."

                    selected_confidence_mask[batch_indices, unconfident_indices] = (
                        True  # (batch,num_hyps)
                    )

                elif rejection_method == "all":

                    selected_confidence_mask = torch.ones_like(
                        selected_confidence_mask
                    ).bool()  # (batch,num_hyps)

                # assert conf_pred_stacked_t.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification.

                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                # if rejection_method in ['uniform_negative','wrong_confidence'] :
                # assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."
                # elif rejection_method=='worst_hypothesis_selection': # In this case, the worst hypothesis can be the same for several sources.
                # assert selected_confidence_mask.sum() <= batch*Max_sources+torch.sum(gt_conf_stacked_t==1) and selected_confidence_mask.sum() >= batch + torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(
                    conf_pred_stacked_t[selected_confidence_mask],
                    gt_conf_stacked_t[selected_confidence_mask],
                )

            else:
                loss0 = torch.tensor(0.0)
                confidence_loss = torch.tensor(0.0)

            # We then the find the other hypothesis, and compute the epsilon weighted loss for them

            # At first, we remove hypothesis corresponding to "fake" ground-truth.
            if distance == "spherical" or distance == "spherical-squared":
                large_mask = dist_matrix_euclidean <= filling_value
            else:
                large_mask = (
                    dist_matrix <= filling_value
                )  # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = (
                dist_matrix * large_mask
            )  # Shape [batch,Max_sources,num_hypothesis].

            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(
                dist_matrix, dtype=bool
            )  # Shape [batch,Max_sources,num_hypothesis]
            mask_selected.scatter_(
                2, idx_selected.unsqueeze(-1), 1
            )  # idx_selected new shape: [batch,Max_sources,1].
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed.
            # Shape of mask_selected: [batch,Max_sources,num_hypothesis]

            # assert mask_selected.shape == (batch,Max_sources,num_hyps)

            mask_selected = (
                ~mask_selected
            )  # Shape [batch,Max_sources,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = (
                dist_matrix * mask_selected
            )  # Shape [batch,Max_sources,num_hypothesis]

            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix != 0)

            if count_non_zeros_2 > 0:
                epsilon_loss = torch.multiply(
                    torch.sum(dist_matrix) / count_non_zeros_2, epsilon
                )  # Scalar for each hyp
            else:
                epsilon_loss = torch.tensor(0.0)

            sum_losses = torch.add(
                sum_losses, epsilon_loss
            )  # Loss for the unselected (i.e., not winners) hypothesis (epsilon weighted)
            sum_losses = torch.add(
                sum_losses, loss0
            )  # Loss for the selected (i.e., the winners) hypothesis (1-epsilon weighted)
            sum_losses = torch.add(
                sum_losses, conf_weight * confidence_loss
            )  # Loss for the confidence prediction.

        elif mode == "wta-top-n" and top_n > 1:

            # dist_matrix.shape == (batch,Max_sources,num_hyps)
            # wta_dist_matrix of shape [batch,Max_sources]

            dist_matrix = torch.multiply(
                dist_matrix, -1
            )  # Shape (batch,Max_sources,num_hyps)
            top_k, indices = torch.topk(
                input=dist_matrix, k=top_n, dim=-1
            )  # top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n)
            dist_matrix_min = torch.multiply(top_k, -1)

            if distance == "spherical" or distance == "spherical-squared":
                dist_matrix_euclidean = torch.multiply(
                    dist_matrix_euclidean, -1
                )  # Shape (batch,Max_sources,num_hyps)
                top_k, _ = torch.topk(
                    input=dist_matrix_euclidean, k=top_n, dim=-1
                )  # top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n)
                dist_matrix_min_euclidean = torch.multiply(top_k, -1)
                mask = dist_matrix_min_euclidean <= filling_value / 2
            else:
                mask = (
                    dist_matrix_min <= filling_value / 2
                )  # We create a mask of shape [batch,Max_sources,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values.
            # assert mask[:,:,0].all() == mask[:,:,-1].all() # This mask related should be constant in the third dimension.

            dist_matrix_min = (
                dist_matrix_min * mask
            )  # [batch,Max_sources,top_n], we select only the active sources.
            # assert dist_matrix_min.shape == (batch,Max_sources,top_n)

            count_non_zeros = torch.sum(
                mask[:, :, 0] != 0
            )  # We count the number of entries (in the first two dimensions) for which the mask is different from zero.

            ### Confidence management
            # Create tensors to index batch and Max_sources and top-n dimensions.
            batch_indices = torch.arange(batch)[:, None, None].expand(
                -1, Max_sources, top_n
            )  # Shape (batch, Max_sources,top_n)
            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], indices[mask]] = (
                1  # Shape (batch,num_hyps)
            )
            ###

            #####
            selected_confidence_mask = (
                gt_conf_stacked_t == 1
            )  # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores
            # to be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.

            for i in range(top_n):

                # assert count_non_zeros == torch.sum(mask[:,:,i]!=0) # We count the number of entries for which the mask is different from zero.

                if count_non_zeros > 0:
                    loss = torch.multiply(
                        torch.sum(dist_matrix_min[:, :, i]) / count_non_zeros, 1.0
                    )

                else:
                    loss = torch.tensor(0.0)

                sum_losses = torch.add(sum_losses, loss / top_n)

            if count_non_zeros > 0:

                unselected_mask = (
                    ~selected_confidence_mask
                )  # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.

                if rejection_method == "uniform_negative":
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack(
                        [
                            torch.multinomial(
                                unselected_mask[b_idx].float(),
                                number_unconfident,
                                replacement=False,
                            )
                            for b_idx in range(batch)
                        ]
                    )  # (batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, number_unconfident
                    )  # (batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = (
                        0  # (Useless) Line added for the sake of completness.
                    )

                elif rejection_method == "wrong_confidence":
                    # We select the negative hypothesis with the highest confidence score predicted.

                    # Create a tensor with a given negative value for the already selected hypotheses (specified in selected_confidence_mask)
                    conf_pred_stacked_t_masked = (
                        conf_pred_stacked_t.clone().detach()
                    )  # (batch,num_hyps)
                    conf_pred_stacked_t_masked[selected_confidence_mask] = (
                        -1
                    )  # (batch,num_hyps)

                    # Get the indices of the highest confidence unselected hypotheses
                    _, ranking_unselected_confidence_pred = torch.sort(
                        conf_pred_stacked_t_masked, dim=-1, descending=True
                    )  # (batch,num_hyps)

                    # Get the top 'number_unconfident' indices
                    unconfident_indices = ranking_unselected_confidence_pred[
                        :, :number_unconfident
                    ]  # (batch,num_unconfident)

                    # Update the selected_confidence_mask using the unconfident_indices
                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, number_unconfident
                    )  # (batch,num_unconfident)

                    # assert selected_confidence_mask[batch_indices, unconfident_indices].all() == False, "The negative hypothesis should not be already selected."

                    selected_confidence_mask[batch_indices, unconfident_indices] = (
                        True  # (batch,num_hyps)
                    )

                elif (
                    rejection_method == "worst_hypothesis_selection"
                ):  # We select the worst hypothesis for each source as unconfident (negative) hypothesis.

                    if distance == "spherical" or distance == "spherical-squared":
                        mask_false_sources = dist_matrix_euclidean <= filling_value / 2
                    else:
                        mask_false_sources = dist_matrix <= filling_value / 2
                    dist_matrix = (
                        dist_matrix * mask_false_sources
                    )  # We set to 0 the distances of the fake sources (as the maximum errors will be computed).
                    _, worst_idx = torch.max(
                        dist_matrix, dim=2
                    )  # Worst hypothesis indexes of shape [batch,Max_sources]

                    # assert worst_idx.shape == (batch, Max_sources)

                    batch_indices = torch.arange(batch)[:, None].expand(
                        -1, Max_sources
                    )  # (batch,Max_sources)
                    unconfident_indices = worst_idx

                    # assert selected_confidence_mask[batch_indices, unconfident_indices].all() == False, "The negative hypothesis should not be already selected."

                    selected_confidence_mask[batch_indices, unconfident_indices] = (
                        True  # (batch,num_hyps)
                    )

                elif rejection_method == "all":

                    selected_confidence_mask = torch.ones_like(
                        selected_confidence_mask
                    ).bool()  # (batch,num_hyps)

                # assert conf_pred_stacked_t.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification.

                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                # if rejection_method in ['uniform_negative','wrong_confidence'] :
                # assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."
                # elif rejection_method=='worst_hypothesis_selection': # In this case, the worst hypothesis can be the same for several sources.
                # assert selected_confidence_mask.sum() <= batch*Max_sources+torch.sum(gt_conf_stacked_t==1) and selected_confidence_mask.sum() >= batch + torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(
                    conf_pred_stacked_t[selected_confidence_mask],
                    gt_conf_stacked_t[selected_confidence_mask],
                )

            else:
                confidence_loss = torch.tensor(0.0)

            sum_losses = torch.add(sum_losses, conf_weight * confidence_loss)

        return sum_losses, conf_weight * confidence_loss


### Negative log-likelihood based (Mixture density networks) loss here


class VonMisesNLLLoss(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        num_modes=1,
        log_kappa_pred=False,
    ) -> None:
        super(VonMisesNLLLoss, self).__init__(size_average, reduce, reduction)
        self.num_modes = num_modes
        self.log_kappa_pred = log_kappa_pred

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for the Negative Log-Likelihood Loss.

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxTxself.num_hypothesisx2]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,T,Max_sources],[batch,T,Max_sources,2]

        Returns:
            loss (torch.tensor)
            meta_data (dict)
        """
        mu_pred_stacked, kappa_pred_stacked, pi_pred_stacked = predictions
        # Shape [batchxTxself.num_modesx2], [batchxTxself.num_modesx1], [batchxTxself.num_modesx1]

        # assert predictions[0].shape[2] == self.num_modes

        if len(targets) == 3:
            source_activity_target, direction_of_arrival_target, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        elif len(targets) == 4:
            source_activity_target, direction_of_arrival_target, _, _ = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )
        else:
            source_activity_target, direction_of_arrival_target = (
                targets  # Shape [batch,T,Max_sources],[batch,T,Max_sources,2]
            )

        T = source_activity_target.shape[1]

        losses = torch.tensor(0.0, device=mu_pred_stacked.device)

        for t in range(T):

            source_activity_target_t = source_activity_target[:, t, :].detach()
            direction_of_arrival_target_t = direction_of_arrival_target[
                :, t, :, :
            ].detach()
            mu_pred_stacked_t = mu_pred_stacked[:, t, :, :]
            kappa_pred_stacked_t = kappa_pred_stacked[:, t, :, :]
            pi_pred_stacked_t = pi_pred_stacked[:, t, :, :]

            loss_t = self.nll_loss(
                mu_pred_stacked_t=mu_pred_stacked_t,
                kappa_pred_stacked_t=kappa_pred_stacked_t,
                pi_pred_stacked_t=pi_pred_stacked_t,
                source_activity_target_t=source_activity_target_t,
                direction_of_arrival_target_t=direction_of_arrival_target_t,
                log_kappa_pred=self.log_kappa_pred,
            )

            losses = torch.add(losses, loss_t)

        losses = losses / T

        return losses, {}

    def compute_spherical_distance(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
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

    def nll_loss(
        self,
        mu_pred_stacked_t,
        kappa_pred_stacked_t,
        pi_pred_stacked_t,
        source_activity_target_t,
        direction_of_arrival_target_t,
        log_kappa_pred=False,
    ):
        batch, num_modes, _ = mu_pred_stacked_t.shape
        max_sources = source_activity_target_t.shape[1]

        # Log probabilities of pi
        log_pi_pred_stacked_t = torch.log(pi_pred_stacked_t)  # [batch,num_modes,1]
        log_pi_pred_stacked_t = log_pi_pred_stacked_t.permute(
            0, 2, 1
        )  # [batch,1,num_modes]

        # Expand targets and predictions to match each other's shapes for batch-wise computation
        expanded_direction_of_arrival_target_t = (
            direction_of_arrival_target_t.unsqueeze(2).expand(-1, -1, num_modes, -1)
        )  # [batch,max_sources,num_modes,2]
        expanded_mu_pred_stacked_t = mu_pred_stacked_t.unsqueeze(1).expand(
            -1, max_sources, -1, -1
        )  # [batch,max_sources,num_modes,2]
        expanded_kappa_pred_stacked_t = kappa_pred_stacked_t.unsqueeze(1).expand(
            -1, max_sources, -1, -1
        )  # [batch,max_sources,num_modes,1]

        # Compute log normal kernel for all combinations of sources and modes
        log_vonmises_values = self.log_vonmises_kernel(
            expanded_direction_of_arrival_target_t,
            expanded_mu_pred_stacked_t,
            expanded_kappa_pred_stacked_t,
            log_kappa_pred,
        )

        # print(log_vonmises_values.shape)
        log_vonmises_values = log_vonmises_values.reshape(batch, max_sources, num_modes)
        # log_vonmises_values of shape [batch, max_sources, num_modes]

        # Include the log probabilities of pi and compute the log-sum-exp
        total_log_prob = (
            log_vonmises_values + log_pi_pred_stacked_t
        )  # [batch,max_sources,num_modes]
        log_sum_exp = torch.logsumexp(total_log_prob, dim=2).unsqueeze(
            -1
        )  # [batch,max_sources,1]

        # Mask out inactive sources
        active_source_mask = source_activity_target_t.unsqueeze(
            -1
        )  # [batch,max_sources,1]
        masked_log_sum_exp = log_sum_exp * active_source_mask
        active_source_mask_sum = active_source_mask.sum(dim=1).expand(
            -1, max_sources
        )  # [batch,max_sources]

        mask = active_source_mask_sum > 0
        masked_log_sum_exp = masked_log_sum_exp.squeeze(-1)

        masked_log_sum_exp[mask] = (
            masked_log_sum_exp[mask] / active_source_mask_sum[mask]
        )  # [batch,max_sources]

        # Compute mean negative log likelihood
        NLL = -torch.sum(masked_log_sum_exp) / batch

        return NLL

    def log_vonmises_kernel(self, y, mu_pred, kappa_pred, log_kappa_pred=False):
        # Assuming compute_spherical_distance is another method of your class
        # y batch,3,3,2
        # mu_pred batch,3,3,2
        # kappa_pred
        # batch, Max_sources, num_modes, _ = y.shape

        mu_pred = mu_pred.reshape(-1, 2)  # Shape [batch*num_modes*max_sources,2]
        kappa_pred = kappa_pred.reshape(-1, 1)  # Shape [batch*num_modes*max_sources,1]
        kappa_pred = kappa_pred.squeeze(-1)
        y = y.reshape(-1, 2)  # Shape [batch*num_modes*max_sources,2]

        # Convert to cartesian coordinates
        mu_pred = self.spherical_to_cartesian(
            mu_pred[:, 0], mu_pred[:, 1]
        )  # Shape [batch*num_modes*max_sources,3]
        y = self.spherical_to_cartesian(
            y[:, 0], y[:, 1]
        )  # Shape [batch*num_modes*max_sources,3]

        if log_kappa_pred == True:
            return (
                kappa_pred
                - torch.log(torch.tensor(4 * np.pi))
                - logsinh_torch(torch.exp(kappa_pred))
                + torch.exp(kappa_pred) * (y * mu_pred).sum(dim=1)
            )

        else:
            return (
                torch.log(kappa_pred)
                - torch.log(4 * np.pi)
                - logsinh_torch(kappa_pred)
                + kappa_pred * (y * mu_pred).sum(dim=1)
            )

    def spherical_to_cartesian(self, azimuth, elevation):
        x = torch.cos(elevation) * torch.cos(azimuth)
        y = torch.cos(elevation) * torch.sin(azimuth)
        z = torch.sin(elevation)
        return torch.stack((x, y, z), dim=-1)

    def von_mises_fisher_kernel(self, vector_x, mu, kappa):
        # Convert inputs to tensors, if they aren't already
        vector_x = torch.tensor(vector_x, dtype=torch.float32)
        mu = torch.tensor(mu, dtype=torch.float32)
        kappa = torch.tensor(kappa, dtype=torch.float32)

        # Convert to Cartesian coordinates
        vector_x_cartesian = self.spherical_to_cartesian(vector_x[:, 0], vector_x[:, 1])
        mu_cartesian = self.spherical_to_cartesian(mu[:, 0], mu[:, 1])

        # Ensure kappa is a column vector for broadcasting
        kappa = kappa.view(-1, 1)

        # Normalization constant C_3(kappa)
        C_3_kappa = kappa / (4 * torch.pi * torch.sinh(kappa))

        # Dot product of vector_x and mu in Cartesian coordinates
        dot_product = (mu_cartesian * vector_x_cartesian).sum(dim=1, keepdim=True)

        # Compute the exponential term
        exp_term = torch.exp(kappa * dot_product)

        # Compute the probability density
        output = C_3_kappa * exp_term

        return output.squeeze()  # Removing the extra dimension
