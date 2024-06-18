from argparse import Namespace
from .modules import AbstractLocalizationModule, FeatureExtraction, MHLocalizationOutput
import torch
import torch.nn as nn
from typing import Tuple
from src.utils.losses import MHSELLoss

from src.models.density_estimation import VoronoiDE

class MHSELDNet(AbstractLocalizationModule, VoronoiDE):
    def __init__(self,
                 dataset_path: str,
                 cv_fold_idx: int,
                 hparams: Namespace) -> None:
        
        AbstractLocalizationModule.__init__(self, dataset_path, cv_fold_idx, hparams)
        VoronoiDE.__init__(self, 
            kernel_type=hparams['kernel_type'],
            scaling_factor=hparams['scaling_factor'],
            kde_mode=hparams['kde_mode'],
            kde_weighted=hparams['kde_weighted'],
        )

        self.kernel_type = hparams['kernel_type']
        self.scaling_factor = hparams['scaling_factor']
        self.kde_mode = hparams['kde_mode']
        self.kde_weighted = hparams['kde_weighted']

        self.closed_form_vol = hparams['closed_form_vol']
        self.hit_and_run_sampling = hparams['hit_and_run_sampling']
        self.n_directions = hparams['n_directions']
        self.square_size = hparams['square_size']

        self.loss_function = self.get_loss_function()

        num_steps_per_chunk = int(2 * hparams['chunk_length'] / hparams['frame_length'])
        self.feature_extraction = FeatureExtraction(num_steps_per_chunk,
                                                    hparams['num_fft_bins'],
                                                    dropout_rate=hparams['dropout_rate'])

        feature_dim = int(hparams['num_fft_bins'] / 4) # See the FeatureExtraction module for the justification of this 
        # value for the feature_dim. 
        
        self.gru = nn.GRU(feature_dim, hparams['hidden_dim'], num_layers=4, batch_first=True, bidirectional=True)

        self.localization_output = MHLocalizationOutput(input_dim =2 * hparams['hidden_dim'], 
                                                        num_hypothesis = hparams['num_hypothesis'])
        # In the localization module, the input_dim is to 2 * hparams.hidden_dim if bidirectional=True in the GRU.

    def get_loss_function(self) -> nn.Module:
        return MHSELLoss(self.hparams['max_num_sources'], alpha=self.hparams['alpha'], mode=self.hparams['mode'], 
                         top_n = self.hparams['top_n'],distance = self.hparams['distance'],epsilon=self.hparams['epsilon'],single_target_loss=self.hparams['single_target_loss'])

    def forward(self,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        extracted_features = self.feature_extraction(audio_features) # extracted_features of shape
        #[batch,T,B/4] where batch is the batch size, T is the number of time steps per chunk, and B is the number of FFT bins.

        output, _ = self.gru(extracted_features) # output of shape [batch,T,hparams['hidden_dim']]

        MHdirection_of_arrival_output = self.localization_output(output) # output of shape [batch,T,num_hypothesis,2]
        meta_data = {}

        return MHdirection_of_arrival_output, meta_data

    def prepare_predictions_emd(self, predictions) :

        if 'kernel_mode_emd' in self.hparams.keys() and self.hparams['kernel_mode_emd'] is True : 

            # raise not implemented error 
            raise NotImplementedError, "The kernel_mode_emd is not implemented yet for the MH class."

        else : 
            predictions_emd = predictions

        return predictions_emd
    
    def prepare_predictions_oracle(self, predictions) :

        return predictions

    def prepare_predictions_nll(self, predictions):

        return predictions
