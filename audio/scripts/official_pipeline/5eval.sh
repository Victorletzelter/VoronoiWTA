#!/bin/bash

cd ${MY_HOME}/src

# Define the hypothesis counts and corresponding sizes for Histogram model
declare -a hyps=("9 3,3" "16 4,4" "20 5,4" "25 5,5")
declare -a scaling_factors=(0.05 0.1 0.15 0.2 0.3 0.5 1.0) 
seed=1234

JSON_PATH=${MY_HOME}/checkpoints/checkpoints.json

# Loop through each hypothesis count and size pair
for hyp in "${hyps[@]}"; do
    # Split hyp into number of hypotheses and histogram sizes
    IFS=' ' read -r num_hypotheses sizes <<< "$hyp"

    nrows=$(echo $sizes | cut -d',' -f1)
    ncols=$(echo $sizes | cut -d',' -f2)

    # Extract the checkpoints path
    CKPT_HISTOGRAM=$(jq -r --arg num_hyp "$num_hypotheses" '.["histogram-\($num_hyp)-hyps"]' "$JSON_PATH")
    CKPT_WTA=$(jq -r --arg num_hyp "$num_hypotheses" '.["score_based_wta-\($num_hyp)-hyps"]' "$JSON_PATH")
    CKPT_MDN=$(jq -r --arg num_hyp "$num_hypotheses" '.["von_mises_fisher_mdn-\($num_hyp)-hyps"]' "$JSON_PATH")

    for scaling_factor in "${scaling_factors[@]}"; do
        # Voronoi-WTA model       
        python eval.py data=ansim.yaml model=WTABased.yaml experiment=ansim/full-training/eval/scoring-ansim-noisy-classes-online_NLL.yaml hydra.job.name=kernel_orig-rmcl-100-epoch-${num_hypotheses}-hyps-${scaling_factor} model.hparams.num_hypothesis=${num_hypotheses} model.hparams.mode='wta' ckpt_path=??? model.hparams.kernel_mode_emd=False model.hparams.compute_nll=True trainer.limit_test_batches=1.0 logger.mlflow.experiment_name=audio_experiments model.hparams.scaling_factor=${scaling_factor}
        
        # Kernel-WTA model
        python eval.py data=ansim.yaml model=WTABased.yaml experiment=ansim/full-training/eval/scoring-ansim-noisy-classes-online_NLL.yaml hydra.job.name=kernel_KDE-orig-rmcl-100-epoch-${num_hypotheses}-hyps-${scaling_factor} model.hparams.num_hypothesis=${num_hypotheses} model.hparams.mode='wta' ckpt_path=??? model.hparams.kernel_mode_emd=False model.hparams.compute_nll=True trainer.limit_test_batches=1.0 logger.mlflow.experiment_name=audio_experiments model.hparams.kde_mode=True model.hparams.kde_weighted=True model.hparams.scaling_factor=${scaling_factor}

        # Histogram model
        python eval.py data=ansim.yaml model=Histogram.yaml experiment=ansim/full-training/eval/histogram-ansim-noisy-classes-online_NLL.yaml hydra.job.name=kernel_100-epoch-histogram-gen-${nrows}-${ncols}-${scaling_factor} model.hparams.nrows=${nrows} model.hparams.ncols=${ncols} ckpt_path=??? model.hparams.kernel_mode_emd=False model.hparams.compute_nll=True trainer.limit_test_batches=1.0 logger.mlflow.experiment_name=audio_experiments model.hparams.kernel_type=von_mises_fisher model.hparams.scaling_factor=${scaling_factor}
    done

    # VoronoiWTA model with uniform kernel
    python eval.py data=ansim.yaml model=WTABased.yaml experiment=ansim/full-training/eval/scoring-ansim-noisy-classes-online_NLL.yaml hydra.job.name=kernel_orig-rmcl-100-epoch-${num_hypotheses}-hyps-uniform model.hparams.num_hypothesis=${num_hypotheses} model.hparams.mode='wta' ckpt_path=??? model.hparams.kernel_mode_emd=False model.hparams.compute_nll=True trainer.limit_test_batches=1.0 logger.mlflow.experiment_name=audio_experiments model.hparams.scaling_factor=None model.hparams.kernel_type=uniform
    
    # Histogram model with uniform kernel
    python eval.py data=ansim.yaml model=Histogram.yaml experiment=ansim/full-training/eval/histogram-ansim-noisy-classes-online_NLL.yaml hydra.job.name=kernel_100-epoch-histogram-gen-${nrows}-${ncols} model.hparams.nrows=${nrows} model.hparams.ncols=${ncols} ckpt_path=??? model.hparams.kernel_mode_emd=False model.hparams.compute_nll=True trainer.limit_test_batches=1.0 logger.mlflow.experiment_name=audio_experiments model.hparams.kernel_type=uniform
    
    # Mixture on Von Mises Fisher
    python eval.py data=ansim.yaml model=Mixture_NLLVonMises.yaml experiment=ansim/full-training/nll-vonmises-ansim-noisy-classes-online.yaml hydra.job.name=vonmises-${num_hypotheses}modes-seed${seed}-logkappapred model.hparams.compute_nll=True model.hparams.num_modes=${num_hypotheses} seed=${seed} model.hparams.log_kappa_pred=True ckpt_path=??? logger.mlflow.experiment_name=audio_experiments
done