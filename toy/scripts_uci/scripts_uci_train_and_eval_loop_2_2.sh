#!/bin/bash
declare -a dataset_names=("naval" "power_plant" "wine" "yacht") # Add your dataset names here
declare -a hyp_counts=(5)
declare -a split_nums=(10 11 12 13 14 15 16 17 18 19) # Add your split numbers here
max_epochs="1000"
lr="0.01"

cd ${MY_HOME}/src

# Loop through the array of dataset names
for dataset_name in "${dataset_names[@]}"
do
    # Loop through the array of hypothesis counts
    for hyp_count in "${hyp_counts[@]}"
    do
        # Loop through the array of split numbers
        for split_num in "${split_nums[@]}"
        do
            # Add cluster jobs for each type of model with different datasets, hypotheses, and splits
            python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml model=WTABased.yaml model.num_hypothesis=${hyp_count} model.square_size=10 hydra.job.name=vwta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr}
	        python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml model=WTABased.yaml model.num_hypothesis=${hyp_count} model.square_size=10 hydra.job.name=kwta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.kde_mode_nll=True
            python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml model=Histogram.yaml model.sizes=[${hyp_count}] model.square_size=10 hydra.job.name=hist-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr}
            python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml model=GaussMix.yaml model.num_hypothesis=${hyp_count} hydra.job.name=mixture-density-networks-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr}
        done
    done
done