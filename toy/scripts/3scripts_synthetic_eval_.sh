#!/bin/bash

# EVALUATIONS
# Define the experiments

cd ${MY_HOME}/src

# Define the hypothesis counts and corresponding sizes for Histogram model
declare -a hyps=("9 3,3" "16 4,4" "20 5,4" "25 5,5" "49 7,7" "100 10,10")
declare -a scaling_factors=(0.05 0.1 0.15 0.2 0.3 0.5 1.0)
declare -a experiments=("mixtures_uni_to_gaussians" "gauss_not_centered" "changing_damier" "rotating_moons")
declare -A dataset_exp_map

declare -a seeds=(1234,1244,1245)

dataset_exp_map["mixtures_uni_to_gaussians"]="mixtureunigauss"
dataset_exp_map["gauss_not_centered"]="gaussnotcentered"
dataset_exp_map["changing_damier"]="changingdamier"
dataset_exp_map["rotating_moons"]="rotatingmoons"

# Loop through each seed
for seed in "${seeds[@]}"; do

    # Loop through each experiment
    for experiment in "${experiments[@]}"; do

        dataset_name="${dataset_exp_map[$experiment]}"
        echo $dataset_name
        JSON_PATH="${MY_HOME}/scripts/seed_${seed}/checkpoints_seed_${seed}_${dataset_name}_2000.json"

        # Loop through each hypothesis count and size pair
        for hyp in "${hyps[@]}"; do
            # Split hyp into number of hypotheses and histogram sizes
            IFS=' ' read -r num_hypotheses sizes <<< "$hyp"

            # Extract the checkpoints path
            CKPT_VORONOI_WTA=$(jq -r ".seed_${seed}_${dataset_name}_2000_Rmcl_${num_hypotheses}_hyp" "$JSON_PATH")
            CKPT_HISTOGRAM=$(jq -r ".seed_${seed}_${dataset_name}_2000_Histogram_${num_hypotheses}_hyp" "$JSON_PATH")
            CKPT_MDN=$(jq -r ".seed_${seed}_${dataset_name}_2000_GaussMix_${num_hypotheses}_hyp" "$JSON_PATH")

            for scaling_factor in "${scaling_factors[@]}"; do
                # Voronoi-WTA model
                python eval.py seed=${seed} data=synthetic_data.yaml model=WTABased.yaml ckpt_path=${CKPT_VORONOI_WTA} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_VoronoiWTA_${num_hypotheses}_hyps_h_${scaling_factor}_seed_${seed} model.num_hypothesis=${num_hypotheses} trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.kernel_type=gauss_normalized model.scaling_factor=${scaling_factor} model.hparams.kde_mode_nll=False model.hparams.kde_weighted_nll=False model.hparams.kernel_mode_emd=True 
                # Kernel-WTA model
                python eval.py seed=${seed} data=synthetic_data.yaml model=WTABased.yaml ckpt_path=${CKPT_VORONOI_WTA} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_KernelWTA_${num_hypotheses}_hyps_h_${scaling_factor}_seed_${seed} model.num_hypothesis=${num_hypotheses} trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.kernel_type=gauss_normalized model.scaling_factor=${scaling_factor} model.hparams.kde_mode_nll=True model.hparams.kde_weighted_nll=True model.hparams.kernel_mode_emd=True
                # Unweighted Kernel-WTA model
                python eval.py seed=${seed} data=synthetic_data.yaml model=WTABased.yaml ckpt_path=${CKPT_VORONOI_WTA} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_UnweightedKernelWTA_${num_hypotheses}_hyps_h_${scaling_factor}_seed_${seed} model.num_hypothesis=${num_hypotheses} trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.kernel_type=gauss_normalized model.scaling_factor=${scaling_factor} model.hparams.kde_mode_nll=True model.hparams.kde_weighted_nll=False model.hparams.kernel_mode_emd=True 
                # Histogram model
                python eval.py seed=${seed} data=synthetic_data.yaml model=Histogram.yaml ckpt_path=${CKPT_HISTOGRAM} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_Histogram_${num_hypotheses}_hyps_h_${scaling_factor}_seed_${seed} model.sizes=[${sizes}] trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.kernel_type=gauss_normalized model.scaling_factor=${scaling_factor} model.hparams.kde_mode_nll=False model.hparams.kde_weighted_nll=False model.hparams.kernel_mode_emd=True
            done

            # Dirac-WTA model (EMD and Quantization Error)
            python eval.py seed=${seed} data=synthetic_data.yaml model=WTABased.yaml ckpt_path=${CKPT_VORONOI_WTA} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_DiracWTA_${num_hypotheses}_hyps_dirac_seed_${seed} model.num_hypothesis=${num_hypotheses} trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=False model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.hparams.kde_mode_nll=False model.hparams.kde_weighted_nll=False model.hparams.kernel_mode_emd=False model.scaling_factor=0.0
            # VoronoiWTA model with uniform kernel
            python eval.py seed=${seed} data=synthetic_data.yaml model=WTABased.yaml ckpt_path=${CKPT_VORONOI_WTA} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_VoronoiWTA_${num_hypotheses}_hyps_uniform_seed_${seed} model.num_hypothesis=${num_hypotheses} trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.kernel_type=uniform model.hparams.kde_mode_nll=False model.hparams.kde_weighted_nll=False model.hparams.kernel_mode_emd=True
            # Histogram model with uniform kernel
            python eval.py seed=${seed} data=synthetic_data.yaml model=Histogram.yaml ckpt_path=${CKPT_HISTOGRAM} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_Histogram_${num_hypotheses}_hyps_uniform_seed_${seed} model.sizes=[${sizes}] trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.kernel_type=uniform model.hparams.kde_mode_nll=False model.hparams.kde_weighted_nll=False model.hparams.kernel_mode_emd=True
            # GaussMix model
            python eval.py seed=${seed} data=synthetic_data.yaml model=GaussMix.yaml ckpt_path=${CKPT_MDN} experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_GaussMix_${num_hypotheses}_hyps_seed_${seed} model.num_hypothesis=${num_hypotheses} trainer.limit_test_batches=2000 logger.mlflow.experiment_name=${experiment}_eval_ok_2000_seed_${seed} model.hparams.compute_nll=True model.hparams.compute_emd=True data.batch_size=1 model.hparams.compute_risk=True model.hparams.kernel_mode_emd=False
        done
    done
done