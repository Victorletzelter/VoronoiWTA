#!/bin/bash

cd ${MY_HOME}/src

# TRAININGS

# Define the experiments
declare -a experiments=("mixtures_uni_to_gaussians" "gauss_not_centered" "changing_damier" "rotating_moons")
# Define the hypothesis counts and corresponding sizes for Histogram model
declare -a hyps=("9 3,3" "16 4,4" "20 5,4" "25 5,5" "49 7,7" "100 10,10")
# Define the seeds
declare -a seeds=(1234,1244,1245)

# Loop through each seed
for seed in "${seeds[@]}"; do
    # Loop through each experiment
    for experiment in "${experiments[@]}"; do
        # Loop through each hypothesis count and size pair
        for hyp in "${hyps[@]}"; do
            # Split hyp into number of hypotheses and histogram sizes
            IFS=' ' read -r num_hypotheses sizes <<< "$hyp"

            # Rmcl model
            python train.py data=synthetic_data.yaml model=WTABased.yaml experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_Rmcl_${num_hypotheses}_hyps test=False model.num_hypothesis=${num_hypotheses} trainer.max_epochs=101 model.square_size=1
            
            # Histogram model
            python train.py data=synthetic_data.yaml model=Histogram.yaml experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_Histogram_${num_hypotheses}_hyps test=False model.sizes=[${sizes}] trainer.max_epochs=101 model.square_size=1
            
            # GaussMix model
            python train.py data=synthetic_data.yaml model=GaussMix.yaml experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_GaussMix_${num_hypotheses}_hyps test=False model.num_hypothesis=${num_hypotheses} trainer.max_epochs=101
        done
    done
done