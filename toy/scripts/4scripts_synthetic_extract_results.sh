#!/bin/bash

# Go in the results directory
cd ${MY_HOME}/results

declare -a seeds=(1234,1244,1245)

for seed in "${seeds[@]}"; do

    # Delete the saved_csv directory if it exists
    if [ -d "saved_csv_seed${seed}" ]; then
        rm -r saved_csv_seed${seed}
    fi

    mkdir saved_csv_seed${seed}

    cd ${MY_HOME}/scripts

    python scripts_download_csv.py --experiment_name=mixtures_uni_to_gaussians_eval_ok_2000_seed_${seed} --save_dir=${MY_HOME}/results/saved_csv_seed${seed}
    python scripts_download_csv.py --experiment_name=changing_damier_eval_ok_2000_seed_${seed} --save_dir=${MY_HOME}/results/saved_csv_seed${seed}
    python scripts_download_csv.py --experiment_name=gauss_not_centered_eval_ok_2000_seed_${seed} --save_dir=${MY_HOME}/results/saved_csv_seed${seed}
    python scripts_download_csv.py --experiment_name=rotating_moons_eval_ok_2000_seed_${seed} --save_dir=${MY_HOME}/results/saved_csv_seed${seed}
done