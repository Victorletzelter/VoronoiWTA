#!/bin/bash

declare -a seeds=(1234,1244,1245)

# Loop through each seed
for seed in "${seeds[@]}"; do

    cd ${MY_HOME}/scripts
    mkdir seed_${seed}
    cd seed_${seed}/..


    # Extract checkpoints path

    python extract_ckpt.py --base_dir=${MY_HOME}/logs/train/seed_${seed}_mixtureunigauss_2000 --save_dir=${MY_HOME}/scripts/seed_${seed}
    python extract_ckpt.py --base_dir=${MY_HOME}/logs/train/seed_${seed}_changingdamier_2000 --save_dir=${MY_HOME}/scripts/seed_${seed}
    python extract_ckpt.py --base_dir=${MY_HOME}/logs/train/seed_${seed}_gaussnotcentered_2000 --save_dir=${MY_HOME}/scripts/seed_${seed}
    python extract_ckpt.py --base_dir=${MY_HOME}/logs/train/seed_${seed}_rotatingmoons_2000 --save_dir=${MY_HOME}/scripts/seed_${seed}
done