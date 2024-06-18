#!/bin/bash

# Check if the dir saved_csv exists and create it if not
if [ ! -d "${MY_HOME}/saved_csv" ]; then
    mkdir ${MY_HOME}/saved_csv
fi

SAVE_PATH=${MY_HOME}/saved_csv
MLFLOW_DIR=${MY_HOME}/logs/mlflow

# We check if the mlflow dir exists
if [ ! -d "${MLFLOW_DIR}" ]; then
    echo "The mlflow directory does not exist. Please run the training script before running this script."
    exit 1
fi

python ${MY_HOME}/scripts/scripts_download_csv.py --experiment_name=audio_experiments --save_dir=${SAVE_PATH} --mlflow_dir=${MLFLOW_DIR}