#!/bin/bash

export CSV_DIR="${HOME}/saved_csv"

mkdir "${HOME}/plots"

python ${HOME}/scripts/plot_results.py --csv_dir=${CSV_DIR} --save_dir=${HOME}/plots