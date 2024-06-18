#!/bin/bash

MY_HOME="/home/victorletzelter/workspace/VoronoiWTA/toy"
cd ${MY_HOME}/src

# Define the experiments
experiment=("plot_figure1")

# Define the hypothesis counts and corresponding sizes for Histogram model
hyp="20 5,4"

IFS=' ' read -r num_hypotheses sizes <<< "$hyp"

# Rmcl model
python train.py ckpt_path=${MY_HOME}/checkpoints/figure1/ckpt_fig1.ckpt data=synthetic_data.yaml model=WTABased.yaml experiment=${experiment}.yaml hydra.job.name=data_${experiment}_model_Rmcl_${num_hypotheses}_hyps test=True model.num_hypothesis=${num_hypotheses} trainer.max_epochs=40 model.hparams.plot_mode=True
