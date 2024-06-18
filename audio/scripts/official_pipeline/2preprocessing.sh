
# Launching this script will pre-process the dataset in tmp file.

# First, create the folder that will contain the processed data
mkdir -p ${MY_HOME}/data/ansim/tmp

cd ${MY_HOME}/src
declare -a experiments=("rmcl")

# Define the hypothesis counts and corresponding sizes for Histogram model
num_hypotheses=9
seed=1234

# Preprocessing
python train.py model=WTABased.yaml experiment=ansim/full-training/scoring-ansim-noisy-classes-online.yaml hydra.job.name=test-rmcl-100-epoch-gen-${num_hypotheses}-hyps model.hparams.num_hypothesis=${num_hypotheses} model.hparams.mode='wta' trainer.max_epochs=1 seed=${seed} test=True