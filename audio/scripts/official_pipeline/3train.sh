
# DATASET ANSIM 

cd ${MY_HOME}/src
declare -a experiments=("rmcl" "mdn" "histogram")

# Define the hypothesis counts and corresponding sizes for Histogram model
declare -a hyps=("9 3,3" "16 4,4" "20 5,4" "25 5,5")
seed=1234

# Loop through each number of hypotheses
for hyp in "${hyps[@]}"; do
    # Split hyp into number of hypotheses and histogram sizes
    IFS=' ' read -r num_hypotheses sizes <<< "$hyp"
    nrows=$(echo $sizes | cut -d',' -f1)
    ncols=$(echo $sizes | cut -d',' -f2)

    # Rmcl model
    python train.py model=WTABased.yaml experiment=ansim/full-training/scoring-ansim-noisy-classes-online.yaml hydra.job.name=test-rmcl-100-epoch-gen-${num_hypotheses}-hyps model.hparams.num_hypothesis=${num_hypotheses} model.hparams.mode='wta' seed=${seed} test=False 
    
    # Mixture of von mises model
    python train.py model=Mixture_NLLVonMises.yaml experiment=ansim/full-training/nll-vonmises-ansim-noisy-classes-online.yaml hydra.job.name=test-100-epoch-vonmises-1modes-seed1234-logkappapred model.hparams.compute_nll=True model.hparams.num_modes=1 seed=${seed} model.hparams.log_kappa_pred=True test=False 

    # Histogram model
    python train.py model=Histogram.yaml experiment=ansim/full-training/histogram-ansim-noisy-classes-online.yaml hydra.job.name=test-NG-100-epoch-histogram-gen-${nrows}-${ncols} model.hparams.nrows=${nrows} model.hparams.ncols=${ncols} seed=${seed} test=False 
done