#!/bin/bash

declare -a hyps=(9 16 20 25 49 100)

cd ${MY_HOME}

if [ -d "figures_std" ]; then
    rm -r figures_std
fi

mkdir figures_std

cd ${MY_HOME}/scripts

python plot_main_fig_std.py --save_path=${MY_HOME}/figures_std --plot_std=True
python plot_main_fig_std.py --save_path=${MY_HOME}/figures_std --plot_std=False

for num_hypotheses in "${hyps[@]}"; do
    python plot_secondary_fig_std.py --save_path=${MY_HOME}/figures_std --plot_unweighted=True --num_hypotheses=${num_hypotheses}
    python plot_secondary_fig_std.py --save_path=${MY_HOME}/figures_std --plot_unweighted=False --num_hypotheses=${num_hypotheses}
    python plot_single_fig_std.py --save_path=${MY_HOME}/figures_std --num_hypotheses=${num_hypotheses} --dataset=mixtures-uni-to-gaussians
done