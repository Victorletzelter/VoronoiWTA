#!/bin/bash

# Extract the checkpoints in a JSON file

# This script takes as input three folder paths corresponding to methods names e.g.,

# <YOUR_PATH>/VoronoiWTA/audio/checkpoints/histogram
# <YOUR_PATH>/VoronoiWTA/audio/checkpoints/score_based_wta
# <YOUR_PATH>/VoronoiWTA/audio/checkpoints/von_mises_fisher_mdn

# and it returns a JSON file with the key associated with the method name and the number of hypotheses
# e.g.,
# histogram-9-hyps: PATH1
# histogram-16-hyps: PATH2
# histogram-20-hyps: PATH3
# histogram-25-hyps: PATH4
# score_based_wta-9-hyps: PATH5
# ... etc

# It assumes a folder structure with checkpoints located at
# methods_name/folder_name/checkpoints/epoch_<epoch_number>.ckpt
# e.g., histogram/histogram-5-5/checkpoints/epoch_<epoch_number>.ckpt
# histogram/histogram-4-4/checkpoints/epoch_<epoch_number>.ckpt

# It also assumes that the number of hypotheses are in the form
# 5-5, 5-4 for Histogram (corresponding to 5x5=25 or 5x4=20 hypotheses) e.g., in the name "histogram-5-5"
# 9-hyps, 16-hyps for score_based_wta, e.g., in the name "score-based-wta-9-hyps"
# 9modes, 16modes, 20modes in mdn, e.g., in the name "vonmises-9modes"

# Note that those information are given in the hydra config files in <folder_name>/.hydra/config.yaml, 
# under the key:
# model.hparams.num_rows (integer) and model.hparams.num_cols (integer) in histogram. The number of hypothesis in this case is the product of the two.
# model.hparams.num_hypothesis (integer) in score based WTA
# models.hparams.num_modes (integer) in mdn

# Define a list of allowed method names
declare -A allowed_methods
allowed_methods[histogram]=1
allowed_methods[score_based_wta]=1
allowed_methods[mdn]=1

# Declare an associative array
declare -A method_paths

args=("$@")  # Store all arguments in an array

# Check for an even number of arguments
if (( ${#args[@]} % 2 != 0 )); then
    echo "Error: Please provide pairs of method names and paths."
    exit 1
fi

# Populate the array with method and path pairs from command-line arguments
for ((i = 0; i < ${#args[@]}; i+=2)); do
    method_name="${args[i]}"
    path="${args[i+1]}"

    # Check if the method name is allowed
    if [[ -z ${allowed_methods[$method_name]} ]]; then
        echo "Error: Unsupported method name '$method_name'."
        exit 1
    fi

    method_paths["$method_name"]="$path"
done

# Define a function to extract hypothesis count based on method and config path
function get_hypothesis_count() {
    local method=$1
    local config_file=$2

    case $method in
        ${method_paths[histogram]})
            local num_rows=$(grep 'nrows:' "$config_file" | awk '{print $2}')
            local num_cols=$(grep 'ncols:' "$config_file" | awk '{print $2}')
            echo $((num_rows * num_cols))
            ;;
        ${method_paths[score_based_wta]})
            grep 'num_hypothesis:' "$config_file" | awk '{print $2}'
            ;;
        ${method_paths[mdn]})
            grep 'num_modes:' "$config_file" | awk '{print $2}'
            ;;
        *)
            echo "Unsupported method: $method"
            exit 1
            ;;
    esac
}

# Main function to process directories
function process_directories() {
    local method_path=$1
    local method_name=$(basename "$method_path")

    first_dir="true"
    # Iterate over all hypothesis folders within the method directory
    for hyp_dir in "$method_path"/*; do
        if [ -d "$hyp_dir" ]; then
            local config_path="$hyp_dir/.hydra/config.yaml"
            local hypothesis_count=$(get_hypothesis_count "$method_name" "$config_path")
            local key="${method_name}-${hypothesis_count}-hyps"
            local paths=$(find "$hyp_dir/checkpoints" -name 'epoch_*.ckpt')

            # Append results to JSON object
            for path in $paths; do
                if [ "$first_dir" = "true" ]; then
                    first_dir="false"
                else
                    echo "," >> $output_json
                fi
                echo -n "\"$key\": " >> $output_json
                echo -n "\"$path\"" >> $output_json
                # echo "," >> $output_json
            done
        fi
    done
}

# Determine the parent directory of the first method directory
common_parent_directory=$(realpath "$2/..")

# Define the JSON output file path
output_json="${common_parent_directory}/checkpoints.json"

echo "{" > $output_json

# Loop over each method directory provided as command-line arguments
first_dir="true"
for dir in "$@"; do
    if [ "$first_dir" = "true" ]; then
        first_dir="false"
    else
        echo "," >> $output_json
    fi
    process_directories "$dir"
done

# Close the JSON object
echo "}" >> $output_json
echo "JSON file created: $output_json"
