# extract_ckpts.py

import json
import os
import argparse

# import yaml

# SAVE_DIR = "${MY_HOME}/scripts"

parser = argparse.ArgumentParser(description="Extract checkpoint paths.")
parser.add_argument(
    "--base_dir",
    type=str,
    help="The base directory where the checkpoint files are stored",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="${MY_HOME}/scripts_annealing",
    help="The directory where the JSON file will be saved",
)
args = parser.parse_args()

# Use the base directory from the command-line argument
base_dir = args.base_dir
SAVE_DIR = args.save_dir
dataset_name = base_dir.split("train/")[1]

ckpt_paths_dic = {}

for folder in os.listdir(base_dir):

    hyp_count = folder.split("hyps")[0].split("_")[-2]
    model_name = folder.split("model")[-1].split("_")[1]
    if "checkpoints" not in os.listdir(os.path.join(base_dir, folder)):
        continue
    ckpt_folder = os.path.join(base_dir, folder, "checkpoints")
    ckpt_files = os.listdir(ckpt_folder)
    ckpt_file_name = [e for e in ckpt_files if "epoch" in e]

    if len(ckpt_file_name) == 0:
        continue
    else:
        ckpt_file_name = [e for e in ckpt_files if "epoch" in e][0]
    ckpt_path = os.path.join(ckpt_folder, ckpt_file_name)

    key = f"{dataset_name}_{model_name}_{hyp_count}_hyp"

    ckpt_paths_dic[key] = ckpt_path

# Define the filename for the JSON file
json_filename = "checkpoints_{}.json".format(dataset_name)
json_filename = os.path.join(SAVE_DIR, json_filename)

# Write the dictionary to a JSON file
with open(json_filename, "w") as json_file:
    json.dump(ckpt_paths_dic, json_file, indent=4)  # 'indent' for pretty printing
