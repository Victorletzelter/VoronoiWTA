import pandas as pd
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

import seaborn as sns
import os
import pickle
import argparse


def str_to_list(l="[1, 2, 3]"):
    return list(map(int, l[1:-1].split(",")))


def _merge_lists(seq, order):
    seen = set()
    seen_add = seen.add
    new_seq = list()
    new_seq_i = list()
    for x in order:
        if x in seq and not (x in seen or seen_add(x)):
            new_seq.append(x)
            new_seq_i.append(seq.index(x))
    return new_seq_i, new_seq


def renaming(name):

    data_name = name.split("_model")[0]
    model_name = name.split("_model_")[1].split("_")[0]
    n_hyps = name.split("_hyps_")[0].split("_")[-1]

    if model_name == "VoronoiWTA":

        if "uniform" not in name and "h_opt" not in name:
            scaling_factor = name.split("_h_")[-1]
            if "seed" in scaling_factor:
                scaling_factor = scaling_factor.split("_seed")[0]
            return f"Voronoi-WTA h={scaling_factor}"
        elif "h_opt" in name:
            if "tol_0.01" in name:
                return f"Voronoi-WTA h opt -2"
            else:
                return f"Voronoi-WTA h opt -1"
        else:
            return "Unif. Voronoi-WTA"

    elif model_name == "DiracWTA":
        return "Dirac Voronoi-WTA"

    elif model_name == "KernelWTA":
        if "uniform" not in name:
            scaling_factor = name.split("_h_")[-1]
            if "seed" in scaling_factor:
                scaling_factor = scaling_factor.split("_seed")[0]
            return f"Kernel-WTA h={scaling_factor}"
        else:
            return "Unif. Kernel-WTA"

    elif "UnweightedKernel" in model_name:
        scaling_factor = name.split("_h_")[-1]
        if "seed" in scaling_factor:
            scaling_factor = scaling_factor.split("_seed")[0]
        return f"Unweighted Kernel-WTA h={scaling_factor}"

    elif model_name == "Histogram":
        if "uniform" not in name:
            scaling_factor = name.split("_h_")[-1]
            if "seed" in scaling_factor:
                scaling_factor = scaling_factor.split("_seed")[0]
            return f"Truncated-Kernel Histogram h={scaling_factor}"
        else:
            return "Unif. Histogram"  # Corresponds to Unif. Histogram

    elif model_name == "GaussMix":
        return "MDN"


def hyp_hist(row):
    if "Histogram" in row["Name"]:
        list_sizes = str_to_list(row["model/sizes"])
        n = len(list_sizes)
        row["n_hyps"] = 1
        for i in range(n):
            row["n_hyps"] *= list_sizes[i]
    return row


def extract_nll_df(csv_file):
    df = pd.read_csv(csv_file)

    df = df.rename(
        columns={
            "model/num_hypothesis": "n_hyps",
            "model/scaling_factor": "scaling_factor",
            "test_nll": "nll",
        }
    )

    df["model"] = df["Name"].apply(renaming)
    df = df[~(df["model"].str.contains("Voronoi-WTA h opt -2"))]
    df = df[~(df["model"].str.contains("Dirac Voronoi-WTA"))]
    df = df[~(df["model"].str.contains("Truncated-Kernel Histogram"))]
    # df = df[~(df['model'].str.contains('Unif. Voronoi-WTA'))]
    df = df[~(df["model"].str.contains("Kernel-WTA"))]
    df = df[~(df["model"].str.contains("Kernel-WTA"))][
        ~(df["model"].str.contains("Unweighted"))
    ]
    df.loc[df["Name"].str.contains("uniform"), "scaling_factor"] = np.nan
    df.loc[df["Name"].str.contains("h_opt"), "scaling_factor"] = -1.0

    df = df.apply(hyp_hist, axis=1)

    return df


def extract_emd_df(csv_file):
    df = pd.read_csv(csv_file)

    df = df.rename(
        columns={
            "model/num_hypothesis": "n_hyps",
            "model/scaling_factor": "scaling_factor",
            "test_emd": "emd",
        }
    )

    df["model"] = df["Name"].apply(renaming)
    df = df[~(df["model"].str.contains("Voronoi-WTA h opt -2"))]
    df = df[~(df["model"].str.contains("Truncated-Kernel Histogram"))]
    # df = df[~(df['model'].str.contains('Unif. Voronoi-WTA'))]
    df.loc[df["Name"].str.contains("uniform"), "scaling_factor"] = np.nan
    df.loc[df["Name"].str.contains("h_opt"), "scaling_factor"] = -1.0
    df = df[~(df["model"].str.contains("Kernel-WTA"))]
    df = df[~(df["model"].str.contains("Kernel-WTA"))][
        ~(df["model"].str.contains("Unweighted"))
    ]

    df = df.apply(hyp_hist, axis=1)

    return df


def extract_risk_df(csv_file, csv_file_zador):
    df = pd.read_csv(csv_file)

    df = df.rename(
        columns={
            "model/num_hypothesis": "n_hyps",
            "model/scaling_factor": "scaling_factor",
            "test_risk": "risk",
        }
    )

    df["model"] = df["Name"].apply(renaming)
    df = df[~(df["model"].str.contains("Voronoi-WTA h opt -2"))]
    df = df[~(df["model"].str.contains("Dirac Voronoi-WTA"))]
    df.loc[df["Name"].str.contains("VoronoiWTA"), "model"] = "Dirac Voronoi-WTA"
    df = df[~(df["model"].str.contains("Truncated-Kernel Histogram"))]
    # df = df[~(df['model'].str.contains('Unif. Voronoi-WTA'))]
    df.loc[df["Name"].str.contains("uniform"), "scaling_factor"] = np.nan
    df.loc[df["Name"].str.contains("h_opt"), "scaling_factor"] = -1.0
    df = df[~(df["model"].str.contains("Kernel-WTA"))]
    df = df[~(df["model"].str.contains("Kernel-WTA"))][
        ~(df["model"].str.contains("Unweighted"))
    ]

    df_zador = pd.read_csv(csv_file_zador)
    df_zador = df_zador.rename(columns={"n": "n_hyps", "zador_risk": "risk"})

    df = df.apply(hyp_hist, axis=1)

    return df, df_zador


#  - MAKE SUBPLOTS

METRICS = {
    "nll": "NLL",
    "emd": "EMD",
    "risk": "Quantization error",
}
DATASETS = {
    "single-gaussian-not-centered": "Single Gaussian",
    "rotating-two-moons": "Rotating Two Moons",
    "changing-damier": "Changing Damier",
}

FUNCTIONS = {
    "nll": extract_nll_df,
    "emd": extract_emd_df,
    "risk": extract_risk_df,
}

std_color_palette = sns.color_palette("muted")
wta_color_palette = sns.dark_palette("seagreen")

COLORS = {
    "Unif. Histogram": std_color_palette[1],
    "Dirac Voronoi-WTA": wta_color_palette[5],
    "MDN": std_color_palette[0],
    "Theoretical WTA": wta_color_palette[5],
    "Theoretical Histogram": std_color_palette[1],
    "Unif. Voronoi-WTA": wta_color_palette[1],
    "Voronoi-WTA h=0.05": wta_color_palette[5],
    "Voronoi-WTA h=0.1": wta_color_palette[5],
    "Voronoi-WTA h=0.15": wta_color_palette[4],
    "Voronoi-WTA h=0.2": wta_color_palette[3],
    "Voronoi-WTA h=0.3": wta_color_palette[2],
    # "Voronoi-WTA h=0.5": wta_color_palette[2],
    # "Voronoi-WTA h opt -1": wta_color_palette[2]
}

MARKERS = {
    "Unif. Histogram": "s",
    "Dirac Voronoi-WTA": "d",
    "MDN": "o",
    "Theoretical WTA": ".",
    "Theoretical Histogram": ".",
    "Unif. Voronoi-WTA": "H",
    "Voronoi-WTA h=0.05": "*",
    "Voronoi-WTA h=0.1": "^",
    "Voronoi-WTA h=0.15": "X",
    "Voronoi-WTA h=0.2": "P",
    "Voronoi-WTA h=0.3": "<",
    # "Voronoi-WTA h=0.5": "P",
    # "Voronoi-WTA h opt -1": "^"
}

TO_KEEP = [np.nan, np.infty, 0.05, 0.0, 0.1, 0.2, 0.3]

METHODS_ORDER = [
    "Dirac Voronoi-WTA",
    # "Voronoi-WTA h=0.05",
    "Voronoi-WTA h=0.1",
    "Voronoi-WTA h=0.15",
    "Voronoi-WTA h=0.2",
    "Voronoi-WTA h=0.3",
    # "Voronoi-WTA h=0.5",
    # "Voronoi-WTA h opt -1",
    "Unif. Voronoi-WTA",
    "MDN",
    "Unif. Histogram",
    "Theoretical WTA",
    "Theoretical Histogram",
]

# Y_MIN_NLL = {
#     "Single Gaussian": -0.4,
#     "Rotating Two Moons": 0.4,
#     "Changing Damier": 1.18,
# }

Y_MAX_NLL = {
    "Single Gaussian": 0.7,
    "Rotating Two Moons": 1.5,
    "Changing Damier": 1.42,
}

Y_MAX_EMD = {
    "Single Gaussian": 0.55,
    "Rotating Two Moons": 0.235,
    # "Changing Damier": 0.145,
    "Changing Damier": 0.245,
}

# Y_MIN_EMD = {
#     "Single Gaussian": 0.0,
#     "Rotating Two Moons": 0.123,
#     "Changing Damier": 0.0955
#     }

TEXT_WIDTH = 3.25
PAGE_WIDTH = 6.875
FONTSIZE = 10


def build_paths(base_dir, metric, dataset, folder_name="saved_csv"):
    csv_files = os.listdir(folder_name)
    csv_files_zador = os.listdir(os.path.join(base_dir, "saved_zador"))

    if dataset == "single-gaussian-not-centered":
        # search for the csv file in the results folder
        csv_file = [e for e in csv_files if "gauss_not_centered" in e][0]
        csv_file_zador = [e for e in csv_files_zador if "gaussnotcentered" in e][0]
    elif dataset == "rotating-two-moons":
        csv_file = [e for e in csv_files if "rotating_moons" in e][0]
        csv_file_zador = [e for e in csv_files_zador if "rotatingmoons" in e][0]
    elif dataset == "changing-damier":
        csv_file = [e for e in csv_files if "changing_damier" in e][0]
        csv_file_zador = [e for e in csv_files_zador if "changingdamier" in e][0]

    csv_file = os.path.join(base_dir, folder_name, csv_file)
    csv_file_zador = os.path.join(base_dir, "saved_zador", csv_file_zador)

    return csv_file, csv_file_zador


def single_plot(
    df,
    metric,
    df_theory=None,
    metric_names=None,
    dataset_name=None,
    models_order=None,
    models_order_theory=None,
    marker_per_model=None,
    color_per_model=None,
    valid_scaling=None,
    xon=True,
    yon=True,
    ax=None,
    row=None,
    plot_std=None,
):

    if ax is None:
        _, ax = plt.subplots()

    if valid_scaling is not None:
        valid_scaling = [float(x) for x in valid_scaling]
        df = df[df["scaling_factor"].isin(valid_scaling)].reset_index(drop=True)

        # Update model order
        models_order_updated = []

        for model in models_order:
            if "h=" in model:
                h = float(model.split("h=")[-1])
                if h in valid_scaling:
                    models_order_updated.append(model)
            else:
                models_order_updated.append(model)

        models_order = models_order_updated

    else:
        valid_scaling = [np.nan]
        df = df[df["scaling_factor"].isin(valid_scaling)].reset_index(drop=True)

    if plot_std:
        ax = sns.lineplot(
            df,
            x="n_hyps",
            y=metric,
            errorbar="sd",
            hue="model",
            style="model",
            markers=marker_per_model,
            dashes=False,
            ax=ax,
            palette=color_per_model,
            hue_order=models_order,
        )
    else:
        ax = sns.lineplot(
            df,
            x="n_hyps",
            y="Mean {}".format(metric),
            errorbar="sd",
            hue="model",
            style="model",
            markers=marker_per_model,
            dashes=False,
            ax=ax,
            palette=color_per_model,
            hue_order=models_order,
        )

    # set limits y axis
    if metric == "nll" and dataset_name in Y_MAX_NLL:
        ax.set_ylim(ax.get_ylim()[0], Y_MAX_NLL[dataset_name])
    elif metric == "emd" and dataset_name in Y_MAX_EMD:
        # ax.set_ylim(Y_MIN_EMD[dataset_name], Y_MAX_EMD[dataset_name])
        ax.set_ylim(ax.get_ylim()[0], Y_MAX_EMD[dataset_name])

    h, l = ax.get_legend_handles_labels()

    # Plot theoretical curves when available in dashed lines
    if df_theory is not None:
        for model_name in df_theory["model"].unique():
            df_theory_model = df_theory.query("model == @model_name").reset_index(
                drop=True
            )

            (line,) = ax.plot(
                df_theory_model["n_hyps"],
                df_theory_model[metric],
                linestyle="--",
                color=color_per_model[model_name],
                label=model_name,
            )

            h.append(line)
            l.append(model_name)

    if row == 0:
        ax.set_title(dataset_name, fontsize=18)
    ax.grid()

    if not xon:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Number of hypotheses", fontsize=13)

    if not yon:
        ax.set_ylabel("")
    elif metric_names is not None:
        ax.set_ylabel(metric_names[metric], fontsize=13)

    ax.get_legend().remove()

    return h, l


def make_subplots(
    base_dir,
    figsize=None,
    save=False,
    save_pth=None,
    list_folder_names=None,
    plot_std=True,
):
    fig, ax_arr = plt.subplots(
        3,
        3,
        sharex=True,
        figsize=figsize,
    )

    legend_handles = []
    legend_labels = []

    for row, metric in enumerate(METRICS):
        extract_func = FUNCTIONS[metric]
        for col, dataset in enumerate(DATASETS):

            df = pd.DataFrame()

            for folder_name in list_folder_names:

                csv_file, csv_file_zador = build_paths(
                    base_dir=base_dir,
                    metric=metric,
                    dataset=dataset,
                    folder_name=folder_name,
                )
                if "changing" in dataset:
                    TO_KEEP = [np.nan, np.infty, 0.0, 0.2, 0.3]
                else:
                    TO_KEEP = [np.nan, np.infty, 0.0, 0.15, 0.1, 0.2]

                if metric != "risk":
                    df_folder = extract_func(csv_file)
                    df_zador = None
                else:
                    df_folder, df_zador = extract_func(csv_file, csv_file_zador)

                df = pd.concat([df, df_folder], ignore_index=True, axis=0)

            # Aggregate by Name and compute mean and std
            df["Mean {}".format(metric)] = df.groupby(["model", "n_hyps"])[
                "{}".format(metric)
            ].transform("mean")
            df["Std {}".format(metric)] = df.groupby(["model", "n_hyps"])[
                "{}".format(metric)
            ].transform("std")

            plot_specific_order = [
                m for m in METHODS_ORDER if m in df["model"].unique()
            ]

            if df_zador is not None:
                plot_specific_order_theory = [
                    m for m in METHODS_ORDER if m in df_zador["model"].unique()
                ]
            else:
                plot_specific_order_theory = None

            h, l = single_plot(
                df,
                metric=metric,
                df_theory=df_zador,
                metric_names=METRICS,
                dataset_name=DATASETS[dataset],
                color_per_model=COLORS,
                marker_per_model=MARKERS,
                models_order=plot_specific_order,
                models_order_theory=plot_specific_order_theory,
                valid_scaling=TO_KEEP if metric != "risk" else None,
                yon=col == 0,
                ax=ax_arr[row, col],
                row=row,
                plot_std=plot_std,
            )
            legend_handles += h
            legend_labels += l

    legend_idx, legend_labels = _merge_lists(legend_labels, METHODS_ORDER)
    legend_handles = [legend_handles[i] for i in legend_idx]

    for i, label in enumerate(legend_labels):
        if "h=" in label:
            value_h = float(label.split("h=")[-1])
            # Replace h=... by $h=...$
            legend_labels[i] = label.replace(f"h={value_h}", f"$h={value_h}$")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=5,
        loc="upper center",
        fontsize=12,
    )
    if save:
        print(plot_std)
        if plot_std is True:
            fig.savefig(os.path.join(save_pth, "main_plots_std.pdf"))
            fig.savefig(os.path.join(save_pth, "main_plots_std.png"))
        else:
            fig.savefig(os.path.join(save_pth, "main_plots.pdf"))
            fig.savefig(os.path.join(save_pth, "main_plots.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_results", type=str, default="${MY_HOME}/results/")
    parser.add_argument("--save_path", type=str, default="${MY_HOME}/figures_std")
    parser.add_argument(
        "--list_folder_names", type=str, default="${MY_HOME}/results/considered_csv"
    )
    parser.add_argument("--plot_std", type=str, default="False")

    args = parser.parse_args()

    def str_to_bool(s):
        if s == "True":
            return True
        elif s == "False":
            return False
        else:
            raise ValueError

    args.plot_std = str_to_bool(args.plot_std)

    list_folder_names = os.listdir(args.list_folder_names)
    for i in range(len(list_folder_names)):
        list_folder_names[i] = os.path.join(
            args.list_folder_names, list_folder_names[i]
        )

    make_subplots(
        figsize=(PAGE_WIDTH * 2, PAGE_WIDTH),
        save=True,
        save_pth=args.save_path,
        base_dir=args.base_dir_results,
        list_folder_names=list_folder_names,
        plot_std=args.plot_std,
    )
