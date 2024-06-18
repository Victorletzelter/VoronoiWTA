import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

import seaborn as sns
import os
import argparse


def hyp_hist(row):
    if "Histogram" in row["Name"]:
        list_sizes = str_to_list(row["model/sizes"])
        n = len(list_sizes)
        row["model/num_hypothesis"] = 1
        for i in range(n):
            row["model/num_hypothesis"] *= list_sizes[i]
    return row


def build_paths(dataset, base_dir, folder_name):
    csv_files = os.listdir(os.path.join(base_dir, folder_name))
    csv_files_zador = os.listdir(os.path.join(base_dir, "saved_zador"))

    if dataset == "single-gaussian-not-centered":
        csv_file = [e for e in csv_files if "gauss_not_centered" in e][0]
        csv_file_zador = [e for e in csv_files_zador if "gaussnotcentered" in e][0]
    elif dataset == "rotating-two-moons":
        csv_file = [e for e in csv_files if "rotating_moons" in e][0]
        csv_file_zador = [e for e in csv_files_zador if "rotatingmoons" in e][0]
    elif dataset == "changing-damier":
        csv_file = [e for e in csv_files if "changing_damier" in e][0]
        csv_file_zador = [e for e in csv_files_zador if "changingdamier" in e][0]
    elif dataset == "mixtures-uni-to-gaussians":
        csv_file = [e for e in csv_files if "mixtures_uni_to_gaussians" in e][0]
        csv_file_zador = "dummy_zador"
    csv_file = os.path.join(base_dir, folder_name, csv_file)
    csv_file_zador = os.path.join(base_dir, "saved_zador", csv_file_zador)

    # csv_files_2 = os.listdir(os.path.join(base_dir,'saved_csv_OK'))

    # if dataset == "single-gaussian-not-centered":
    #     csv_file_2 = [e for e in csv_files_2 if 'gauss_not_centered' in e][0]
    # elif dataset == "rotating-two-moons":
    #     csv_file_2 = [e for e in csv_files_2 if 'rotating_moons' in e][0]
    # elif dataset == "changing-damier":
    #     csv_file_2 = [e for e in csv_files_2 if 'changing_damier' in e][0]
    # elif dataset == 'mixtures-uni-to-gaussians':
    #     csv_file_2 = [e for e in csv_files_2 if 'mixtures_uni_to_gaussians' in e][0]
    # csv_file_2 = os.path.join(base_dir,'saved_csv_OK',csv_file_2)

    return csv_file, csv_file_zador  # , csv_file_2


# def build_paths(dataset):
#     base_dir = '${MY_HOME}/results/'
#     csv_files = os.listdir(os.path.join(base_dir,'saved_csv'))
#     csv_files_zador = os.listdir(os.path.join(base_dir,'saved_zador'))

#     if dataset == "single-gaussian-not-centered":

#         #search for the csv file in the results folder
#         csv_file = [e for e in csv_files if 'gauss_not_centered' in e][0]
#         csv_file_zador = [e for e in csv_files_zador if 'gaussnotcentered' in e][0]

#     elif dataset == "rotating-two-moons":
#         csv_file = [e for e in csv_files if 'rotating_moons' in e][0]
#         csv_file_zador = [e for e in csv_files_zador if 'rotatingmoons' in e][0]
#     elif dataset == "changing-damier":
#         csv_file = [e for e in csv_files if 'changing_damier' in e][0]
#         csv_file_zador = [e for e in csv_files_zador if 'changingdamier' in e][0]
#     elif dataset == 'mixtures-uni-to-gaussians':
#         csv_file = [e for e in csv_files if 'mixtures_uni_to_gaussians' in e][0]
#         csv_file_zador = 'dummy_zador'
#     csv_file = os.path.join(base_dir,'saved_csv',csv_file)
#     csv_file_zador = os.path.join(base_dir,'saved_zador',csv_file_zador)

#     return csv_file, csv_file_zador


def construct_nll_dict(
    base_dir, dataset="changing-damier", num_hypotheses=20, folder_name=None
):
    csv_file = build_paths(dataset=dataset, base_dir=base_dir, folder_name=folder_name)[
        0
    ]
    # csv_file_2 = build_paths(dataset=dataset, base_dir=base_dir)[2]
    df = pd.read_csv(csv_file)
    # df = pd.concat([df,pd.read_csv(csv_file_2)], ignore_index=True, axis=0)
    df = df.apply(hyp_hist, axis=1)
    # Get unique entries of df by df['Name']
    df = df.drop_duplicates(subset="Name")
    df["model"] = df["Name"].apply(renaming)
    df = df[df["model/num_hypothesis"] == num_hypotheses]
    df = df[~(df["model"].isna())]
    df = df[~(df["model"].str.contains("h opt"))]

    models_names = {
        "Truncated-Kernel Histogram": {},
        "Voronoi-WTA h=": {},
        "Unweighted Kernel-WTA h=": {},
        "Kernel-WTA h=": {},
        "Unif. Voronoi-WTA": {},
        "Unif. Histogram": {},
    }
    models_mapping = {
        "Truncated-Kernel Histogram": "Histogram",
        "Voronoi-WTA h=": "rMCL",
        "Unweighted Kernel-WTA h=": "kde_rMCL",
        "Kernel-WTA h=": "kde_weighted_rMCL",
        "Unif. Voronoi-WTA": "rMCL_uniform",
        "Unif. Histogram": "Histogram_uniform",
    }
    models = {
        "Histogram": {},
        "rMCL": {},
        "kde_rMCL": {},
        "kde_weighted_rMCL": {},
        "rMCL_uniform": {},
        "Histogram_uniform": {},
    }
    list_h = {}

    for model in models_names.keys():

        df_model = df[df["model"].str.contains(model)]
        df_model = df[df["model"].str[0:8] == model[0:8]]

        list_h[model] = df_model["model/scaling_factor"].unique()
        list_h[model].sort()

        model_key = models_mapping[model]

        for h in list_h[model]:
            df_h = df_model[df_model["model/scaling_factor"] == h]
            if df_h.shape[0] != 1:
                print(f"Warning: {model} with h={h} has {df_h.shape[0]} entries")
                assert df_h.shape[0] == 1
            models[model_key][h] = df_h["test_nll"].values[0]

    return models


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
        if "h_opt" in name:
            if "tol_0.01" in name:
                return f"Voronoi-WTA h opt"
            else:
                return "Voronoi-WTA h opt -1"
        scaling_factor = name.split("_h_")[-1]
        if "seed" in scaling_factor:
            scaling_factor = scaling_factor.split("_seed")[0]
        if "uniform" not in name:
            return f"Voronoi-WTA h={scaling_factor}"
        else:
            return "Unif. Voronoi-WTA"

    elif model_name == "KernelWTA":
        scaling_factor = name.split("_h_")[-1]
        if "seed" in scaling_factor:
            scaling_factor = scaling_factor.split("_seed")[0]
        if "uniform" not in name:
            return f"Kernel-WTA h={scaling_factor}"
        else:
            return "Unif. Kernel-WTA"

    elif "UnweightedKernel" in model_name:
        scaling_factor = name.split("_h_")[-1]
        if "seed" in scaling_factor:
            scaling_factor = scaling_factor.split("_seed")[0]
        return f"Unweighted Kernel-WTA h={scaling_factor}"

    elif model_name == "Histogram":
        scaling_factor = name.split("_h_")[-1]
        if "seed" in scaling_factor:
            scaling_factor = scaling_factor.split("_seed")[0]
        if "uniform" not in name:
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
    df = df[~(df["model"].str.contains("h opt"))]
    df = df[~(df["model"].isna())]
    df = df[~(df["model"].str.contains("Truncated-Kernel Histogram"))]
    df = df[~(df["model"].str.contains("Kernel-WTA"))]
    df = df[~(df["model"].str.contains("Kernel-WTA"))][
        ~(df["model"].str.contains("Unweighted"))
    ]
    df.loc[df["Name"].str.contains("uniform"), "scaling_factor"] = np.nan

    df = df.apply(hyp_hist, axis=1)

    return df


def hyp_hist(row):
    if "Histogram" in row["Name"]:
        list_sizes = str_to_list(row["model/sizes"])
        n = len(list_sizes)
        row["model/num_hypothesis"] = 1
        for i in range(n):
            row["model/num_hypothesis"] *= list_sizes[i]
    return row


def gen_plot(dataset, num_hypothesis, ax, plot_unweighted, base_dir, folder_name):
    # dataset = 'changing_damier'
    # dataset = 'rotating_two_moons'
    # dataset = 'single_gaussian_not_centered'
    # dataset = 'mixture_uni_to_gaussians'

    models = {
        "Histogram": {},
        "rMCL": {},
        "kde_rMCL": {},
        "kde_weighted_rMCL": {},
        "rMCL_uniform": {},
        "Histogram_uniform": {},
    }

    models = construct_nll_dict(
        base_dir=base_dir,
        dataset=dataset,
        num_hypotheses=num_hypothesis,
        folder_name=folder_name,
    )

    std_color_palette = sns.color_palette("muted")
    wta_color_palette = sns.dark_palette("seagreen")

    COLORS = {
        "Histogram": std_color_palette[1],
        "Dirac Voronoi-WTA": wta_color_palette[5],
        "MDN": std_color_palette[0],
        "Theoretical WTA": wta_color_palette[5],
        "Theoretical Histogram": std_color_palette[1],
        "Unif. Voronoi-WTA": wta_color_palette[1],
        "Voronoi-WTA h=0.1": wta_color_palette[4],
        "Voronoi-WTA h=0.2": wta_color_palette[3],
        "Voronoi-WTA h=0.3": wta_color_palette[2],
    }

    MARKERS = {
        "Histogram": "s",
        "Dirac Voronoi-WTA": "d",
        "MDN": "o",
        "Theoretical WTA": ".",
        "Theoretical Histogram": ".",
        "Unif. Voronoi-WTA": "H",
        "Voronoi-WTA h=0.1": "^",
        "Voronoi-WTA h=0.2": "X",
        "Voronoi-WTA h=0.3": "P",
    }

    ##########

    TEXT_WIDTH = 3.25
    PAGE_WIDTH = 6.875
    FONTSIZE = 10

    list_dic = []
    color_per_model = {}
    marker_per_model = {}

    list_hist_histogram_uniform = []
    color_histogram_uniform = {}
    list_hist_rMCL_uniform = []
    color_rmcl_uniform = {}

    if plot_unweighted is True:
        list_models = ["Histogram", "rMCL", "kde_rMCL", "kde_weighted_rMCL"]
    else:
        list_models = ["Histogram", "rMCL", "kde_weighted_rMCL"]

    for model in list_models:
        if model == "Histogram":
            model_label = "Truncated-Kernel Histogram"
        elif model == "rMCL":
            model_label = "Voronoi-WTA"
        elif model == "kde_weighted_rMCL":
            model_label = "Kernel-WTA"
        elif model == "kde_rMCL":
            model_label = "Unweighted Kernel-WTA"

        list_h = list(models[model].keys())
        list_h_plot = [float(h) for h in list_h]

        if model_label == "Truncated-Kernel Histogram":  #'o'
            marker = "s"
            color = COLORS["Histogram"]
            list_nll_plot = [models[model][h] for h in list_h]
            list_h_uniform_plot = list_h
            h_uniform = list(models["Histogram_uniform"].keys())[0]
            list_nll_uniform_plot = [
                models["Histogram_uniform"][h_uniform] for h in list_h_uniform_plot
            ]
            for i, h in enumerate(list_h_uniform_plot):
                list_hist_histogram_uniform.append(
                    {
                        "h": float(list_h_uniform_plot[i]),
                        "model": "Unif. Histogram",
                        "nll": list_nll_uniform_plot[i],
                    }
                )
            color_histogram_uniform["Unif. Histogram"] = color

        elif model_label == "Voronoi-WTA":  # marker='^',color='green')
            marker = "^"
            color = COLORS["Dirac Voronoi-WTA"]
            list_nll_plot = [models[model][h] for h in list_h]
            list_h_uniform_plot = list_h
            h_uniform = list(models["rMCL_uniform"].keys())[0]
            list_nll_uniform_plot = [
                models["rMCL_uniform"][h_uniform] for h in list_h_uniform_plot
            ]
            for i, h in enumerate(list_h_uniform_plot):
                list_hist_rMCL_uniform.append(
                    {
                        "h": float(list_h_uniform_plot[i]),
                        "model": "Unif. Voronoi-WTA",
                        "nll": list_nll_uniform_plot[i],
                    }
                )
            color_rmcl_uniform["Unif. Voronoi-WTA"] = color

        elif model_label == "Unweighted Kernel-WTA":  #'o'
            marker = "d"
            color = "indianred"
            list_nll_plot = [models[model][h] for h in list_h]

        elif model_label == "Kernel-WTA":
            marker = "X"
            color = "indianred"
            list_nll_plot = [models[model][h] for h in list_h]

        for i, h in enumerate(list_h_plot):
            list_dic.append(
                {"h": list_h_plot[i], "model": model_label, "nll": list_nll_plot[i]}
            )

        color_per_model[model_label] = color
        marker_per_model[model_label] = marker

    df = pd.DataFrame(list_dic)
    df_hist_uniform = pd.DataFrame(list_hist_histogram_uniform)
    df_rmcl_uniform = pd.DataFrame(list_hist_rMCL_uniform)
    df["h"] = df["h"].astype(float)

    sns.lineplot(
        x="h",
        y="nll",
        data=df,
        markers=marker_per_model,
        hue="model",
        style="model",
        palette=color_per_model,
        dashes=False,
        markersize=12,
        linewidth=2.5,
        ax=ax,
    )
    sns.lineplot(
        x="h",
        y="nll",
        data=df_hist_uniform,
        color=COLORS["Histogram"],
        dashes=True,
        linestyle="--",
        label="Unif. Histogram",
        linewidth=2.5,
        ax=ax,
    )
    sns.lineplot(
        x="h",
        y="nll",
        data=df_rmcl_uniform,
        color=COLORS["Dirac Voronoi-WTA"],
        dashes=True,
        linestyle="--",
        label="Unif. Voronoi-WTA",
        linewidth=2.5,
        ax=ax,
    )

    # print(dataset)

    if dataset == "single-gaussian-not-centered":
        ax.set_title(f"$\\it{{Single\;Gaussian}}$", fontsize=20)
        ax.set_ylim(-0.4, 2)  # single gaussian
    elif dataset == "mixtures-uni-to-gaussians":
        ax.set_title(f"$\\it{{Uniform\;to\;Gaussians}}$", fontsize=20)
        ax.set_ylim(0.5, 4)  # mixture uni to gaussians
    elif dataset == "changing-damier":
        ax.set_title(f"$\\it{{Changing\;Damier}}$", fontsize=20)
        ax.set_ylim(1.2, 3)  # changing damier
    elif dataset == "rotating-two-moons":
        ax.set_title(f"$\\it{{Rotating\;Two\;Moons}}$", fontsize=20, pad=10)
        ax.set_ylim(0.45, 2.5)  # rotating moons

    ax.set_xlabel("Scaling factor $\\it{h}$", fontsize=20)
    ax.set_ylabel("NLL", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    # ax.set_ylim(-0.4,2) #single gaussian
    ax.grid(True, color="gray")
    # ax.legend(fontsize=15, prop={'size':20})

    h, l = ax.get_legend_handles_labels()

    ax.get_legend().remove()

    return h, l


# %% - MAKE SUBPLOTS

METRICS = {
    "nll": "NLL",
    "emd": "EMD",
    #    "risk": "Quantization error",
}
DATASETS = {
    "single-gaussian-not-centered": "Single Gaussian",
    "rotating-two-moons": "Rotating Two Moons",
    "changing-damier": "Changing Damier",
    "mixtures-uni-to-gaussians": "Uniform to Gaussians",
}

std_color_palette = sns.color_palette("muted")
wta_color_palette = sns.dark_palette("seagreen")

COLORS = {
    "Histogram": std_color_palette[1],
    "Dirac Voronoi-WTA": wta_color_palette[5],
    "MDN": std_color_palette[0],
    "Theoretical WTA": wta_color_palette[5],
    "Theoretical Histogram": std_color_palette[1],
    "Unif. Voronoi-WTA": wta_color_palette[1],
    "Voronoi-WTA h=0.1": wta_color_palette[4],
    "Voronoi-WTA h=0.2": wta_color_palette[3],
    "Voronoi-WTA h=0.3": wta_color_palette[2],
}

MARKERS = {
    "Histogram": "s",
    "Dirac Voronoi-WTA": "d",
    "MDN": "o",
    "Theoretical WTA": ".",
    "Theoretical Histogram": ".",
    "Unif. Voronoi-WTA": "H",
    "Voronoi-WTA h=0.1": "^",
    "Voronoi-WTA h=0.2": "X",
    "Voronoi-WTA h=0.3": "P",
}

TO_KEEP = [np.nan, np.infty, 0.0, 0.1, 0.2, 0.3]

METHODS_ORDER = [
    "Dirac Voronoi-WTA",
    "Voronoi-WTA h=0.1",
    "Voronoi-WTA h=0.2",
    "Voronoi-WTA h=0.3",
    "Unif. Voronoi-WTA",
    "MDN",
    "Histogram",
    "Theoretical WTA",
    "Theoretical Histogram",
]

TEXT_WIDTH = 3.25
PAGE_WIDTH = 6.875
FONTSIZE = 10


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
):

    if ax is None:
        _, ax = plt.subplots()

    if valid_scaling is not None:
        valid_scaling = [float(x) for x in valid_scaling]
        df = df[df["scaling_factor"].isin(valid_scaling)].reset_index(drop=True)
    else:
        valid_scaling = [np.nan]
        df = df[df["scaling_factor"].isin(valid_scaling)].reset_index(drop=True)

    ax = sns.lineplot(
        df,
        x="scaling_factor",
        y=metric,
        hue="model",
        style="model",
        markers=marker_per_model,
        dashes=False,
        ax=ax,
        palette=color_per_model,
        hue_order=models_order,
    )

    h, l = ax.get_legend_handles_labels()

    if dataset_name is not None:
        ax.set_title(dataset_name)
    ax.grid()

    if not xon:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Number of hypotheses")

    if not yon:
        ax.set_ylabel("")
    elif metric_names is not None:
        ax.set_ylabel(metric_names[metric])

    ax.get_legend().remove()

    return h, l


def map_to_row_col(i):
    if i == 0:
        return 0, 0
    elif i == 1:
        return 0, 1
    elif i == 2:
        return 1, 0
    elif i == 3:
        return 1, 1


METHODS_ORDER = [
    "Truncated-Kernel Histogram",
    "Voronoi-WTA",
    "Unweighted Kernel-WTA",
    "Kernel-WTA",
    "Unif. Histogram",
    "Unif. Voronoi-WTA",
]


def make_subplots(
    save_pth,
    base_dir,
    figsize=None,
    save=False,
    num_hypotheses=20,
    plot_unweighted=True,
    folder_name=None,
):
    fig, ax_arr = plt.subplots(
        2,
        2,
        sharex=True,
        figsize=figsize,
    )

    legend_handles = []
    legend_labels = []
    for i, dataset in enumerate(DATASETS):

        row, col = map_to_row_col(i)
        h, l = gen_plot(
            dataset=dataset,
            num_hypothesis=num_hypotheses,
            ax=ax_arr[row, col],
            plot_unweighted=plot_unweighted,
            base_dir=base_dir,
            folder_name=folder_name,
        )

        legend_handles += h
        legend_labels += l

    legend_idx, legend_labels = _merge_lists(legend_labels, METHODS_ORDER)
    legend_handles = [legend_handles[i] for i in legend_idx]

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),  # Adjust this tuple to shift the legend
    )
    fig.suptitle(
        "NLL vs $\\it{{h}}$ with {} hypotheses".format(num_hypotheses),
        fontsize=25,
        y=1.02,
    )
    if save:
        fig.savefig(
            os.path.join(save_pth, "secondary_plots_{}.pdf".format(num_hypotheses))
        )
        fig.savefig(
            os.path.join(save_pth, "secondary_plots_{}.png".format(num_hypotheses))
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_results", type=str, default="${MY_HOME}/results/")
    parser.add_argument("--save_path", type=str, default="${MY_HOME}/figures")
    parser.add_argument("--num_hypotheses", type=int, default=20)
    parser.add_argument("--plot_unweighted", type=bool, default=False)
    parser.add_argument("--folder_name", type=str, default="saved_csv")
    args = parser.parse_args()
    make_subplots(
        figsize=(PAGE_WIDTH * 2, PAGE_WIDTH * 1.3),
        save=True,
        save_pth=args.save_path,
        num_hypotheses=args.num_hypotheses,
        plot_unweighted=args.plot_unweighted,
        base_dir=args.base_dir_results,
        folder_name=args.folder_name,
    )
