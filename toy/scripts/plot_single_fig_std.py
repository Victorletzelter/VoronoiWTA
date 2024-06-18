import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

import pandas as pd
import seaborn as sns
import argparse


def str_to_list(l="[1, 2, 3]"):
    return list(map(int, l[1:-1].split(",")))


def hyp_hist(row):
    if "Histogram" in row["Name"]:
        list_sizes = str_to_list(row["model/sizes"])
        n = len(list_sizes)
        row["model/num_hypothesis"] = 1
        for i in range(n):
            row["model/num_hypothesis"] *= list_sizes[i]
    return row


def renaming(name):

    data_name = name.split("_model")[0]
    model_name = name.split("_model_")[1].split("_")[0]
    n_hyps = name.split("_hyps_")[0].split("_")[-1]

    if model_name == "VoronoiWTA":
        scaling_factor = name.split("_h_")[-1]
        if "uniform" not in name:
            return f"Voronoi-WTA h={scaling_factor}"
        else:
            return "Unif. Voronoi-WTA"

    elif model_name == "KernelWTA":
        scaling_factor = name.split("_h_")[-1]
        if "uniform" not in name:
            return f"Kernel-WTA h={scaling_factor}"
        else:
            return "Unif. Kernel-WTA"

    elif "UnweightedKernel" in model_name:
        scaling_factor = name.split("_h_")[-1]
        return f"Unweighted Kernel-WTA h={scaling_factor}"

    elif model_name == "Histogram":
        scaling_factor = name.split("_h_")[-1]
        if "uniform" not in name:
            return f"Truncated-Kernel Histogram h={scaling_factor}"
        else:
            return "Unif. Histogram"  # Corresponds to Unif. Histogram

    elif model_name == "GaussMix":
        return "MDN"


def build_paths(base_dir, dataset, folder_name):
    # base_dir = '${MY_HOME}/results/'
    # csv_files = os.listdir(os.path.join(base_dir,folder_name))
    csv_files = os.listdir(folder_name)
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


def construct_nll_dict(
    base_dir, dataset="changing-damier", num_hypotheses=20, folder_name=None
):
    csv_file = build_paths(base_dir=base_dir, dataset=dataset, folder_name=folder_name)[
        0
    ]
    # csv_file_2 = build_paths(base_dir=base_dir, dataset=dataset)[2]
    df = pd.read_csv(csv_file)
    # df = pd.concat([df,pd.read_csv(csv_file_2)], ignore_index=True, axis=0)
    df = df.apply(hyp_hist, axis=1)
    df["model"] = df["Name"].apply(renaming)
    df = df[df["model/num_hypothesis"] == num_hypotheses]
    df = df[~(df["model"].isna())]
    df = df[~(df["Name"].str.contains("h_opt"))]

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
                df_h = df_h.drop_duplicates(subset="Name", keep="first", inplace=False)
                assert df_h.shape[0] == 1
            models[model_key][h] = df_h["test_nll"].values[0]

    return models


def gen_plot(
    dataset,
    num_hypothesis,
    save_path,
    base_dir,
    plot_title=False,
    list_folder_names=None,
    plot_std=False,
):
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

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 2, PAGE_WIDTH))

    df = pd.DataFrame()
    df_hist_uniform = pd.DataFrame()
    df_rmcl_uniform = pd.DataFrame()

    for folder_name in list_folder_names:

        models = construct_nll_dict(
            base_dir=base_dir,
            dataset=dataset,
            num_hypotheses=num_hypothesis,
            folder_name=folder_name,
        )

        list_dic = []
        color_per_model = {}
        marker_per_model = {}

        list_hist_histogram_uniform = []
        color_histogram_uniform = {}
        list_hist_rMCL_uniform = []
        color_rmcl_uniform = {}

        # print(models)

        for model in ["Histogram", "rMCL", "kde_rMCL", "kde_weighted_rMCL"]:
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

        df_folder = pd.DataFrame(list_dic)
        df_hist_uniform_folder = pd.DataFrame(list_hist_histogram_uniform)
        df_rmcl_uniform_folder = pd.DataFrame(list_hist_rMCL_uniform)
        df_folder["h"] = df_folder["h"].astype(float)

        df = pd.concat([df, df_folder], ignore_index=True, axis=0)
        df_hist_uniform = pd.concat(
            [df_hist_uniform, df_hist_uniform_folder], ignore_index=True, axis=0
        )
        df_rmcl_uniform = pd.concat(
            [df_rmcl_uniform, df_rmcl_uniform_folder], ignore_index=True, axis=0
        )

    TO_KEEP_h_uniform = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    df_hist_uniform = df_hist_uniform[df_hist_uniform["h"].isin(TO_KEEP_h_uniform)]
    df_rmcl_uniform = df_rmcl_uniform[df_rmcl_uniform["h"].isin(TO_KEEP_h_uniform)]

    if plot_std is True:
        sns.lineplot(
            x="h",
            y="nll",
            errorbar="sd",
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
            errorbar="sd",
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
            errorbar="sd",
            data=df_rmcl_uniform,
            color=COLORS["Dirac Voronoi-WTA"],
            dashes=True,
            linestyle="--",
            label="Unif. Voronoi-WTA",
            linewidth=2.5,
            ax=ax,
        )

    if dataset == "single-gaussian-not-centered":
        if plot_title:
            plt.title(
                f"NLL vs $\\it{{h}}$ with {num_hypothesis} hypotheses on $\\it{{Single\;Gaussian}}$",
                fontsize=20,
            )
        ax.set_ylim(-0.4, 2)  # single gaussian
    elif dataset == "mixtures-uni-to-gaussians":
        if plot_title:
            plt.title(
                f"NLL vs $\\it{{h}}$ with {num_hypothesis} hypotheses on $\\it{{Uniform\;to\;Gaussians}}$",
                fontsize=20,
            )
        ax.set_ylim(0.5, 4)  # mixture uni to gaussians
    elif dataset == "changing-damier":
        if plot_title:
            plt.title(
                f"NLL vs $\\it{{h}}$ with {num_hypothesis} hypotheses on $\\it{{Changing\;Damier}}$",
                fontsize=20,
            )
        ax.set_ylim(1.2, 3)  # changing damier
    elif dataset == "rotating-two-moons":
        if plot_title:
            plt.title(
                f"NLL vs $\\it{{h}}$ with {num_hypothesis} hypotheses on $\\it{{Rotating\;Two\;Moons}}$",
                fontsize=20,
                pad=10,
            )
        ax.set_ylim(0.45, 2.5)  # rotating moons

    ax.set_xlabel("Scaling factor $\\it{h}$", fontsize=20)
    ax.set_ylabel("NLL", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    # ax.set_ylim(-0.4,2) #single gaussian
    ax.grid(True, color="gray")
    ax.legend(fontsize=15, prop={"size": 20})

    # savefig
    plt.savefig(
        os.path.join(save_path, f"NLL-vs-h-{num_hypothesis}hyp-dataset-{dataset}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(save_path, f"NLL-vs-h-{num_hypothesis}hyp-dataset-{dataset}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # Show plot
    plt.show()
    plt.close()


if __name__ == "__main__":
    # dataset = 'mixtures-uni-to-gaussians'
    # dataset = "single-gaussian-not-centered"
    # dataset = "rotating-two-moons"
    # dataset = "changing-damier"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_results", type=str, default="${MY_HOME}/results/")
    parser.add_argument("--save_path", type=str, default="${MY_HOME}/figures_std")
    parser.add_argument("--num_hypotheses", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="mixtures-uni-to-gaussians")
    parser.add_argument("--plot_title", type=bool, default=False)
    # parser.add_argument("--list_folder_names", type=str, default=['saved_csv','saved_csv_seed1244','saved_csv_seed1245'])
    parser.add_argument(
        "--list_folder_names", type=str, default="${MY_HOME}/results/considered_csv"
    )
    parser.add_argument("--plot_std", type=str, default="True")
    parser.add_argument("--considered_seeds", type=str, default="12345,1244,1245")
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
    list_folder_names_considered = []

    for i in range(len(list_folder_names)):
        if list_folder_names[i].split("seed")[-1] in args.considered_seeds.split(","):
            list_folder_names_considered.append(
                os.path.join(args.list_folder_names, list_folder_names[i])
            )

    gen_plot(
        dataset=args.dataset,
        num_hypothesis=args.num_hypotheses,
        save_path=args.save_path,
        base_dir=args.base_dir_results,
        plot_title=args.plot_title,
        list_folder_names=list_folder_names_considered,
        plot_std=args.plot_std,
    )
