# %%
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# %% - mlflow csv extraction


# %% - PATH BUILDER


def build_paths(metric, dataset):
    base = Path("./pickle_results")
    mbase = base / f"metrics-{dataset}"
    paths = list()
    if metric in ["emd", "nll"]:
        path_folder = mbase / f"{metric}_vs_n_hyps_updated_results"
        for file in os.listdir(path_folder):
            if file.endswith(".pickle"):
                paths.append(str(path_folder / file))
    elif metric == "risk":
        main_path = mbase / "pickle_risk"
        main_path /= f"risk_vs_num_hyps_dataset_{dataset}.pickle"
        zador_path = base / "log_dict_1.pkl"
        paths = [str(main_path), str(zador_path)]
    else:
        raise ValueError(f"unknown metric {metric}")

    return paths


# %% - NEW NLL EXTRACTION FUNCTION (PANDAS)


def extract_nll_to_df(
    list_file_path,
):
    """Assuming file_path = 'data_saved/emd_scores_vs_number_hypothesis.pickle
    with a dict in the form:
    {'5': {'rMCL': <value>, 'Histogram': <value>},
    '10': {'rMCL': <value>, 'Histogram': <value>},
    ...
    """

    nll_scores_statistics = []

    for file_path in list_file_path:
        with open(file_path, "rb") as f:
            nll_scores = pickle.load(f)

        file_path_hist = file_path.replace("updated_results", "updated_new_hist")

        if os.path.exists(file_path_hist):
            with open(file_path_hist, "rb") as f:
                hist = pickle.load(f)

        if "scaling" in file_path:
            scaling_factor = (
                file_path.split(".pickle")[-2].split("/")[-1].split("_")[-1]
            )

            for (
                nhyps
            ) in nll_scores.keys():  # the key corresponds to the number of hypotheses
                if "rMCL" in nll_scores[nhyps].keys():
                    nll_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": float(scaling_factor),
                            "old_model_name": f"rMCL_{scaling_factor}",
                            "model": f"Voronoi-WTA h={scaling_factor}",
                            "nll": nll_scores[nhyps]["rMCL"],
                        }
                    )

                if "gauss_mix" in nll_scores[nhyps].keys():
                    nll_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": np.nan,
                            "old_model_name": "gauss_mix",
                            "model": "MDN",
                            "nll": nll_scores[nhyps]["gauss_mix"],
                        }
                    )

        elif "unifkernel" in file_path:
            for nhyps in nll_scores.keys():
                if "rMCL" in nll_scores[nhyps].keys():
                    nll_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": np.infty,
                            "old_model_name": "rMCL_unifkernel",
                            "model": "Unif. Voronoi-WTA",
                            "nll": nll_scores[nhyps]["rMCL"],
                        }
                    )
                if "Histogram" in nll_scores[nhyps].keys():
                    if nhyps in hist.keys():
                        nll_scores_statistics.append(
                            {
                                "n_hyps": int(nhyps),
                                "scaling_factor": np.nan,
                                "old_model_name": "Histogram_unifkernel",
                                "model": "Histogram",
                                "nll": hist[nhyps]["Histogram"],
                            }
                        )

    return pd.DataFrame(nll_scores_statistics)


# %% - NEW EMD EXTRACTION FUNCTION (PANDAS)
def extract_emd_to_df(list_file_path):
    """Assuming file_path = 'data_saved/emd_scores_vs_number_hypothesis.pickle
    with a dict in the form:
    {'5': {'rMCL': <value>, 'Histogram': <value>},
    '10': {'rMCL': <value>, 'Histogram': <value>},
    ...
    """
    emd_scores_statistics = list()

    for file_path in list_file_path:

        with open(file_path, "rb") as f:
            # print(file_path)
            emd_scores = pickle.load(f)

        # replace updated_result by updated_new_hist in file_path
        file_path_hist = file_path.replace("updated_results", "updated_new_hist")

        hist_active = False
        if os.path.exists(file_path_hist):
            hist_active = True
            with open(file_path_hist, "rb") as f:
                hist = pickle.load(f)

        if "scaling" in file_path:
            scaling_factor = (
                file_path.split(".pickle")[-2].split("/")[-1].split("_")[-1]
            )

            for (
                nhyps
            ) in emd_scores.keys():  # the key corresponds to the number of hypotheses
                if "rMCL" in emd_scores[nhyps].keys():
                    emd_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": float(scaling_factor),
                            "old_model_name": f"rMCL_{scaling_factor}",
                            "model": f"Voronoi-WTA h={scaling_factor}",
                            "emd": emd_scores[nhyps]["rMCL"],
                        }
                    )

                if (
                    hist_active is True
                    and nhyps in hist
                    and "Histogram" in hist[nhyps].keys()
                ):
                    emd_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": np.nan,
                            "old_model_name": "Histogram",
                            "model": "Histogram",
                            "emd": hist[nhyps]["Histogram"],
                        }
                    )
                if "gauss_mix" in emd_scores[nhyps].keys():
                    emd_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": np.nan,
                            "old_model_name": "gauss_mix",
                            "model": "MDN",
                            "emd": emd_scores[nhyps]["gauss_mix"],
                        }
                    )

        elif "unifkernel" in file_path:
            for nhyps in emd_scores.keys():
                if "rMCL" in emd_scores[nhyps].keys():
                    emd_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": np.infty,
                            "old_model_name": "rMCL_unifkernel",
                            "model": "Unif. Voronoi-WTA",
                            "emd": emd_scores[nhyps]["rMCL"],
                        }
                    )
        elif "dirackernel" in file_path:
            for nhyps in emd_scores.keys():
                if "rMCL" in emd_scores[nhyps].keys():
                    emd_scores_statistics.append(
                        {
                            "n_hyps": int(nhyps),
                            "scaling_factor": 0.0,
                            "old_model_name": "rMCL_dirackernel",
                            "model": "Dirac Voronoi-WTA",
                            "emd": emd_scores[nhyps]["rMCL"],
                        }
                    )
    return pd.DataFrame(emd_scores_statistics)


# %% - NEW RISK EXTRACTION FUNCTION (PANDAS)


def extract_risk_to_df(paths_list, dataset):
    """Assuming file_path = 'data_saved/NLL_scores_vs_number_hypothesis.pickle
    with a dict in the form:
    {'5': {'rMCL': <value>, 'Histogram': <value>},
    '10': {'rMCL': <value>, 'Histogram': <value>},
    ..."""

    main_path, zador_path = paths_list

    with open(main_path, "rb") as f:
        risk_values = pickle.load(f)

    with open(zador_path, "rb") as f:
        zador_values = pickle.load(f)

    zador_ds_map = {
        "changing-damier": "changing_damier",
        "rotating-two-moons": "rotating_moons",
        "single-gaussian-not-centered": "uncentered_gaussian",
    }

    file_path_hist = main_path.replace("updated_results", "updated_new_hist")
    with open(file_path_hist, "rb") as f:
        risk_values_hist = pickle.load(f)

    df = list()
    zador_df = list()

    for nhyps in risk_values.keys():
        df.append(
            {
                "n_hyps": int(nhyps),
                "old_model_name": "rMCL",
                "model": "Dirac Voronoi-WTA",
                "risk": float(risk_values[nhyps]["rMCL"][0]),
            }
        )
        df.append(
            {
                "n_hyps": int(nhyps),
                "old_model_name": "Histogram",
                "model": "Histogram",
                "risk": float(risk_values_hist[nhyps]["Histogram"][0]),
            }
        )
        df.append(
            {
                "n_hyps": int(nhyps),
                "old_model_name": "gauss_mix",
                "model": "MDN",
                "risk": float(risk_values[nhyps]["gauss_mix"][0]),
            }
        )
        zador_path = f"{zador_ds_map[dataset]}/zador/{nhyps}"
        zador_df.append(
            {
                "n_hyps": int(nhyps),
                "old_model_name": "zador",
                "model": "Theoretical WTA",
                "risk": float(zador_values[zador_path]),
            }
        )
        zador_path = f"{zador_ds_map[dataset]}/histogram/{nhyps}"
        zador_df.append(
            {
                "n_hyps": int(nhyps),
                "old_model_name": "histogram",
                "model": "Theoretical Histogram",
                "risk": float(zador_values[zador_path]),
            }
        )

    return pd.DataFrame(df), pd.DataFrame(zador_df)


# %% - NEW GENERAL SINGLE PLOTTING FUNC


def single_plot(
    df,
    metric,
    df_theory=None,
    metric_names=None,
    dataset_name=None,
    models_order=None,
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
        df = df.query("scaling_factor in @valid_scaling").reset_index(drop=True)

    ax = sns.lineplot(
        df,
        x="n_hyps",
        y=metric,
        hue="model",
        style="model",
        markers=marker_per_model,
        dashes=False,
        ax=ax,
        palette=color_per_model,
        hue_order=models_order,
    )

    # Plot theoretical curves when available in dashed lines
    if df_theory is not None:
        for model_name in df_theory["model"].unique():
            df_theory_model = df_theory.query("model == @model_name").reset_index(
                drop=True
            )

            ax.plot(
                df_theory_model["n_hyps"],
                df_theory_model[metric],
                linestyle="--",
                color=color_per_model[model_name],
                label=model_name,
            )

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

    h, l = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    return h, l


# %% - MAKE SUBPLOTS

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
    "nll": extract_nll_to_df,
    "emd": extract_emd_to_df,
    "risk": extract_risk_to_df,
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

# %%


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


def make_subplots(figsize=None, save=False):
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
            paths = build_paths(metric, dataset)

            if metric != "risk":
                df = extract_func(paths)
                df_zador = None
            else:
                df, df_zador = extract_func(paths, dataset=dataset)

            plot_specific_order = [
                m for m in METHODS_ORDER if m in df["model"].unique()
            ]
            if df_zador is not None:
                plot_specific_order += [
                    m for m in METHODS_ORDER if m in df_zador["model"].unique()
                ]

            h, l = single_plot(
                df,
                metric=metric,
                df_theory=df_zador,
                metric_names=METRICS,
                dataset_name=DATASETS[dataset] if row == 0 else None,
                color_per_model=COLORS,
                marker_per_model=MARKERS,
                models_order=plot_specific_order,
                valid_scaling=TO_KEEP if metric != "risk" else None,
                # xon=row==2,
                yon=col == 0,
                ax=ax_arr[row, col],
            )
            legend_handles += h
            legend_labels += l

    legend_idx, legend_labels = _merge_lists(legend_labels, METHODS_ORDER)
    legend_handles = [legend_handles[i] for i in legend_idx]

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(legend_handles, legend_labels, ncol=5, loc="upper center")
    if save:
        fig.savefig("figures/main_plots.pdf")
        fig.savefig("figures/main_plots.png")


# %%
make_subplots((PAGE_WIDTH * 2, PAGE_WIDTH), True)

# %%
