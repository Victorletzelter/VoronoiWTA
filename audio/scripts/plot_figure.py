import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Extract plots from csv file results.")
parser.add_argument(
    "--path_csv_file",
    type=str,
    default=os.path.join(CSV_DIR, os.listdir(CSV_DIR)[0]),
    help="Path of the csv file containing the results.",
)
parser.add_argument(
    "--metric", type=str, default="test_mean_nll_metric", help="Metric to plot."
)
parser.add_argument(
    "--save_dir", type=str, default=".", help="Directory to save the plots."
)
args = parser.parse_args()


def generate_results(path_csv_file, num_hypothesis, metric):

    cols = [
        "Name",
        metric,
        "model/hparams/scaling_factor",
        "model/hparams/num_hypothesis",
        "model/hparams/kde_weighted",
        "model/hparams/kde_mode",
        "model/hparams/kernel_type",
        "model/hparams/name",
        "model/hparams/nrows",
        "model/hparams/ncols",
    ]

    cols_kde = [
        "Name",
        metric,
        "model/hparams/scaling_factor",
        "model/hparams/num_hypothesis",
        "model/hparams/kde_weighted",
        "model/hparams/kde_mode",
        "model/hparams/kernel_type",
        "model/hparams/name",
    ]

    cols_mdn = ["Name", metric, "model/hparams/num_modes", "model/hparams/name"]

    csv_file = pd.read_csv(path_csv_file, usecols=cols)
    csv_file_kde = pd.read_csv(path_csv_file, usecols=cols_kde)
    csv_file_mdn = pd.read_csv(path_csv_file, usecols=cols_mdn)

    x_mdn = csv_file_mdn["Name"][csv_file_mdn["Name"].str.contains("not")][
        csv_file_mdn["model/hparams/num_modes"] == num_hypothesis
    ][csv_file_mdn["Name"].str.contains("image") == False]
    y_mdn = csv_file_mdn[metric][csv_file_mdn["Name"].str.contains("not")][
        csv_file_mdn["model/hparams/num_modes"] == num_hypothesis
    ][csv_file_mdn["Name"].str.contains("image") == False]

    # convert string to float
    x5_original = csv_file["model/hparams/kernel_type"][
        csv_file["model/hparams/kde_weighted"] == False
    ][csv_file["model/hparams/kde_mode"] == False][
        csv_file["model/hparams/nrows"] * csv_file["model/hparams/ncols"]
        == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "uniform"
    ][
        csv_file["model/hparams/name"] == "MHistogramCONFSELDNet"
    ]
    x6_original = csv_file["model/hparams/scaling_factor"][
        csv_file["model/hparams/kde_weighted"] == False
    ][csv_file["model/hparams/kde_mode"] == False][
        csv_file["model/hparams/nrows"] * csv_file["model/hparams/ncols"]
        == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file["model/hparams/name"] == "MHistogramCONFSELDNet"
    ]

    x1_original = csv_file["model/hparams/scaling_factor"][
        csv_file["model/hparams/kde_weighted"] == False
    ][csv_file["model/hparams/kde_mode"] == False][
        csv_file["model/hparams/num_hypothesis"] == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file["model/hparams/scaling_factor"] != 2.0
    ][
        csv_file["model/hparams/scaling_factor"] != 3.0
    ]
    x2_original = csv_file_kde["model/hparams/scaling_factor"][
        csv_file_kde["model/hparams/kde_weighted"] == True
    ][csv_file_kde["model/hparams/kde_mode"] == True][
        csv_file_kde["model/hparams/num_hypothesis"] == num_hypothesis
    ][
        csv_file_kde["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file_kde["model/hparams/scaling_factor"] != 2.0
    ]
    x3_original = csv_file["model/hparams/kernel_type"][
        csv_file["model/hparams/kde_weighted"] == False
    ][csv_file["model/hparams/kde_mode"] == False][
        csv_file["model/hparams/num_hypothesis"] == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "uniform"
    ]
    x4_original = csv_file_kde["model/hparams/scaling_factor"][
        csv_file_kde["model/hparams/kde_weighted"] == False
    ][csv_file_kde["model/hparams/kde_mode"] == True][
        csv_file_kde["model/hparams/num_hypothesis"] == num_hypothesis
    ][
        csv_file_kde["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file_kde["model/hparams/scaling_factor"] != str(2.0)
    ]
    x5_original = csv_file["model/hparams/kernel_type"][
        csv_file["model/hparams/kde_weighted"] == False
    ][csv_file["model/hparams/kde_mode"] == False][
        csv_file["model/hparams/nrows"] * csv_file["model/hparams/ncols"]
        == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "uniform"
    ][
        csv_file["model/hparams/name"] == "MHistogramCONFSELDNet"
    ]
    x6_original = csv_file["model/hparams/scaling_factor"][
        csv_file["model/hparams/kde_weighted"] == False
    ][csv_file["model/hparams/kde_mode"] == False][
        csv_file["model/hparams/nrows"] * csv_file["model/hparams/ncols"]
        == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file["model/hparams/name"] == "MHistogramCONFSELDNet"
    ][
        csv_file["model/hparams/scaling_factor"] != 2.0
    ]

    y1_original = csv_file[metric][csv_file["model/hparams/kde_weighted"] == False][
        csv_file["model/hparams/kde_mode"] == False
    ][csv_file["model/hparams/num_hypothesis"] == num_hypothesis][
        csv_file["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file["model/hparams/scaling_factor"] != 2.0
    ][
        csv_file["model/hparams/scaling_factor"] != 3.0
    ]
    y2_original = csv_file_kde[metric][
        csv_file_kde["model/hparams/kde_weighted"] == True
    ][csv_file_kde["model/hparams/kde_mode"] == True][
        csv_file_kde["model/hparams/num_hypothesis"] == num_hypothesis
    ][
        csv_file_kde["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file_kde["model/hparams/scaling_factor"] != 2.0
    ]
    y3_original = csv_file[metric][csv_file["model/hparams/kde_weighted"] == False][
        csv_file["model/hparams/kde_mode"] == False
    ][csv_file["model/hparams/num_hypothesis"] == num_hypothesis][
        csv_file["model/hparams/kernel_type"] == "uniform"
    ]
    y4_original = csv_file_kde[metric][
        csv_file_kde["model/hparams/kde_weighted"] == False
    ][csv_file_kde["model/hparams/kde_mode"] == True][
        csv_file_kde["model/hparams/num_hypothesis"] == num_hypothesis
    ][
        csv_file_kde["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file_kde["model/hparams/scaling_factor"] != str(2.0)
    ]
    y5_original = csv_file[metric][csv_file["model/hparams/kde_weighted"] == False][
        csv_file["model/hparams/kde_mode"] == False
    ][
        csv_file["model/hparams/nrows"] * csv_file["model/hparams/ncols"]
        == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "uniform"
    ][
        csv_file["model/hparams/name"] == "MHistogramCONFSELDNet"
    ]
    y6_original = csv_file[metric][csv_file["model/hparams/kde_weighted"] == False][
        csv_file["model/hparams/kde_mode"] == False
    ][
        csv_file["model/hparams/nrows"] * csv_file["model/hparams/ncols"]
        == num_hypothesis
    ][
        csv_file["model/hparams/kernel_type"] == "von_mises_fisher"
    ][
        csv_file["model/hparams/name"] == "MHistogramCONFSELDNet"
    ][
        csv_file["model/hparams/scaling_factor"] != 2.0
    ]

    x1_clean = x1_original.dropna().astype(float)
    x2_clean = x2_original.dropna().astype(float)
    x3_clean = x3_original.dropna()
    x4_clean = x4_original.dropna().astype(float)
    x5_clean = x5_original.dropna()
    x6_clean = x6_original.dropna().astype(float)
    y1_clean = y1_original[x1_original.notna()].astype(float)
    y2_clean = y2_original[x2_original.notna()].astype(float)
    y3_clean = y3_original[x3_original.notna()].astype(float)
    y4_clean = y4_original[x4_original.notna()].astype(float)
    y5_clean = y5_original[x5_original.notna()].astype(float)
    y6_clean = y6_original[x6_original.notna()].astype(float)

    # Remove duplicates from x2 while keeping associated y2 values
    if len(x1_clean) > 0:
        # Separate float and non-float items
        float_items = {}
        non_float_items = {}

        for x, y in zip(x1_clean, y1_clean):
            if isinstance(x, float):  # Check if x is a float
                float_items[x] = y
            else:
                non_float_items[x] = y

        # Sort the float items
        sorted_float_items = sorted(float_items.items())

        # Combine sorted float items with non-float items
        final_sorted_items = sorted_float_items + list(non_float_items.items())

        # Unzip keys and values
        x1, y1 = zip(*final_sorted_items)

    else:
        x1 = []
        y1 = []

    # Remove duplicates from x2 while keeping associated y2 values
    unique_x2_y2 = {}  # Dictionary to store unique x2 values and associated y2 values
    if len(x2_clean) > 0:

        float_items = {}
        non_float_items = {}

        for x, y in zip(x2_clean, y2_clean):
            if isinstance(x, float):  # Check if x is a float
                float_items[x] = y
            else:
                non_float_items[x] = y

        # Sort the float items
        sorted_float_items = sorted(float_items.items())

        # Combine sorted float items with non-float items
        final_sorted_items = sorted_float_items + list(non_float_items.items())

        # Unzip keys and values
        x2, y2 = zip(*final_sorted_items)

    else:
        x2 = []
        y2 = []

    # Remove duplicates from x2 while keeping associated y2 values
    unique_x3_y3 = {}  # Dictionary to store unique x2 values and associated y2 values
    if len(x3_clean) > 0:

        float_items = {}
        non_float_items = {}

        for x, y in zip(x3_clean, y3_clean):
            if isinstance(x, float):  # Check if x is a float
                float_items[x] = y
            else:
                non_float_items[x] = y

        # Sort the float items
        sorted_float_items = sorted(float_items.items())

        # Combine sorted float items with non-float items
        final_sorted_items = sorted_float_items + list(non_float_items.items())

        # Unzip keys and values
        x3, y3 = zip(*final_sorted_items)

    else:
        x3 = []
        y3 = []

    # Remove duplicates from x2 while keeping associated y2 values
    unique_x4_y4 = {}  # Dictionary to store unique x4 values and associated y4 values
    if len(x4_clean) > 0:

        float_items = {}
        non_float_items = {}

        for x, y in zip(x4_clean, y4_clean):
            if isinstance(x, float):  # Check if x is a float
                float_items[x] = y
            else:
                non_float_items[x] = y

        # Sort the float items
        sorted_float_items = sorted(float_items.items())

        # Combine sorted float items with non-float items
        final_sorted_items = sorted_float_items + list(non_float_items.items())

        # Unzip keys and values
        x4, y4 = zip(*final_sorted_items)
    else:
        x4 = []
        y4 = []

    if len(x5_clean) > 0:

        float_items = {}
        non_float_items = {}

        for x, y in zip(x5_clean, y5_clean):
            if isinstance(x, float):  # Check if x is a float
                float_items[x] = y
            else:
                non_float_items[x] = y

        # Sort the float items
        sorted_float_items = sorted(float_items.items())

        # Combine sorted float items with non-float items
        final_sorted_items = sorted_float_items + list(non_float_items.items())

        # Unzip keys and values
        x5, y5 = zip(*final_sorted_items)
    else:
        x5 = []
        y5 = []

    if len(x6_clean) > 0:

        float_items = {}
        non_float_items = {}

        for x, y in zip(x6_clean, y6_clean):
            if isinstance(x, float):  # Check if x is a float
                float_items[x] = y
            else:
                non_float_items[x] = y

        # Sort the float items
        sorted_float_items = sorted(float_items.items())

        # Combine sorted float items with non-float items
        final_sorted_items = sorted_float_items + list(non_float_items.items())

        # Unzip keys and values
        x6, y6 = zip(*final_sorted_items)

    else:
        x6 = []
        y6 = []

    return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x_mdn, y_mdn


def plotting(
    ax,
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    x4,
    y4,
    x5,
    y5,
    x6,
    y6,
    x_mdn,
    y_mdn,
    set_x_label=True,
    set_y_label=True,
    set_legend=True,
    show_x_ticks=True,
    show_y_ticks=True,
):
    # Set the aesthetic style of the plots
    # sns.set_theme(style='white')
    # sns.set_style("white", {"grid.color": 'black'})

    sns.set_style("white", {"xtick.bottom": True, "ytick.left": True})

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    # Create a figure with a specific size

    TEXT_WIDTH = 3.25
    PAGE_WIDTH = 6.875
    FONTSIZE = 12

    # Plotting the data using Seaborn lineplot
    sns.lineplot(
        x=x1,
        y=y1,
        label="Voronoi-WTA",
        marker="^",
        color=COLORS["Dirac Voronoi-WTA"],
        markersize=FONTSIZE,
        ax=ax,
        linewidth=2.5,
    )
    sns.lineplot(
        x=x2,
        y=y2,
        label="Kernel-WTA",
        marker="X",
        color="indianred",
        markersize=FONTSIZE,
        ax=ax,
        linewidth=2.5,
    )
    sns.lineplot(
        x=x4,
        y=y4,
        label="Unweighted Kernel-WTA",
        marker="d",
        color="indianred",
        markersize=FONTSIZE,
        ax=ax,
        linewidth=2.5,
    )
    sns.lineplot(
        x=x6,
        y=y6,
        label="Truncated-Kernel-Histogram",
        marker="s",
        color=COLORS["Histogram"],
        markersize=FONTSIZE,
        ax=ax,
        linewidth=2.5,
    )

    # For the constant lines, we will plot them using matplotlib since they require duplication of x values
    ax.plot(
        x2,
        [y3] * len(x2),
        label="Unif. Voronoi-WTA",
        linestyle="--",
        color=COLORS["Dirac Voronoi-WTA"],
        linewidth=2.5,
    )
    ax.plot(
        x2,
        [y5] * len(x2),
        label="Unif. Histogram",
        linestyle="--",
        color=COLORS["Histogram"],
        linewidth=2.5,
    )
    ax.plot(
        x2,
        [y_mdn] * len(x2),
        label="von Mises-Fisher Mixture",
        linestyle="--",
        color=COLORS["MDN"],
        linewidth=2.5,
    )

    # Adding the title and labels

    if metric == "test_mean_nll_metric":
        ax.set_title(f"{num_hypothesis} hypotheses", fontsize=20)
        if set_y_label is True:
            ax.set_ylabel("NLL", fontsize=20)
    elif metric == "test_mean_wta_risk":
        ax.set_title(
            f"WTA Risk vs $\\it{{h}}$ with {num_hypothesis} hypotheses", fontsize=20
        )
        if set_y_label is True:
            ax.set_ylabel("WTA Risk", fontsize=20)

    if set_x_label is True:
        ax.set_xlabel("Scaling factor $\\it{h}$", fontsize=20)

    ax.grid(True)

    ax.get_legend().remove()

    if show_x_ticks is False:
        ax.xaxis.set_tick_params(labelbottom=False)  # Hides the x-axis labels
        ax.xaxis.set_tick_params(bottom=False)  # Hides the x-axis ticks


TEXT_WIDTH = 3.25
PAGE_WIDTH = 10
FONTSIZE = 12

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig, ax = plt.subplots(2, 2, figsize=(PAGE_WIDTH * 2, PAGE_WIDTH))

path_csv_file = args.path_csv_file

metric = args.metric
# metric = 'test_mean_nll_metric'
# metric = 'test_mean_wta_risk'


def n_hyp_to_ij(n_hyp):
    if n_hyp == 9:
        return 0, 0
    elif n_hyp == 16:
        return 0, 1
    elif n_hyp == 20:
        return 1, 0
    elif n_hyp == 25:
        return 1, 1


for num_hypothesis in [9, 16, 20, 25]:
    i, j = n_hyp_to_ij(num_hypothesis)
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x_mdn, y_mdn = generate_results(
        path_csv_file, num_hypothesis, metric
    )
    plotting(
        ax[i, j],
        x1,
        y1,
        x2,
        y2,
        x3,
        y3,
        x4,
        y4,
        x5,
        y5,
        x6,
        y6,
        x_mdn,
        y_mdn,
        set_x_label=(i == 1),
        set_y_label=(j == 0),
        set_legend=(i == 1 and j == 1),
        show_x_ticks=(i == 1),
        show_y_ticks=(j == 0),
    )
    ax[i, j].tick_params(axis="both", which="major", labelsize=20)

    print(x_mdn)
    print(y_mdn)

# Create the legend from the first subplot's handle and label
handles, labels = ax[0, 0].get_legend_handles_labels()

# Position the legend outside the subplot on the right
# 'bbox_to_anchor' specifies the (x, y) position of the legend's anchor point relative to the figure
# 'loc' specifies which part of the legend box is at the anchor point
fig.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(0.4, 0.65),
    bbox_transform=plt.gcf().transFigure,
    fontsize=18,
)

# Adjust layout to make room for the legend
plt.tight_layout()

plt.gca()

plt.savefig(os.path.join(args.save_dir, "NLL_vs_h.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(args.save_dir, "NLL_vs_h.pdf"), dpi=300, bbox_inches="tight")

# Show plot
plt.show()
plt.close()
