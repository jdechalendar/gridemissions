import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from cycler import cycler


PAGE_WIDTH = 6.99  # in
ROW_HEIGHT = 2.5  # in
COLORS = sns.color_palette("colorblind")


def set_plots(
    use=None, style="small", SMALL_SIZE=None, MEDIUM_SIZE=None, BIGGER_SIZE=None
):
    """
    Set plots

    Returns
    -------
    COLORS
    PAGE_WIDTH
    ROW_HEIGHT

    """
    if style == "big":
        SMALL_SIZE = 22
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 26
    if style == "small":
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

    if SMALL_SIZE is None:
        SMALL_SIZE = 16

    if MEDIUM_SIZE is None:
        MEDIUM_SIZE = 18

    if BIGGER_SIZE is None:
        BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    pd.plotting.register_matplotlib_converters()
    plt.rcParams.update(
        {
            "figure.figsize": [PAGE_WIDTH / 2, ROW_HEIGHT],
            "grid.color": "k",
            "axes.grid": True,
            "axes.prop_cycle": cycler("color", COLORS),
        }
    )
    if use is None:
        plt.rcParams.update(
            {
                "grid.linestyle": ":",
                "grid.linewidth": 0.5,
                "figure.dpi": 200,
            }
        )

    elif use == "notebook":
        plt.rcParams.update(
            {
                "figure.figsize": [PAGE_WIDTH, ROW_HEIGHT],
                "grid.color": "k",
                "axes.grid": True,
                # 'grid.linestyle': ':',
                "grid.linewidth": 0.2,
                "grid.color": "grey",
                "figure.dpi": 300,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "savefig.transparent": True,
                "legend.fontsize": 7,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "legend.markerscale": 3.0,
            }
        )

    return (COLORS, PAGE_WIDTH, ROW_HEIGHT)
