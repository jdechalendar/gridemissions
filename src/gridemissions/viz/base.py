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


def add_watermark(ax, x=0.99, y=0.98, ha="right", va="top", fontsize="x-small"):
    ax.text(
        x,
        y,
        "energy.stanford.edu/gridemissions",
        ha=ha,
        va=va,
        fontsize=fontsize,
        transform=ax.transAxes,
        bbox={
            "facecolor": "w",
            "ec": "k",
            "lw": 0.25,
            "alpha": 0.8,
            "boxstyle": "round,pad=.2",
        },
    )


def heatmap(
    s,
    fax=None,
    fill_value=None,
    nxticks=8,
    nyticks=6,
    cbar_label=None,
    scaling=1.0,
    vmin=None,
    vmax=None,
    with_cbar=True,
    transpose=True,
    cmap="viridis",
    cbar_pad=0.05,
    cbar_ax=None,
    remove_day_label=True,
):
    """
        Plot a heatmap from time series data.

        Usage for dataframe df with timestamps in column "date_time" and data
        to plot in column "col" (grouped by frequency 'freq':
            df_hmap = df.groupby(pd.Grouper(key='date_time', freq=freq, axis=1))\
                .mean().fillna(method='pad',limit=fillna_limit)
            plotting.plotHeatmap(df_hmap, col)

        Parameters
        ----------
        s: pd.Series or pd.DataFrame
            data to plot, if pd.DataFrame, expects one column only
        fax: default None
            (figure, axes) handles returned from a call to e.g. plt.subplots
        fill_value: float, default None
            the value to use to fill holes
        title:str, default None
            plot title
        nxticks: int, default 8
            number of xticks for the plot
        nyticks: int, default 10
            number of yticks for the plot
        cbar_label: str, default None
            label for colorbar, defaults to series name. Use a string with a leading "_" for no label
        scaling: float
            multiplier to rescale the data
        vmin: float
            min value for the colorbar
        vmax: float
            max value for the colorbar
        cmap: str, default "viridis"
            color map
        cbar_pad: float, default 0.05
            padding between colorbar and figure
        remove_day_label: bool, default True
    """
    if type(s) == pd.DataFrame:
        if len(s.columns) != 1:
            raise ValueError(f"Expecting 1 column but got: {len(s.columns)}")
        s = s[s.columns[0]]

    if fill_value is None:
        fill_value = s.min()
    if cbar_label is None:
        cbar_label = s.name
    elif cbar_label.startswith("_"):
        cbar_label = None
    if vmin is None:
        vmin = s.min()
    if vmax is None:
        vmax = s.max()
    if transpose:
        xlabel = "" if remove_day_label else "day"

        def xtickformatter(el):
            return el.strftime("%b-%y")

        ylabel = "Hour of day"

        def ytickformatter(el):
            return el.hour

        cbar_orientation = "horizontal"

    else:
        xlabel = "Hour of day"

        def xtickformatter(el):
            return el.hour

        ylabel = "" if remove_day_label else "day"

        def ytickformatter(el):
            return el.strftime("%m-%y")

        cbar_orientation = "vertical"

    df_heatmap = pd.DataFrame(
        {
            "date": s.index.date,
            "time": s.index.time,
            "value_col": s.values * scaling,
        }
    )

    df_heatmap = df_heatmap.pivot(index="date", columns="time", values="value_col")
    df_heatmap.fillna(value=fill_value, inplace=True)  # fill holes for plotting

    if fax is None:
        f, ax = plt.subplots()
    else:
        f, ax = fax

    if transpose:
        df_heatmap = df_heatmap.transpose()

    mappable = ax.pcolor(df_heatmap, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.invert_yaxis()
    nyticks = min(len(df_heatmap.index), nyticks)
    nxticks = min(len(df_heatmap.columns), nxticks)
    yticks = range(0, len(df_heatmap.index), int(len(df_heatmap.index) / nyticks))
    xticks = range(0, len(df_heatmap.columns), int(len(df_heatmap.columns) / nxticks))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([xtickformatter(el) for el in df_heatmap.columns[xticks]])
    ax.set_yticklabels([ytickformatter(el) for el in df_heatmap.index[yticks]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if with_cbar:
        f.colorbar(
            mappable,
            ax=cbar_ax,
            label=cbar_label,
            orientation=cbar_orientation,
            pad=cbar_pad,
        )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    return f, ax
