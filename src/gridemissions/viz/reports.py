from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.dates as mdates

from .base import PAGE_WIDTH, ROW_HEIGHT, COLORS, heatmap, add_watermark
from gridemissions.eia_api import SRC
from gridemissions import eia_api_v2
from gridemissions.load import GraphData

HEATMAP_BAS = [
    "MISO",
    "PJM",
    "ERCO",
    "SWPP",
    "SOCO",
    "CISO",
    "FPL",
    "TVA",
    "NYIS",
    "FPC",
    "ISNE",
    "LGEE",
    "PACE",
    "DUK",
    "PSCO",
    "NEVP",
    "CPLE",
    "AECI",
    "WACM",
    "SC",
    "TEC",
    "SRP",
    "FMPP",
    "LDWP",
    "AZPS",
    "SCEG",
    "JEA",
    "TEPC",
    "PNM",
    "WALC",
    "PACW",
    "NWMT",
    "PSEI",
    "EPE",
    "IPCO",
    "BANC",
    "PGE",
    "AEC",
    "SEC",
    "BPAT",
]


def separate_imp_exp(data: GraphData, ba: str):
    imp = 0.0
    exp = 0.0
    for col in data.get_cols(ba, field="ID"):
        imp += data.df.loc[:, col].apply(lambda x: min(x, 0)).fillna(0.0)
        exp += data.df.loc[:, col].apply(lambda x: max(x, 0)).fillna(0.0)
    return imp, exp


def annual_plot_hourly(
    elec: GraphData, co2: GraphData, ba: str, save=False, fig_folder=None
):
    scaling_elec = 1e-3
    scaling_co2 = 1e-6

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 12.5))
    for ax, data, scale in zip((ax1, ax2), [elec, co2], [scaling_elec, scaling_co2]):
        df_plot = data.df
        ax.plot(data.get_data(region=ba, field="D") * scale, label="D")
        ax.plot(
            data.get_data(region=ba, field="NG") * scale,
            label="G",
            alpha=0.8,
        )

    # CO2i plot
    co2iD = (
        co2.get_data(region=ba, field="D").values.flatten()
        / elec.get_data(region=ba, field="D").values.flatten()
    )
    co2iG = (
        co2.get_data(region=ba, field="NG").values.flatten()
        / elec.get_data(region=ba, field="NG").values.flatten()
    )
    co2iD[co2iD > 2000] = np.nan
    co2iG[co2iG > 2000] = np.nan

    impC, expC = separate_imp_exp(co2, ba)
    impE, expE = separate_imp_exp(elec, ba)

    co2i_imp = impC / impE

    ax1.plot(impE * scaling_elec, label="Imp", alpha=0.7)
    ax1.plot(expE * scaling_elec, label="Exp", alpha=0.7)

    ax2.plot(impC * scaling_co2, label="Imp", alpha=0.7)
    ax2.plot(expC * scaling_co2, label="Exp", alpha=0.7)

    ax3.plot(co2.df.index, co2iD, label="D")
    ax3.plot(co2.df.index, co2iG, label="G", alpha=0.8)
    ax3.plot(co2.df.index, co2i_imp, label="Imp", alpha=0.7)
    ax3.set_ylim(bottom=0.0)

    for ax, data, scale in zip((ax4, ax5), [elec, co2], [scaling_elec, scaling_co2]):
        df_plot = data.df
        for col in data.get_cols(ba, field="ID"):
            ba2 = (eia_api_v2.parse_column(col)["region2"],)
            ax.plot(df_plot.loc[:, col] * scale, label=ba2, alpha=0.7)

    f.autofmt_xdate()
    ax1.set_title(ba)
    ax1.set_ylabel("Electricity (GWh)")
    ax2.set_ylabel("Carbon (ktons)")
    ax3.set_ylabel("Carbon intensity (kg/MWh)")
    ax4.set_ylabel("Electricity trade (MWh)")
    ax5.set_ylabel("Carbon trade (ktons)")

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.legend(loc=7)
    f.tight_layout()

    if save and (fig_folder is not None):
        f.savefig(join(fig_folder, "%s.pdf" % ba))
        plt.close(f)
    return (f, (ax1, ax2, ax3, ax4, ax5))


def summ_stats(s, ax, color, label, q_up=0.9, q_down=0.1):
    s1 = s.groupby(s.index.weekofyear).mean()
    s1_up = s.groupby(s.index.weekofyear).quantile(q_up)
    s1_down = s.groupby(s.index.weekofyear).quantile(q_down)
    ax.plot(s1, label=label, color=color)
    ax.plot(s1_up, color=color, ls="--", lw=0.5)
    ax.plot(s1_down, color=color, ls="--", lw=0.5)
    ax.fill_between(
        s1_up.index,
        s1_down.values.flatten(),
        s1_up.values.flatten(),
        color=color,
        alpha=0.1,
    )


def annual_plot_weekly(
    elec: GraphData, co2: GraphData, ba: str, save=False, fig_folder=None
):
    scaling_elec = 1e-3
    scaling_co2 = 1e-6

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 12.5))

    for ifield, field in enumerate(["D", "NG"]):
        summ_stats(
            elec.get_data(region=ba, field=field) * scaling_elec,
            ax1,
            COLORS[ifield],
            field,
        )
        summ_stats(
            co2.get_data(region=ba, field=field) * scaling_co2,
            ax2,
            COLORS[ifield],
            field,
        )

    # CO2i plot
    co2iD = (
        co2.get_data(region=ba, field="D").values.flatten()
        / elec.get_data(region=ba, field="D").values.flatten()
    )
    co2iG = (
        co2.get_data(region=ba, field="NG").values.flatten()
        / elec.get_data(region=ba, field="NG").values.flatten()
    )

    co2iD[co2iD > 2000] = np.nan
    co2iG[co2iG > 2000] = np.nan

    summ_stats(pd.DataFrame(co2iD, index=co2.df.index), ax3, COLORS[0], "D")
    summ_stats(pd.DataFrame(co2iG, index=co2.df.index), ax3, COLORS[1], "G")

    impC, expC = separate_imp_exp(co2, ba)
    impE, expE = separate_imp_exp(elec, ba)

    co2i_imp = impC / impE

    summ_stats(pd.DataFrame(co2i_imp, index=co2.df.index), ax3, COLORS[2], "Imp")
    summ_stats(
        pd.DataFrame(impE * scaling_elec, index=elec.df.index), ax1, COLORS[2], "Imp"
    )
    summ_stats(
        pd.DataFrame(expE * scaling_elec, index=elec.df.index), ax1, COLORS[3], "Exp"
    )
    summ_stats(
        pd.DataFrame(impC * scaling_co2, index=co2.df.index), ax2, COLORS[2], "Imp"
    )
    summ_stats(
        pd.DataFrame(expC * scaling_co2, index=co2.df.index), ax2, COLORS[3], "Exp"
    )

    ax3.set_ylim(bottom=0.0)

    for ax, data, scaling in zip((ax4, ax5), [elec, co2], [scaling_elec, scaling_co2]):
        for icol, col in enumerate(data.get_cols(ba, field="ID")):
            summ_stats(
                data.df.loc[:, col] * scaling,
                ax,
                COLORS[iba % len(COLORS)],
                label=eia_api_v2.parse_column(col)["region2"],
            )

    ax1.set_title(ba)
    ax1.set_ylabel("Electricity (GWh)")
    ax2.set_ylabel("Carbon (ktons)")
    ax3.set_ylabel("Carbon intensity (kg/MWh)")
    ax4.set_ylabel("Electricity trade (MWh)")
    ax5.set_ylabel("Carbon trade (ktons)")

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.legend(loc=7)
    f.tight_layout()

    if save and (fig_folder is not None):
        f.savefig(join(fig_folder, "%s.pdf" % ba))
        plt.close(f)
    return (f, (ax1, ax2, ax3, ax4, ax5))


def myplot(ax, s, color=None, summarize=True, **kwargs):
    if summarize:
        s1 = s.resample("1W").mean()
        s1_up = s.resample("1W").quantile(0.9)
        s1_down = s.resample("1W").quantile(0.1)
        ax.plot(s1, color=color, **kwargs)
        ax.plot(s1_up, color=color, ls="--", lw=0.2, label="__nolegend__")
        ax.plot(s1_down, color=color, ls="--", lw=0.2, label="__nolegend__")
        ax.fill_between(
            s1_up.index,
            s1_down.values.flatten(),
            s1_up.values.flatten(),
            color=color,
            alpha=0.1,
        )
    else:
        ax.plot(s, color=color, **kwargs)


def cleaning_plot(
    elec,
    region,
    after=None,
    save=False,
    fig_folder=None,
    w_id=False,
    w_balance=False,
    scale=1.0,
    add_title="",
    w_src=True,
    summarize=True,
    fax=None,
):
    scale = float(scale)
    if scale == 1.0:
        unit = "MWh"
    elif scale == 1.0e-3:
        unit = "GWh"
    else:
        raise ValueError(f"Unknown unit for scale {scale}!")

    if w_id and w_src:
        if fax is None:
            f, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(PAGE_WIDTH, ROW_HEIGHT * 3)
            )
        else:
            f, (ax1, ax2, ax3) = fax
    elif w_id:
        if fax is None:
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(PAGE_WIDTH, ROW_HEIGHT * 2))
        else:
            f, (ax1, ax2) = fax
    else:
        if fax is None:
            f, ax1 = plt.subplots(1, 1, figsize=(PAGE_WIDTH, ROW_HEIGHT))
        else:
            f, ax1 = fax
    D = elec.get_data(region=region, field="D") * scale
    G = elec.get_data(region=region, field="NG") * scale
    TI = elec.get_data(region, field="TI") * scale
    myplot(ax1, D, label="D", color=COLORS[0], summarize=summarize)
    myplot(ax1, G, label="G", alpha=0.8, color=COLORS[1], summarize=summarize)
    myplot(ax1, TI, label="TI", alpha=0.8, color=COLORS[2], summarize=summarize)
    if w_balance:
        myplot(
            ax1,
            D + TI - G,
            label="D+TI-G",
            alpha=0.8,
            color=COLORS[3],
            summarize=summarize,
        )

    if after is not None:
        D = after.get_data(region=region, field="D") * scale
        G = after.get_data(region=region, field="NG") * scale
        TI = after.get_data(region=region, field="TI") * scale
        myplot(
            ax1, D, color=COLORS[0], ls="--", label="__nolegend__", summarize=summarize
        )
        myplot(
            ax1,
            G,
            alpha=0.8,
            color=COLORS[1],
            ls="--",
            label="__nolegend__",
            summarize=summarize,
        )
        myplot(
            ax1,
            TI,
            alpha=0.8,
            color=COLORS[2],
            ls="--",
            label="__nolegend__",
            summarize=summarize,
        )

        if w_balance:
            myplot(
                ax1,
                D + TI - G,
                alpha=0.8,
                color=COLORS[3],
                ls="--",
                label="__nolegend__",
                summarize=summarize,
            )
        ax1.plot([], [], alpha=0.8, color="k", ls="--", label="after")

    if len(add_title) > 0:
        add_title = ": " + add_title
    ax1.set_title(region + add_title)
    ax1.legend(loc=6)
    ax1.set_ylabel(f"Electricity ({unit})")
    axes = ax1

    if w_id:
        partners = elec.partners[region]
        for ir2, region2 in enumerate(partners):
            myplot(
                ax2,
                elec.get_data(region=region, region2=region2, field="ID") * scale,
                label=region2,
                alpha=0.7,
                color=COLORS[ir2 % len(COLORS)],
                summarize=summarize,
            )
            if after is not None:
                myplot(
                    ax2,
                    after.get_data(region=region, region2=region2, field="ID") * scale,
                    label="__nolegend__",
                    ls="--",
                    alpha=0.7,
                    color=COLORS[ir2 % len(COLORS)],
                    summarize=summarize,
                )
        ncol = 1
        if len(ax2.lines) / 2 > 10:
            ncol = 2
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(loc=6, ncol=ncol)
        ax2.set_ylabel(f"Electricity trade ({unit})")

        axes = (ax1, ax2)

        if w_src:
            for isrc, src in enumerate(SRC):
                field = f"SRC_{src}"
                if field not in elec.region_fields:
                    continue
                s = elec.get_data(region=region, field=field) * scale
                if len(s) > 0:
                    myplot(
                        ax3,
                        s,
                        label=src,
                        alpha=0.7,
                        color=COLORS[isrc % len(COLORS)],
                        summarize=summarize,
                    )
                    if after is not None:
                        myplot(
                            ax3,
                            after.get_data(region=region, field=field) * scale,
                            ls="--",
                            alpha=0.7,
                            color=COLORS[isrc % len(COLORS)],
                            label="__nolegend__",
                            summarize=summarize,
                        )
            ncol = 1
            if len(ax3.lines) / 2 > 10:
                ncol = 2
            ax3.legend(loc=6, ncol=ncol)
            ax3.set_ylabel(f"Generation by source ({unit})")
            if after is not None:
                ax2.plot([], [], alpha=0.8, color="k", ls="--", label="after")
                ax3.plot([], [], alpha=0.8, color="k", ls="--", label="after")
            axes = (ax1, ax2, ax3)

    if summarize:
        for a in list(axes):
            a.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    else:
        for a in list(axes):
            a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            for label in a.get_xticklabels(which="major"):
                label.set_ha("right")
                label.set_rotation(30)
    f.tight_layout()

    if save and (fig_folder is not None):
        f.savefig(join(fig_folder, "%s.pdf" % region))
        plt.close(f)
    return (f, axes)


def heatmap_report(
    co2, elec, year=2021, which="individual", fig_folder=None, tz_offset=6
):
    """

    Parameters
    ----------
    co2: BaData
        carbon data
    elec: BaData
        electricity data
    which: str in ["individual", "group"]
    fig_folder: str, default None
    tz_offset: int, default 6
        offset to shift the time stamps (assumes data is provided in UTC)
        default is Mountain time
    """
    start = pd.to_datetime(f"{year}0101T0000Z")
    end = pd.to_datetime(f"{int(year)+1}0101T0000Z")
    co2i = pd.DataFrame(
        {
            ba: (
                co2.df.loc[start:end, co2.get_cols(ba, field="D")].values.flatten()
                / elec.df.loc[start:end, elec.get_cols(ba, field="D")].values.flatten()
            )
            for ba in HEATMAP_BAS
        },
        index=co2.df.loc[start:end].index,
    )

    # Change timezone
    co2i.index -= pd.Timedelta(f"{tz_offset}h")

    free_folder = fig_folder / f"{year}" / "free_scale"
    fixed_folder = fig_folder / f"{year}" / "fixed_scale"
    free_folder.mkdir(parents=True, exist_ok=True)
    fixed_folder.mkdir(parents=True, exist_ok=True)

    for ba in HEATMAP_BAS:
        f, ax = plt.subplots(figsize=(PAGE_WIDTH, 1.5 * ROW_HEIGHT))
        heatmap(
            co2i[ba],
            fax=(f, ax),
            cmap="RdYlGn_r",
            cbar_label="kg/MWh",
            transpose=True,
        )
        add_watermark(ax)
        ax.set_title(f"{ba}: Consumption-based carbon intensity", fontsize="large")
        f.tight_layout()
        if fig_folder is not None:
            f.savefig(free_folder / f"{ba}.pdf")
            f.savefig(free_folder / f"{ba}.png")
            image.thumbnail(
                free_folder / f"{ba}.png",
                free_folder / f"{ba}-thumbnail.png",
                scale=0.1,
            )
        plt.close(f)

        f, ax = plt.subplots(figsize=(PAGE_WIDTH, 1.5 * ROW_HEIGHT))
        heatmap(
            co2i[ba],
            fax=(f, ax),
            vmin=100,
            vmax=900,
            cmap="RdYlGn_r",
            cbar_label="kg/MWh",
            transpose=True,
        )
        add_watermark(ax)
        ax.set_title(f"{ba}: Consumption-based carbon intensity", fontsize="large")
        f.tight_layout()
        if fig_folder is not None:
            f.savefig(fixed_folder / f"{ba}.pdf")
            f.savefig(fixed_folder / f"{ba}.png")
            image.thumbnail(
                fixed_folder / f"{ba}.png",
                fixed_folder / f"{ba}-thumbnail.png",
                scale=0.1,
            )
        plt.close(f)

    n = len(HEATMAP_BAS)
    nrows = n // 4
    f, ax = plt.subplots(nrows, 4, figsize=(1.2 * PAGE_WIDTH, nrows / 2 * ROW_HEIGHT))
    ax = ax.flatten()

    for iba, ba in enumerate(HEATMAP_BAS[:-1]):
        ax[iba].set_title(ba)
        heatmap(
            co2i[ba],
            fax=(f, ax[iba]),
            vmin=100,
            vmax=900,
            cmap="RdYlGn_r",
            cbar_label="kg/MWh",
            with_cbar=False,
        )

    for a in ax:
        a.set_yticks([])
        a.set_xticks([])
        a.set_ylabel("")
        a.set_xlabel("")
    f.tight_layout()

    ba = HEATMAP_BAS[-1]
    iba = len(HEATMAP_BAS) - 1
    ax[iba].set_title(ba)
    heatmap(
        co2i[ba],
        fax=(f, ax[iba]),
        vmin=100,
        vmax=900,
        cmap="RdYlGn_r",
        cbar_label="kg/MWh",
        with_cbar=True,
        cbar_ax=[ax[-4:]],
    )
    for a in ax:
        a.set_yticks([])
        a.set_xticks([])
        a.set_ylabel("")
        a.set_xlabel("")
    add_watermark(ax[iba], y=-0.05)

    if fig_folder is not None:
        f.savefig(fig_folder / f"{year}" / f"Top {n} heatmaps.pdf")
        f.savefig(fig_folder / f"{year}" / f"Top {n} heatmaps.png")


def timeseries_report_plot(func):
    """
    Decorator for plotting functions for the timeseries report
    """

    def decorated(*args, **kwargs):
        ba = args[0]
        if "add_title" not in kwargs:
            kwargs["add_title"] = ""
        add_title = kwargs["add_title"]
        if ("fig_folder" in kwargs) and (kwargs["fig_folder"] is not None):
            kwargs["fig_folder"] = kwargs["fig_folder"] / add_title
        else:
            kwargs["fig_folder"] = None
        if len(add_title) > 0:
            add_title = f": {add_title}"

        if not (("fax" in kwargs) and (kwargs["fax"] is not None)):
            fax = plt.subplots(1, 1, figsize=(PAGE_WIDTH, ROW_HEIGHT))
            kwargs["fax"] = fax

        f, ax = func(*args, **kwargs)

        # Post process plot
        ncol = 1
        if len(ax.lines) / 2 > 10:
            ncol = 2
        ax.legend(loc=6, ncol=ncol)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        ax.set_title(ba + add_title)
        ax.set_ylabel(f"{kwargs['unit']}")
        add_watermark(ax)
        f.tight_layout()

        if kwargs["fig_folder"] is not None:
            fig_folder = kwargs["fig_folder"]
            fig_folder.mkdir(exist_ok=True, parents=True)
            f.savefig(fig_folder / f"{ba}.png")
            f.savefig(fig_folder / f"{ba}.pdf")
            image.thumbnail(
                fig_folder / f"{ba}.png", fig_folder / f"{ba}-thumbnail.png", scale=0.1
            )
            plt.close(f)

    return decorated


@timeseries_report_plot
def _plot_electricity_carbon(
    ba: str, ba_data: GraphData, fax=None, scale=1e-3, **kwargs
):
    f, ax = fax
    D = ba_data.get_data(region=ba, field="D") * scale
    G = ba_data.get_data(region=ba, field="NG") * scale
    TI = ba_data.get_data(region=ba, field="TI") * scale
    myplot(ax, D, label="Demand", color=COLORS[0])
    myplot(ax, G, label="Generation", alpha=0.8, color=COLORS[1])
    myplot(ax, TI, label="Total Interchange", alpha=0.8, color=COLORS[2])

    return f, ax


@timeseries_report_plot
def _plot_trade(ba: str, ba_data: GraphData, fax=None, scale=1e-3, **kwargs):
    f, ax = fax
    for icol, col in enumerate(ba_data.get_cols(ba, field="ID")):
        myplot(
            ax,
            ba_data.df.loc[:, col] * scale,
            label=eia_api_v2.parse_column(col)["region2"],
            alpha=0.7,
            color=COLORS[icol % len(COLORS)],
        )
    return f, ax


@timeseries_report_plot
def _plot_generation_by_source(
    ba: str, elec: GraphData, fax=None, scale=1e-3, **kwargs
):
    f, ax = fax
    for ifuel, fuel in enumerate(eia_api_v2.FUELS):
        if not elec.has_field([fuel], ba):
            continue
        myplot(
            ax,
            elec.get_data(region=ba, field=fuel) * scale,
            label=fuel,
            alpha=0.7,
            color=COLORS[ifuel % len(COLORS)],
        )
    return f, ax


@timeseries_report_plot
def _plot_carbon_intensity(
    ba: str, co2: GraphData, elec: GraphData, fax=None, **kwargs
):
    f, ax = fax
    co2iD = (
        co2.get_data(region=ba, field="D").values.flatten()
        / elec.get_data(region=ba, field="D").values.flatten()
    )
    co2iG = (
        co2.get_data(region=ba, field="NG").values.flatten()
        / elec.get_data(region=ba, field="NG").values.flatten()
    )
    co2iD[co2iD > 2000] = np.nan
    co2iG[co2iG > 2000] = np.nan

    impC, expC = separate_imp_exp(co2, ba)
    impE, expE = separate_imp_exp(elec, ba)

    co2i_imp = impC / impE
    myplot(ax, pd.Series(co2iD, index=co2.df.index), label="Demand", color=COLORS[0])
    myplot(
        ax, pd.Series(co2iG, index=co2.df.index), label="Generation", color=COLORS[1]
    )
    myplot(
        ax, pd.Series(co2i_imp, index=co2.df.index), label="Imports", color=COLORS[2]
    )
    # Sanity check: Exports is same as Demand
    #     myplot(ax, pd.Series(co2i_exp, index=co2.df.index), label="Exports", color=COLORS[3])
    ax.set_ylim(bottom=0.0)
    return f, ax


def timeseries_report(co2, elec, fig_folder=None, regions=None):
    regions = HEATMAP_BAS if regions is None else regions
    for ba in regions:
        _plot_electricity_carbon(
            ba,
            elec,
            fig_folder=fig_folder,
            scale=1e-3,
            unit="GWh",
            add_title="Electricity",
        )
        _plot_electricity_carbon(
            ba, co2, fig_folder=fig_folder, scale=1e-6, unit="kton", add_title="Carbon"
        )
        _plot_trade(
            ba,
            elec,
            fig_folder=fig_folder,
            scale=1e-3,
            unit="GWh",
            add_title="Electricity trade",
        )
        _plot_trade(
            ba,
            co2,
            fig_folder=fig_folder,
            scale=1e-6,
            unit="kton",
            add_title="Carbon trade",
        )
        _plot_carbon_intensity(
            ba,
            co2,
            elec,
            fig_folder=fig_folder,
            add_title="Carbon intensity",
            unit="kg/MWh",
        )
        _plot_generation_by_source(
            ba,
            elec,
            fig_folder=fig_folder,
            scale=1e-3,
            unit="GWh",
            add_title="Generation by source",
        )
