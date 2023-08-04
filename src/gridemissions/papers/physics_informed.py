import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from gridemissions.viz.base import PAGE_WIDTH, ROW_HEIGHT, COLORS
from gridemissions.eia_api import KEYS, SRC


def figure1(ba, ba_data_A, ba_data_D, ba_data_C=None, scale=1e-3, save_fig=False):
    f, ax = plt.subplots(3, 2, figsize=(PAGE_WIDTH, ROW_HEIGHT * 3))

    d_col = ba_data_A.get_cols(ba, "D")[0]

    # Plot 1: demand at different steps
    ax[0, 0].plot(ba_data_A.df.loc[:, d_col] * scale, "-o", ms=2, lw=0.7, label="raw")
    if ba_data_C is not None:
        ax[0, 0].plot(
            ba_data_C.df.loc[:, d_col] * scale, lw=0.7, ls="--", label="pre-processed"
        )
    ax[0, 0].plot(ba_data_D.df.loc[:, d_col] * scale, lw=0.7, label="reconciled")
    ax[0, 0].set_ylim(bottom=0)

    # Plot 2: D, G, TI before after
    ax1 = ax[0, 1]
    df_plot = ba_data_A.df
    D = df_plot.loc[:, ba_data_A.get_cols(r=ba, field="D")[0]] * scale
    G = df_plot.loc[:, ba_data_A.get_cols(r=ba, field="NG")[0]] * scale
    TI = df_plot.loc[:, ba_data_A.get_cols(r=ba, field="TI")[0]] * scale
    ax1.plot(D, label="D", color=COLORS[0])
    ax1.plot(G, label="G", alpha=0.8, color=COLORS[1])
    ax1.plot(TI, label="TI", alpha=0.8, color=COLORS[2])
    ax1.plot(D + TI - G, label="D+TI-G", alpha=0.8, color=COLORS[3])

    df_plot = ba_data_D.df
    D = df_plot.loc[:, ba_data_A.get_cols(r=ba, field="D")[0]] * scale
    G = df_plot.loc[:, ba_data_A.get_cols(r=ba, field="NG")[0]] * scale
    TI = df_plot.loc[:, ba_data_A.get_cols(r=ba, field="TI")[0]] * scale
    ax1.plot(D, color=COLORS[0], ls="--", label="__nolegend__")
    ax1.plot(G, alpha=0.8, color=COLORS[1], ls="--", label="__nolegend__")
    ax1.plot(TI, alpha=0.8, color=COLORS[2], ls="--", label="__nolegend__")

    ax1.plot(D + TI - G, alpha=0.8, color=COLORS[3], ls="--", label="__nolegend__")

    partners = ba_data_A.get_trade_partners(ba)
    thresh = len(partners) / 2
    ax2 = ax[1, 0]
    for iba2, ba2 in enumerate(partners):
        if iba2 >= thresh:
            ax2 = ax[2, 0]
        ax2.plot(
            ba_data_A.df.loc[:, ba_data_A.KEY["ID"] % (ba, ba2)] * scale,
            label=ba2,
            alpha=0.7,
            color=COLORS[iba2 % len(COLORS)],
        )
        ax2.plot(
            ba_data_D.df.loc[:, ba_data_A.KEY["ID"] % (ba, ba2)] * scale,
            label="__nolegend__",
            ls="--",
            alpha=0.7,
            color=COLORS[iba2 % len(COLORS)],
        )

    src_cols = ["COL", "NG", "OIL", "OTH", "NUC", "SUN", "WND", "WAT"]
    thresh = len(src_cols) / 2
    ax3 = ax[1, 1]

    colors = COLORS.copy()
    colors[0] = "k"
    colors[1] = COLORS[6]
    colors[2] = COLORS[7]
    colors[7] = COLORS[0]
    colors[5] = COLORS[1]
    colors[6] = COLORS[2]
    for isrc, src in enumerate(src_cols):
        if isrc >= thresh:
            ax3 = ax[2, 1]
        if ba_data_A.KEY[f"SRC_{src}"] % ba in ba_data_A.df.columns:
            ax3.plot(
                ba_data_A.df.loc[:, ba_data_A.KEY[f"SRC_{src}"] % ba] * scale,
                label=src,
                alpha=0.7,
                color=colors[isrc % len(colors)],
            )
            ax3.plot(
                ba_data_D.df.loc[:, ba_data_A.KEY[f"SRC_{src}"] % ba] * scale,
                ls="--",
                alpha=0.7,
                color=colors[isrc % len(colors)],
                label="__nolegend__",
            )

    ax[0, 0].set_title("(a) Cleaning steps (shown for demand)")
    ax[0, 1].set_title("(b) Demand, Generation, Tot. Interchange")
    ax[1, 0].set_title("(c) Interchange")
    ax[2, 0].set_title("(e) Interchange (cont'd)")
    ax[1, 1].set_title("(d) Generation by source")
    ax[2, 1].set_title("(f) Generation by source (cont'd)")

    for a in [ax[1, 0], ax[2, 0]]:
        a.set_ylim(-1.5, 1.1)
        a.legend(loc=3, ncol=2)
    ax[0, 0].legend(loc=3, ncol=2, handlelength=1.5, columnspacing=1)
    ax[0, 1].legend(loc=6, ncol=2)
    ax[1, 1].legend(loc=3, ncol=2)
    ax[2, 1].legend(loc=6, ncol=2)

    import matplotlib.dates as mdates

    for a in ax.flatten():
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
        a.set_ylabel("GW")

    ax[1, 1].set_yticks([0.0, 5, 10, 15])
    ax[1, 0].set_yticks([-2, -1, 0, 1])
    ax[2, 0].set_yticks([-2, -1, 0, 1])

    f.autofmt_xdate()
    f.tight_layout()

    return f, ax


def figure2(ba_list, n, delta_quantiles, median_raw, raw):
    def make_plot(ax0, delta_quantiles, median_raw, regions):
        for iba, ba in enumerate(regions):
            if iba % 2 == 0:
                ax0.axhspan(iba - 0.5, iba + 0.5, color=bgcolor)

            x_num = -900

            ha = "right"
            va = "center"

            def myformat(n):
                if n > 1.0:
                    return "%2.0f" % n
                else:
                    return "%2.1f" % n

            def plotter(ba, field, color, yval):
                col = raw.get_cols(ba, field)[0]
                denom = median_raw.loc[raw.get_cols(ba, "D")[0]]
                ax0.plot(
                    delta_quantiles.loc[[q1, q2], col] / denom * 100,
                    [yval, yval],
                    color=color,
                )
                ax0.plot(
                    [delta_quantiles.loc[0.5, col] / denom * 100],
                    [yval],
                    color=color,
                    marker="|",
                    ms=4,
                )
                # label
                if field == "D":
                    ax0.text(
                        x_num,
                        iba,
                        f"{myformat(denom/1e3)} GW",
                        fontsize="x-small",
                        ha="left",
                        va=va,
                    )

            # Demand
            plotter(ba, "D", COLORS[0], iba + barWidth)

            # Generation
            plotter(ba, "NG", COLORS[1], iba)

            # Interchange
            plotter(ba, "TI", COLORS[2], iba - barWidth)

        base = 10
        ax0.set_xscale(
            "symlog", linscale=np.log(np.e) / np.log(base) * (1 - (1 / base))
        )

        for iba, ba in enumerate(regions):
            ax0.text(-1150, iba, ba, ha=ha, va=va, fontsize="small")
        ax0.set_ylim(-0.5, len(regions) - 0.5)

        ax0.set_xlim([-1000, 100])
        ax0.set_yticklabels([])
        ax0.set_xticks([-100, -10, -1, 1, 10, 100])
        ax0.set_xticklabels(["-100", "-10", "-1", "1", "10", "100"], fontsize="small")
        ax0.yaxis.set_ticks_position("none")
        ax0.xaxis.tick_top()
        ax0.yaxis.grid(False)

        alpha = 0.1
        ax0.axvspan(-1.0, 1.0, color="green", alpha=alpha)
        ax0.axvspan(-10.0, -1.0, color="yellow", alpha=alpha)
        ax0.axvspan(-100.0, -10.0, color="red", alpha=alpha)
        ax0.axvspan(1.0, 10.0, color="yellow", alpha=alpha)
        ax0.axvspan(10.0, 100.0, color="red", alpha=alpha)

    bgcolor = "#eeeeee"
    barWidth = 0.3
    f, ax = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2 * ROW_HEIGHT))
    q1 = 0.1
    q2 = 0.9

    make_plot(ax[0], delta_quantiles, median_raw, ba_list[2 * n :])
    make_plot(ax[1], delta_quantiles, median_raw, ba_list[n : 2 * n])
    make_plot(ax[2], delta_quantiles, median_raw, ba_list[0:n])
    ax[1].plot([], [], label="Demand", color=COLORS[0])
    ax[1].plot([], [], label="Generation", color=COLORS[1])
    ax[1].plot([], [], label="Interchange", color=COLORS[2])
    ax[1].legend(loc=9, bbox_to_anchor=(0.5, 0), ncol=3)
    f.suptitle(
        "Adjustments for hourly data (% of median demand)", y=0.99, fontsize="medium"
    )
    f.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.9, wspace=0.25)
    return f, ax


def figure3(regions, deltas_abs, median_raw, raw):
    f, ax = plt.subplots(1, 1, figsize=(1.2 * PAGE_WIDTH, 1.2 * PAGE_WIDTH))
    vals = []
    bgcolor = "#eeeeee"

    def myformat(n):
        if n > 1.0:
            return "%2.0f" % n
        else:
            return "%2.1f" % n

    def get_color(s):
        if s < 1.0:
            return "green"
        elif s < 10.0:
            return "yellow"
        else:
            return "red"

    for ibay, bay in enumerate(regions[::-1]):
        #         print(ibay)
        if ibay % 2 == 0:
            ax.axvspan(ibay - 0.5, ibay + 0.5, color=bgcolor)
        denom = median_raw.loc[raw.get_cols(bay, "TI")[0]]
        ax.text(
            -2.4,
            ibay,
            f"{myformat(denom/1e3)} GW",
            fontsize="small",
            ha="left",
            va="center",
        )
        for ibax, bax in enumerate(regions):
            if ibax % 2 == 0:
                ax.axhspan(ibax - 0.5, ibax + 0.5, color=bgcolor)

            col_ID = KEYS["E"]["ID"] % (bay, bax)
            if col_ID in raw.df.columns:
                s = deltas_abs.loc[0.1, col_ID] / denom * 100
                s2 = deltas_abs.loc[0.9, col_ID] / denom * 100
                vals.append([ibax, ibay, s, s2, get_color(s), get_color(s2)])
                #                 if "".join(np.sort([bay, bax])) == "LGEEMISO":
                #                     print(f"{bay}, {bax}, {s:.2f}, {s2:.2f}, {denom:.2f}")

                ax.add_patch(
                    Rectangle(
                        (ibax - 0.5, ibay - 0.5),
                        1,
                        1,
                        facecolor=get_color(s2),
                        alpha=0.1,
                        zorder=2,
                    )
                )

    x, y, s, s2, fc, fc2 = tuple(zip(*vals))

    def rescale(x):
        return np.interp(x, [min(s), max(s2)], [1, 300])

    ax.scatter(
        x,
        y,
        s=rescale(s2),
        zorder=3,
        facecolor="none",
        edgecolor="k",
        linewidths=0.5,
        linestyle="--",
    )
    ax.scatter(
        x, y, s=rescale(s), zorder=3, facecolor="none", edgecolor="k", linewidths=0.5
    )

    ax.grid()
    ax.set_xlim([-2.5, len(regions) - 0.5])
    ax.set_ylim([-0.5, len(regions) - 0.5])
    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_yticklabels(regions[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    dummy = ax.scatter(
        x=[-1, -1, -1], y=[-1, -1, -1], s=rescale(np.array([1, 10, 100]))
    )
    handles, labels = dummy.legend_elements(prop="sizes", alpha=0.6)
    ax.legend(handles, ["1%", "10%", "100%"], loc=9, bbox_to_anchor=(0.5, 0), ncol=3)
    plt.subplots_adjust(left=0.065, bottom=0.045, right=0.99, top=0.94, wspace=0.25)
    return f, ax


def figure4(regions, nr, deltas_abs, median_raw, raw):
    f, ax = plt.subplots(1, 3, figsize=(1.5 * PAGE_WIDTH, 2 * ROW_HEIGHT))
    vals = []
    bgcolor = "#eeeeee"

    def myformat(n):
        if n > 1.0:
            return "%2.0f" % n
        else:
            return "%2.1f" % n

    def get_color(s):
        if s < 1.0:
            return "green"
        elif s < 10.0:
            return "yellow"
        else:
            return "red"

    for ibay, bay in enumerate(regions):
        if ibay % 2 == 0:
            ax[ibay // nr].axhspan(
                (nr - (ibay % nr)) - 1.5, (nr - (ibay % nr)) - 0.5, color=bgcolor
            )
        denom = median_raw.loc[raw.get_cols(bay, "NG")[0]]
        ax[ibay // nr].text(
            -2.4,
            (nr - (ibay % nr)) - 1,
            f"{myformat(denom/1e3)} GW",
            fontsize="small",
            ha="left",
            va="center",
        )
        src_cols = [src for src in SRC if src not in ["UNK", "GEO", "BIO"]]
        for ibax, src in enumerate(src_cols):
            if ibax % 2 == 0:
                ax[ibay // nr].axvspan(ibax - 0.5, ibax + 0.5, color=bgcolor)

            col_SRC = KEYS["E"]["SRC_%s" % src] % bay
            if col_SRC in raw.df.columns:
                s = deltas_abs.loc[0.1, col_SRC] / denom * 100
                s2 = deltas_abs.loc[0.9, col_SRC] / denom * 100
                vals.append([ibax, ibay, s, s2])

    x_vals, y_vals, s_vals, s2_vals = tuple(zip(*vals))

    def rescale(x):
        return np.interp(x, [min(s_vals), max(s2_vals)], [1, 300])

    for x, y, s, s2 in vals:
        ax[y // nr].add_patch(
            Rectangle(
                (x - 0.5, (nr - (y % nr)) - 1.5),
                1,
                1,
                facecolor=get_color(s2),
                alpha=0.1,
                zorder=2,
            )
        )

        ax[y // nr].scatter(
            x,
            (nr - (y % nr)) - 1,
            s=rescale(s2),
            zorder=3,
            facecolor="none",
            edgecolor="k",
            linewidths=0.5,
            linestyle="--",
        )
        ax[y // nr].scatter(
            x,
            (nr - (y % nr)) - 1,
            s=rescale(s),
            zorder=3,
            facecolor="none",
            edgecolor="k",
            linewidths=0.5,
        )

    for i, a in enumerate(ax):
        a.grid()
        a.set_xlim([-2.5, len(src_cols) - 0.5])
        a.set_ylim([-0.5, nr - 0.5])
        a.set_xticks([])
        a.set_yticks([])
        for j, txt in enumerate(src_cols):
            a.text(j - 0.5, nr - 0.35, txt, rotation=45, ha="left")
        for j, txt in enumerate(regions[i * nr : (i + 1) * nr][::-1]):
            a.text(-2.65, j, txt, ha="right", va="center")
        a.xaxis.tick_top()
        a.yaxis.set_ticks_position("none")
        a.xaxis.set_ticks_position("none")

    dummy = ax[1].scatter(
        x=[-1, -1, -1], y=[-1, -1, -1], s=rescale(np.array([1, 10, 100]))
    )
    handles, labels = dummy.legend_elements(prop="sizes", alpha=0.6)
    ax[1].legend(handles, ["1%", "10%", "100%"], loc=9, bbox_to_anchor=(0.5, 0), ncol=3)
    # f.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.92, wspace=0.25)
    return f, ax


def get_changes(df, ref_years=[2015, 2016, 2017, 2018, 2019], which="percent"):
    ref_data = df[df.index.year.isin(ref_years)]
    ref_values = ref_data.groupby([ref_data.index.month, ref_data.index.day]).mean()
    target_data = df[df.index.year.isin([2020])]
    target_values = target_data.groupby(
        [target_data.index.month, target_data.index.day]
    ).mean()
    if which == "percent":
        percent_changes = (target_values - ref_values) / ref_values * 100
    else:
        percent_changes = target_values - ref_values
    percent_changes.index = pd.to_datetime(
        (
            10000 * 2020
            + 100 * percent_changes.index.get_level_values(0)
            + percent_changes.index.get_level_values(1)
        ).astype(str)
    )
    return percent_changes
