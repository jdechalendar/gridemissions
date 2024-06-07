#!/usr/bin/env python

import pathlib
import json
import argparse
import logging
from os.path import join
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

import gridemissions as ge
from gridemissions.viz.reports import (
    annual_plot_hourly,
    annual_plot_weekly,
    heatmap_report,
    timeseries_report,
)


def main():
    """
    Create reports from bulk datasets
    """
    # Setup plotting
    register_matplotlib_converters()
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["figure.figsize"] = [6.99, 2.5]
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 10

    # Parse args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--report", default="1", help="Which report to make")
    args = argparser.parse_args()

    # Configure logging
    logger = logging.getLogger("gridemissions")
    FIG_PATH = ge.config["FIG_PATH"]

    # Load data
    folder = ge.config["DATA_PATH"] / "EIA_Grid_Monitor" / "processed"
    elec_files = [f for f in folder.iterdir() if f.name.endswith("elec.csv")]
    co2_files = [f for f in folder.iterdir() if f.name.endswith("co2.csv")]

    elec = ge.GraphData(
        pd.concat(
            [pd.read_csv(path, index_col=0, parse_dates=True) for path in elec_files],
            axis=0,
        )
    )
    co2 = ge.GraphData(
        pd.concat(
            [pd.read_csv(path, index_col=0, parse_dates=True) for path in co2_files],
            axis=0,
        )
    )
    co2.df.sort_index(inplace=True)
    elec.df.sort_index(inplace=True)

    # Note: we need to drop duplicate indices here because there seems to be duplicates at the beginning/end of the files
    elec.df = elec.df[~elec.df.index.duplicated(keep="last")]
    co2.df = co2.df[~co2.df.index.duplicated(keep="last")]

    # Do work
    if args.report == "1":
        logger.info("Creating full hourly report")
        fig_folder = join(FIG_PATH, "hourly_full")
        for ba in elec.regions:
            annual_plot_hourly(elec, co2, ba, save=True, fig_folder=fig_folder)

    elif args.report == "2":
        logger.info("Creating full weekly report")
        fig_folder = join(FIG_PATH, "weekly_full")
        for ba in elec.regions:
            annual_plot_weekly(elec, co2, ba, save=True, fig_folder=fig_folder)

    elif args.report == "3":
        logger.info("Creating hourly report for last 2 weeks")
        fig_folder = join(FIG_PATH, "hourly_2weeks")
        now = datetime.utcnow()
        start = now - timedelta(hours=14 * 30)
        end = now

        small_elec = BaData(df=elec.df.loc[start:end])
        small_co2 = BaData(df=co2.df.loc[start:end], variable="CO2")
        for ba in elec.regions:
            annual_plot_hourly(
                small_elec, small_co2, ba, save=True, fig_folder=fig_folder
            )

    elif args.report == "heatmap":
        fig_folder = pathlib.Path(FIG_PATH) / "heatmap_report"
        for year in co2.df.index.year.unique():
            logger.info(f"Running report heatmap for year {year}")
            heatmap_report(co2, elec, year=year, fig_folder=fig_folder)
        _generate_contents_heatmap(fig_folder)

    elif args.report == "timeseries":
        logger.info("Running report timeseries")
        fig_folder = pathlib.Path(FIG_PATH) / "timeseries_report"
        timeseries_report(co2, elec, fig_folder=fig_folder)
        _generate_contents_timeseries(fig_folder)

    else:
        logger.error("Unknown report option! %s" % args.report)


def _generate_contents_heatmap(folder):
    """Generate json file with map to the different heatmaps"""
    contents = {}

    for year in folder.iterdir():
        if not (year.is_dir() and year.name.isnumeric()):
            continue
        contents[year.name] = {}
        for scale in year.iterdir():
            if not scale.is_dir():
                continue
            contents[year.name][scale.name] = [
                x.name.strip("-thumbnail.png")
                for x in scale.iterdir()
                if x.name.endswith("-thumbnail.png")
            ]

    with open(folder / "contents.json", "w") as fw:
        json.dump(contents, fw)


def _generate_contents_timeseries(folder):
    """Generate json file with map to timeseries plots"""
    contents = {}
    for ff in folder.iterdir():
        if not ff.is_dir():
            continue
        contents[ff.name] = [
            x.name.strip("-thumbnail.png")
            for x in ff.iterdir()
            if x.name.endswith("-thumbnail.png")
        ]

    with open(folder / "contents.json", "w") as fw:
        json.dump(contents, fw)
