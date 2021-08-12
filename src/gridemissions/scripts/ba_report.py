#!/usr/bin/env python

import argparse
import logging
import os
from os.path import join
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cmocean
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

import gridemissions
from gridemissions.viz.reports import (
    annual_plot_hourly,
    annual_plot_weekly,
    heatmap_report,
)
from gridemissions.load import BaData


def main():
    # Setup plotting
    register_matplotlib_converters()
    plt.style.use("seaborn-paper")
    plt.rcParams["figure.figsize"] = [6.99, 2.5]
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 10
    cmap = cmocean.cm.cmap_d["phase"]
    colors = sns.color_palette("colorblind")

    # Parse args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--report", default="1", help="Which report to make")
    argparser.add_argument(
        "--year", default="2021", help="""Which year, for report "heatmap" """
    )
    args = argparser.parse_args()

    # Configure logging
    logger = logging.getLogger("gridemissions")
    FIG_PATH = gridemissions.config["FIG_PATH"]
    # Load data
    file_name = join(gridemissions.config["DATA_PATH"], "analysis", "webapp", "EBA_%s.csv")
    co2 = BaData(fileNm=file_name % "co2", variable="CO2")
    elec = BaData(fileNm=file_name % "elec", variable="E")

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
        logger.info(f"Running report heatmap for year {args.year}")
        fig_folder = join(FIG_PATH, "heatmap_report")
        os.makedirs(fig_folder, exist_ok=True)
        heatmap_report(co2, elec, year=args.year, fig_folder=fig_folder)

    else:
        logger.error("Unknown report option! %s" % args.report)
