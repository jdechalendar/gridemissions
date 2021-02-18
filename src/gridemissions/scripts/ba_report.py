#!/usr/bin/env python

from gridemissions import config
import argparse
import logging
from os.path import join
from gridemissions.viz.ba_plots import annual_plot_hourly, annual_plot_weekly
from datetime import datetime, timedelta
from gridemissions.load import BaData
import matplotlib.pyplot as plt
import cmocean
import seaborn as sns
from pandas.plotting import register_matplotlib_converters


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

    # Load data
    folder_in = join(config["DATA_PATH"], "analysis", "local")
    file_name = "EBA"
    co2 = BaData(fileNm=join(folder_in, "%s_co2.csv" % file_name), variable="CO2")
    elec = BaData(fileNm=join(folder_in, "%s_elec.csv" % file_name))

    # Parse args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--report", default="1", help="Which report to make")
    args = argparser.parse_args()

    # Configure logging
    logger = logging.getLogger("scraper")

    # Do work
    if args.report == "1":
        logger.info("Creating full hourly report")
        fig_folder = join(config["FIG_PATH"], "hourly_full")
        for ba in elec.regions:
            annual_plot_hourly(elec, co2, ba, save=True, fig_folder=fig_folder)

    elif args.report == "2":
        logger.info("Creating full weekly report")
        fig_folder = join(config["FIG_PATH"], "weekly_full")
        for ba in elec.regions:
            annual_plot_weekly(elec, co2, ba, save=True, fig_folder=fig_folder)

    elif args.report == "3":
        logger.info("Creating hourly report for last 2 weeks")
        fig_folder = join(config["FIG_PATH"], "hourly_2weeks")
        now = datetime.utcnow()
        start = now - timedelta(hours=14 * 30)
        end = now

        small_elec = BaData(df=elec.df.loc[start:end])
        small_co2 = BaData(df=co2.df.loc[start:end], variable="CO2")
        for ba in elec.regions:
            annual_plot_hourly(
                small_elec, small_co2, ba, save=True, fig_folder=fig_folder
            )

    else:
        logger.error("Unknown report option! %s" % args.report)
