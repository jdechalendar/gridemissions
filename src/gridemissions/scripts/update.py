#!/usr/bin/env python

import os
from os.path import join
import pandas as pd
import argparse
import logging.config
from datetime import datetime, timedelta

from gridemissions import config
from gridemissions.workflows import make_dataset, update_dataset, update_d3map
from .utils import str2bool


def main():
    # Parse command-line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--make",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="Create new data set",
    )
    argparser.add_argument(
        "-s",
        "--scrape",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="Pull new data from the EIA",
    )
    argparser.add_argument(
        "-u",
        "--update",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="Update the historical data with what is in the tmp folder",
    )
    argparser.add_argument(
        "-um",
        "--update_d3map",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="Update d3 map",
    )
    argparser.add_argument("--start", default="now", help="start date (%Y-%m-%d)")
    argparser.add_argument("--end", default="now", help="end date (%Y-%m-%d)")
    argparser.add_argument(
        "--folder_hist", default="webapp", help="folder in which history is saved"
    )
    argparser.add_argument(
        "--folder_new",
        default=config["TMP_PATH"],
        help="folder in which history is saved",
    )
    argparser.add_argument(
        "--file_name", default="EBA", help="folder in which history is saved"
    )
    argparser.add_argument(
        "-d",
        "--debug",
        default=False,
        const=True,
        nargs="?",
        type=str2bool,
        help="Whether to include debug information in logging",
    )

    file_name = "EBA"
    argparser.add_argument(
        "--extract2weeks",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="whether to extract 2 weeks",
    )
    args = argparser.parse_args()

    # Configure logging
    logger = logging.getLogger("scraper")
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("clean").setLevel(logging.DEBUG)
    if config["ENV"] == "vm":
       # Store logs
       log_path = join(config["DATA_PATH"], "Logs")
       os.makedirs(log_path, exist_ok=True)
       log_path = join(log_path, "log")
       logger.info("Saving logs to: %s" % str(log_path))
       fh = logging.handlers.TimedRotatingFileHandler(log_path, when="midnight")
       if args.debug:
           fh.setLevel(logging.DEBUG)
       else:
           fh.setLevel(logging.INFO)
           fh.setFormatter(
                   logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
           )
           logger.addHandler(fh)
           logging.getLogger("clean").addHandler(fh)


    now = datetime.utcnow()
    if args.start == "now":
        start = now - timedelta(hours=24 * 14)
    else:
        start = pd.to_datetime(args.start)
    if args.end == "now":
        end = now
    else:
        end = pd.to_datetime(args.end)

    if end <= start:
        raise ValueError("end <= start")
    end = end.strftime("%Y%m%dT%HZ")
    start = start.strftime("%Y%m%dT%HZ")

    logger.info(args)

    if args.folder_hist == "webapp":
        args.folder_hist = join(config["DATA_PATH"], "analysis", "webapp")
    elif args.folder_hist == "local":
        args.folder_hist = join(config["DATA_PATH"], "analysis", "local")

    if args.folder_new == config["TMP_PATH"]:
        args.folder_new = join(config["TMP_PATH"], "emissions_app")

    os.makedirs(args.folder_new, exist_ok=True)

    if args.extract2weeks:
        folder_2weeks = join(config["DATA_PATH"], "analysis", "s3", "data2weeks")
    else:
        folder_2weeks = None

    file_names = [
        el % args.file_name
        for el in [
            "%s_raw.csv",
            "%s_basic.csv",
            "%s_rolling.csv",
            "%s_opt.csv",
            "%s_elec.csv",
            "%s_co2.csv",
        ]
    ]

    ###############################################################################
    if args.make:
        make_dataset(
            start=start,
            end=end,
            file_name=file_name,
            tmp_folder=args.folder_new,
            folder_hist=args.folder_hist,
            scrape=args.scrape,
        )

    if args.update:
        update_dataset(
            args.folder_hist,
            file_names,
            args.folder_new,
            folder_2weeks=folder_2weeks,
        )

    thresh_date = (now - timedelta(hours=24 * 30)).isoformat()
    if args.update_d3map:
        update_d3map(
            args.folder_new, join(args.folder_hist, "d3map"), file_name=file_name, thresh_date=thresh_date
        )

    logger.debug("Writing last_update.txt")
    with open(join(args.folder_hist, "last_update.txt"), "w") as fw:
        fw.write(datetime.utcnow().isoformat())
