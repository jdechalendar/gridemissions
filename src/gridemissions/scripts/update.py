#!/usr/bin/env python

import pathlib
import pandas as pd
import argparse
import logging.config
from datetime import datetime, timedelta

import gridemissions as ge
from gridemissions import eia_api_v2
from gridemissions.workflows import make_dataset, update_dataset, update_d3map
from .utils import str2bool

logger = logging.getLogger(__name__)


def main():
    ge.configure_logging(logging.INFO)
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
        default=ge.config["TMP_PATH"],
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
        "--data_extract",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="whether to extract last month",
    )
    args = argparser.parse_args()

    # Configure logging
    if args.debug:
        print("Activating debug mode")
        logging.getLogger("gridemissions").setLevel("DEBUG")
    if ge.config["ENV"] == "vm":
        # Store logs
        log_path = ge.config["DATA_PATH"] / "Logs"
        log_path.mkdir(exist_ok=True)
        log_path = log_path / "log"
        logger.info("Saving logs to: %s" % str(log_path))
        fh = logging.handlers.TimedRotatingFileHandler(log_path, when="midnight")
        if args.debug:
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logging.getLogger("gridemissions").addHandler(fh)

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
    end = end.strftime(eia_api_v2.EIA_DATETIME_FORMAT)
    start = start.strftime(eia_api_v2.EIA_DATETIME_FORMAT)

    if args.folder_hist == "webapp":
        args.folder_hist = ge.config["DATA_PATH"] / "analysis" / "webapp"
    elif args.folder_hist == "local":
        args.folder_hist = ge.config["DATA_PATH"] / "analysis" / "local"
    else:
        args.folder_hist = pathlib.Path(args.folder_hist)

    if args.folder_new == ge.config["TMP_PATH"]:
        args.folder_new = ge.config["TMP_PATH"] / "emissions_app"

    args.folder_new.mkdir(exist_ok=True)

    if args.data_extract:
        folder_extract = ge.config["DATA_PATH"] / "analysis" / "webapp" / "data_extract"
    else:
        folder_extract = None

    file_names = [
        el % args.file_name
        for el in [
            "%s_raw.csv",
            "%s_basic.csv",
            "%s_rolling.csv",
            "%s_opt.csv",
            "%s_elec.csv",
            "%s_co2.csv",
            "%s_co2i.csv",
        ]
    ]

    logger.info(args)
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

    thresh_date = (pd.to_datetime(end) - timedelta(hours=24 * 30)).isoformat()
    if args.update:
        update_dataset(
            args.folder_hist,
            file_names,
            args.folder_new,
            folder_extract=folder_extract,
            thresh_date_extract=thresh_date,
        )

    if args.update_d3map:
        update_d3map(
            args.folder_new,
            args.folder_hist / "d3map",
            file_name=file_name,
            thresh_date=thresh_date,
        )

    logger.debug("Writing last_update.txt")
    with open(args.folder_hist / "last_update.txt", "w") as fw:
        fw.write(datetime.utcnow().isoformat())
