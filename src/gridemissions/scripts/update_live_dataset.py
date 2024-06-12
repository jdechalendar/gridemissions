#!/usr/bin/env python

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
        "-um",
        "--update_d3map",
        default=True,
        nargs="?",
        const=True,
        type=str2bool,
        help="Update d3 map",
    )
    argparser.add_argument("--start", default="now", help="start date (%%Y-%%m-%%d)")
    argparser.add_argument("--end", default="now", help="end date (%%Y-%%m-%%d)")
    argparser.add_argument(
        "-d",
        "--debug",
        default=False,
        const=True,
        nargs="?",
        type=str2bool,
        help="Whether to include debug information in logging",
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
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

    folder_hist = ge.config["DATA_PATH_LIVE"] / "hist"
    base_name = "live"

    logger.info(args)

    ###############################################################################
    if args.make:
        make_dataset(
            ge.config["DATA_PATH_LIVE"],
            start=start,
            end=end,
            base_name=base_name,
            folder_hist=folder_hist,
            scrape=args.scrape,
        )

        logger.info("Updating dataset")
        update_dataset(
            folder_hist,
            ge.config["DATA_PATH_LIVE"],
            cutoff_date=(pd.to_datetime(end) - timedelta(hours=24 * 30)).isoformat(),
        )

    if args.update_d3map:
        update_d3map(
            ge.config["DATA_PATH_LIVE"],
            ge.config["DATA_PATH_LIVE"] / "d3map",
            base_name=base_name,
        )

    logger.debug("Writing last_update.txt")
    with open(ge.config["DATA_PATH_LIVE"] / "last_update.txt", "w") as fw:
        fw.write(datetime.utcnow().isoformat())
