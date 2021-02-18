import os
from os.path import join
import pandas as pd
import argparse
import logging.config
from seed import config
from seed.ampd import AMPD_download, extract_state


def main():
    # Parse command-line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--job", default="d", help="Which job to run")
    argparser.add_argument("--state", default="all", help="Which state to process")
    argparser.add_argument("--year", default="all", help="Which year to process")
    args = argparser.parse_args()

    logger = logging.getLogger("scraper")
    DATA_PATH = config["DATA_PATH"]

    # Download data
    if args.job == "d":
        if args.year == "all":
            raise NotImplemented("Not done yet")
        count_timed_out = 0
        status = "timedout"
        y = int(args.year)
        while status == "timedout":
            ampd = AMPD_download(os.path.join(DATA_PATH, "raw", "AMPD"))
            ampd.login()
            ampd.getStatus(y)
            status = ampd.download(y)
            if status == "timed_out":
                count_timed_out += 1
            if count_timed_out > 5:
                raise ValueError("Too many timeouts")
            ampd.close()

    if args.job == "e":
        if args.state == "all":
            raise NotImplemented("Not done yet")
        if args.year == "all":
            year_lst = ["2015", "2016", "2017", "2018", "2019"]
        else:
            year_lst = [args.year]

        for year in year_lst:
            path_in = join(config["DATA_PATH"], "raw", "AMPD", year)
            path_out = join(config["DATA_PATH"], "analysis", "AMPD")
            os.makedirs(path_out, exist_ok=True)
            extract_state((path_in, path_out, year, args.state))
