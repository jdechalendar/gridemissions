import os
import argparse
from gridemissions import config
from gridemissions.ampd import AMPD_download, extract_state


def main():
    # Parse command-line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--job", default="d", help="Which job to run")
    argparser.add_argument("--state", default="all", help="Which state to process")
    argparser.add_argument("--year", default="all", help="Which year to process")
    args = argparser.parse_args()

    DATA_PATH = config["DATA_PATH"]

    # Download data
    if args.job == "d":
        if args.year == "all":
            raise NotImplementedError("Not done yet")
        count_timed_out = 0
        status = "timedout"
        y = int(args.year)
        while status == "timedout":
            ampd = AMPD_download(DATA_PATH / "raw" / "AMPD")
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
            raise NotImplementedError("Not done yet")
        if args.year == "all":
            year_lst = ["2015", "2016", "2017", "2018", "2019"]
        else:
            year_lst = [args.year]

        for year in year_lst:
            path_in = config["DATA_PATH"] / "raw" / "AMPD" / year
            path_out = config["DATA_PATH"] / "analysis" / "AMPD"
            os.makedirs(path_out, exist_ok=True)
            extract_state((path_in, path_out, year, args.state))
