import argparse
import logging
import pandas as pd

import gridemissions
from gridemissions import api

from .utils import str2bool


def main():
    """
    Download some data from the API and save to a file.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
    )
    logger = logging.getLogger(__name__)

    # Parse command line options
    # Optionally pass in a path name

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--variable",
        default="co2",
        choices=["co2", "elec", "raw", "co2i"],
        help="Variable to get data for",
    )
    argparser.add_argument("--ba", default="CISO", help='Balancing area')
    argparser.add_argument(
        "--start", default="20200101", help='parseable by pd.to_datetime, e.g. "%%Y%%m%%d"'
    )
    argparser.add_argument(
        "--end", default="20200102", help='parseable by pd.to_datetime, e.g. "%%Y%%m%%d"'
    )
    argparser.add_argument("--field", default="D", choices=["D", "NG", "ID", "TI"])
    argparser.add_argument("--ba2", default=None, help='Second balancing area, if using field="TI "')
    argparser.add_argument("--file_name", default='default')
    argparser.add_argument("--all", default=False, const=True, nargs="?", type=str2bool, help="Whether to download the full dataset")

    args = argparser.parse_args()
    logger.info(args)

    if args.all:
        logger.info(f"Downloading full dataset for {args.variable}")
        download_full_dataset(args.variable)
        return

    res = api.retrieve(
        variable=args.variable,
        ba=args.ba,
        start=args.start,
        end=args.end,
        field=args.field,
        ba2=args.ba2,
        return_type="text"
    )

    if args.file_name == "default":
        if args.ba2 is None:
            ba2 = ""
        else:
            ba2 = args.ba2
        file_name = (
            "_".join([args.variable, args.ba, ba2, args.field, args.start, args.end])
            + ".csv"
        )
    else:
        file_name = args.file_name

    # Save to a file

    with open(file_name, "w") as fw:
        fw.write(res)


def download_full_dataset(variable: str):
    """
    """
    if variable not in ["co2", "elec", "raw"]:
        raise ValueError(f"Unsupported argument {variable}")

    file_name = f"EBA_{variable}.csv.gz"
    from urllib import request
    request.urlretrieve(gridemissions.s3_url + file_name, file_name)
