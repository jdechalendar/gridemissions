import argparse
import gzip
import logging
import os
import pathlib
from urllib import request
import shutil
from typing import Union

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
        "--dataset",
        default="co2",
        choices=["co2", "elec", "raw", "co2i"],
        help="Variable to get data for",
    )
    argparser.add_argument("--region", default=None, help="Region")
    argparser.add_argument(
        "--start",
        default=None,
        help='parseable by pd.to_datetime, e.g. "%%Y%%m%%d"',
    )
    argparser.add_argument(
        "--end",
        default=None,
        help='parseable by pd.to_datetime, e.g. "%%Y%%m%%d"',
    )
    argparser.add_argument("--field", default=None)
    argparser.add_argument(
        "--region2", default=None, help='Second region, if using field="ID"'
    )
    argparser.add_argument("--file_name", default=None)
    argparser.add_argument(
        "--all",
        default=False,
        const=True,
        nargs="?",
        type=str2bool,
        help="Whether to download the full dataset",
    )

    args = argparser.parse_args()
    logger.info(args)

    if args.all:
        logger.info(f"Downloading full dataset: {args.dataset}")
        download_full_dataset(args.dataset, args.file_name)
        return

    res = api.retrieve(
        dataset=args.dataset,
        region=args.region,
        start=args.start,
        end=args.end,
        field=args.field,
        region2=args.region2,
        return_type="text",
    )

    if args.file_name is None:
        file_name = gridemissions.config["DATA_PATH"] / (
            "_".join(
                [
                    args.dataset,
                    args.region or "",
                    args.region2 or "",
                    args.field or "",
                    args.start or "",
                    args.end or "",
                ]
            )
            + ".csv"
        )
    else:
        file_name = args.file_name

    # Save to a file

    print(f"Downloading to {file_name}")
    with open(file_name, "w") as fw:
        fw.write(res)


def download_full_dataset(dataset: str, path_out: Union[str, os.PathLike, None] = None):
    """ """
    if dataset not in ["co2", "elec", "raw"]:
        raise ValueError(f"Unsupported argument {dataset}")

    fname = f"EBA_{dataset}.csv.gz"
    if path_out is None:
        path_out = gridemissions.config["DATA_PATH"] / fname
    else:
        path_out = pathlib.Path(path_out)

    if not path_out.name.endswith(".csv.gz"):
        raise ValueError(f"path_out should end in .csv.gz but got {path_out}")
    path_out_csv = path_out.parent / path_out.stem

    print(f"Downloading to {path_out}...")
    request.urlretrieve(gridemissions.config["S3_URL"] + fname, path_out)

    print(f"Decompressing to {path_out_csv}...")
    with gzip.open(path_out, "rb") as f_in:
        with open(path_out_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
