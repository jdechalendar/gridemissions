import argparse
import logging
import tarfile
from urllib import request

import gridemissions as ge
from gridemissions import api

from .utils import str2bool

logger = logging.getLogger(__name__)


def main():
    """
    Download some data from the API and save to a file.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
    )

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
        "--bulk",
        default=False,
        const=True,
        nargs="?",
        type=str2bool,
        help="Whether to download the bulk dataset",
    )

    args = argparser.parse_args()
    logger.info(args)

    if args.bulk:
        download_bulk_dataset()
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
        file_name = ge.config["DATA_PATH"] / (
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


def download_bulk_dataset():
    path = ge.config["DATA_PATH"] / "EIA_Grid_Monitor" / "processed.tar.gz"
    path.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"Downloading bulk dataset to {path}...")
    request.urlretrieve(
        "https://gridemissions.s3.us-east-2.amazonaws.com/processed.tar.gz", path
    )
    logger.info("Extracting...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=path.parent)
