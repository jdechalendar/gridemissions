from gridemissions import api
import argparse
import logging
import pandas as pd


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
        choices=["co2", "elec", "raw"],
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

    args = argparser.parse_args()
    logger.info(args)

    res = api.retrieve(
        variable=args.variable,
        ba=args.ba,
        start=args.start,
        end=args.end,
        field=args.field,
        ba2=args.ba2,
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
