import logging
import pandas as pd

import gridemissions as ge
from gridemissions.eia_bulk_grid_monitor import (
    parse_balance_file,
    parse_interchange_file,
)
from gridemissions.workflows import make_dataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ge.configure_logging(logging.INFO)
    folder = ge.config["DATA_PATH"] / "EIA_Grid_Monitor"
    folder_in = folder / "downloads"
    folder_out = folder / "processed"
    folder_out.mkdir(exist_ok=True)
    balance_files = [
        f
        for f in folder_in.iterdir()
        if f.suffix == ".csv" and f.stem.startswith("EIA930_BALANCE")
    ]
    files_to_run = []
    for bfile in balance_files:
        ifile = bfile.parent / bfile.name.replace(
            "EIA930_BALANCE", "EIA930_INTERCHANGE"
        )
        co2_file = (
            folder_out / f"{bfile.stem.replace('EIA930_BALANCE', 'EIA930')}_co2.csv"
        )

        # 2 conditions must be met to run a file:
        # ifile.is_file() -> we have both the balance file and the interchange file
        # not co2_file.isfile() -> we have not run it yet
        if ifile.is_file() and not co2_file.is_file():
            files_to_run.append((bfile, ifile))
    print(f"Collected {len(files_to_run)} six-month datasets to process")

    for bfile, ifile in files_to_run:
        filename_out = bfile.stem.replace("EIA930_BALANCE", "EIA930")

        # Convert data to a format our workflow can process
        logger.info(f"Parsing six month files for {filename_out}")
        df = pd.concat(
            [parse_balance_file(bfile), parse_interchange_file(ifile)], axis=1
        )
        df.to_csv(folder_out / f"{filename_out}_raw.csv")

        # Run the rest of the gridemissions workflow on this new file
        make_dataset(
            base_name=filename_out,
            folder_out=folder_out,
            scrape=False,
        )
