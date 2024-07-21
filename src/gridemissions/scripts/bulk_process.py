import logging
import pandas as pd

import gridemissions as ge
from gridemissions.eia_bulk_grid_monitor import (
    parse_balance_files,
    parse_interchange_files,
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
    bfiles = []
    ifiles = []
    for bfile in balance_files:
        ifile = bfile.parent / bfile.name.replace(
            "EIA930_BALANCE", "EIA930_INTERCHANGE"
        )

        # ifile.is_file() -> we have both the balance file and the interchange file
        if ifile.is_file():
            bfiles.append(bfile)
            ifiles.append(ifile)
    print(f"Collected {len(bfiles)} six-month datasets to process")

    # Convert data to a format our workflow can process
    logger.info(f"Parsing {len(bfiles)} balance files")
    df_balance = parse_balance_files(bfiles)

    logger.info(f"Parsing {len(ifiles)} interchange files")
    df_interchange = parse_interchange_files(ifiles)

    df = pd.concat([df_balance, df_interchange], axis=1)

    # Re-save dataset in 6 month files to follow what the EIA does
    # use a duplicate index to make sure we are not changing the formatting
    # of the index before we write to the filesystem
    files = []
    idx = pd.to_datetime(df.index)
    for year in idx.year.unique():
        for firstpart in [True, False]:
            if firstpart:
                df_ = df.loc[(idx.year == year) & (idx.month.isin(range(7)))]
                filename_out = f"EIA930_{year}_Jan_Jun"
            else:
                df_ = df.loc[(idx.year == year) & ~(idx.month.isin(range(7)))]
                filename_out = f"EIA930_{year}_Jul_Dec"
            if len(df_) > 0:
                df_.to_csv(folder_out / f"{filename_out}_raw.csv")
                files.append(filename_out)

    # Run the rest of the gridemissions workflow on these new files
    for filename_out in files:
        make_dataset(
            base_name=filename_out,
            folder_out=folder_out,
            scrape=False,
        )
