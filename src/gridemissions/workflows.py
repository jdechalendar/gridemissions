"""
"""
import pathlib
import os
import shutil
import time
import logging
import pandas as pd

import gridemissions as ge
from gridemissions import eia_api_v2 as eia_api
from gridemissions.viz.d3map import create_graph


# Optimization-based cleaning is different pre and post July 2018
THRESH_DATE = pd.to_datetime("20180701")
logger = logging.getLogger(__name__)


def make_dataset(
    start=None,
    end=None,
    file_name="EBA",
    tmp_folder=None,
    folder_hist=None,
    scrape=True,
):
    """
    Make gridemissions dataset

    Parameters
    ----------
    start: datetime-like, optional
    end: datetime-like, optional
    file_name: str, default "EBA"
    tmp_folder: pathlib.Path, optional
        Folder where the dataset is made
    folder_hist: pathlib.Path, optional
        Historical data to use for `ge.RollingCleaner`
    scrape: bool, default `True`

    Notes
    -----
    If `scrape`, pull fresh data from the EIA API between `start` and `end`. Otherwise,
    assume the starting file already exists and is called f"{file_name}_raw.csv".

    Run the data through the cleaning workflow before computing consumption emissions.
    """
    start_time = time.time()
    tmp_folder = tmp_folder or ge.config["TMP_PATH"]

    tmp_folder.mkdir(exist_ok=True)
    file_name_raw = tmp_folder / f"{file_name}_raw.csv"
    file_name_basic = tmp_folder / f"{file_name}_basic.csv"

    if scrape:  # else: assume that the file exists
        # Scrape EIA data
        logger.info(f"Scraping EIA data from {start} to {end}")
        df = eia_api.scrape(start, end)
        df.to_csv(file_name_raw)

    # Basic data cleaning
    logger.info("Basic data cleaning")
    data = ge.read_csv(file_name_raw)

    if len(data.df) == 0:
        raise ValueError(f"Aborting make_dataset: no new data in {file_name_raw}")
    cleaner = ge.BasicCleaner(data)
    cleaner.process()
    cleaner.out.to_csv(file_name_basic)
    data = cleaner.out

    weights = None
    if folder_hist is not None:  # Rolling-window-based data cleaning
        logger.info("Rolling window data cleaning")
        data = ge.read_csv(file_name_basic)
        cleaner = ge.RollingCleaner(data)
        cleaner.process(folder_hist / f"{file_name}_basic.csv")
        cleaner.out.to_csv(tmp_folder / f"{file_name}_rolling.csv")
        data = cleaner.out
        weights = cleaner.weights
        weights.to_csv(tmp_folder / f"{file_name}_weights.csv")
    else:
        logger.warning("No rolling window data cleaning!")

    # Note: The following test throws an error if data.df.index is not monotonic
    # See: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#non-monotonic-indexes-require-exact-matches
    if len(data.df.loc[:THRESH_DATE, :]) > 0:
        logger.info(f"Optimization-based cleaning without fuel data: pre {THRESH_DATE}")
        ba_data = ge.GraphData(df=data.df.loc[:THRESH_DATE, :])
        if weights is not None:
            cleaner = ge.CvxCleaner(ba_data, weights=weights.loc[:THRESH_DATE, :])
        else:
            cleaner = ge.CvxCleaner(ba_data)
        cleaner.process(debug=False, with_ng_src=False)
        cleaner.out.to_csv(tmp_folder / f"{file_name}_opt_no_src.csv")
        cleaner.CleaningObjective.to_csv(
            tmp_folder / f"{file_name}_objective_no_src.csv"
        )

    # Only keep going if we have data post THRESH_DATE
    if len(data.df.loc[THRESH_DATE:, :]) == 0:
        return

    logger.info(f"Optimization-based cleaning with fuel data: post {THRESH_DATE}")
    data.df = data.df.loc[THRESH_DATE:, :]
    if weights is not None:
        cleaner = ge.CvxCleaner(data, weights=weights.loc[THRESH_DATE:, :])
    else:
        cleaner = ge.CvxCleaner(data)
    cleaner.process(debug=False)
    cleaner.out.to_csv(tmp_folder / f"{file_name}_opt.csv")
    cleaner.CleaningObjective.to_csv(tmp_folder / f"{file_name}_objective.csv")

    # Post-processing (none for now)
    cleaner.out.to_csv(tmp_folder / f"{file_name}_elec.csv")
    data = cleaner.out

    # Consumption-based emissions
    logger.info("Computing consumption-based emissions")
    co2_calc = ge.EmissionsCalc(data)
    co2_calc.process()
    co2_calc.poll_data.to_csv(tmp_folder / f"{file_name}_co2.csv")
    co2_calc.polli_data.to_csv(tmp_folder / f"{file_name}_co2i.csv")

    logger.info(
        "gridemissions.workflows.make_dataset took %.2f seconds"
        % (time.time() - start_time)
    )


def update_dataset(
    folder_hist: pathlib.Path,
    file_names: pathlib.Path,
    folder_new=None,
    folder_extract=None,
    thresh_date_extract=None,
):
    """
    Update dataset in storage with new data.

    Assumes fresh data has just been pulled into a temporary working folder.
    Deletes that folder when updating is finished.
    """
    folder_hist.mkdir(exist_ok=True)
    for file_name in file_names:
        _update_dataset(folder_hist, file_name, folder_new)

    if folder_extract is not None:
        logger.info(f"Creating {folder_extract}")
        folder_extract.mkdir(exist_ok=True)
        shutil.rmtree(folder_extract, ignore_errors=True)
        os.makedirs(folder_extract, exist_ok=True)
        for file_name in file_names:
            _extract_data(folder_hist, file_name, folder_extract, thresh_date_extract)

    # Remove
    logger.warning("Removal of temporary folder not yet implemented")


def _extract_data(
    folder_hist: pathlib.Path,
    file_name: str,
    folder_extract: pathlib.Path,
    thresh_date_extract,
):
    """
    Helper for `update_dataset` to save data after a given date
    """
    logger = logging.getLogger("scraper")
    file_hist = folder_hist / file_name
    file_new = folder_extract / file_name

    def load_file(x):
        logger.debug("Reading %s" % x)
        return pd.read_csv(x, index_col=0, parse_dates=True)

    logger.debug(f"Saving data from {thresh_date_extract} for {file_name}")
    df_hist = load_file(file_hist)
    df_hist.loc[thresh_date_extract:].to_csv(file_new)


def _update_dataset(
    folder_hist: pathlib.Path, file_name: str, folder_new: pathlib.Path = None
):
    """
    Helper for `udpate_dataset`

    Note: we prioritize new rows over old rows, when merging the old dataframe
    with the new incoming data. Accordingly, we look for the rows in the old
    dataset that are not in the new dataset, and append them to the new dataset
    """
    folder_new = folder_new or pathlib.Path("tmp")
    file_hist = folder_hist / file_name
    file_new = folder_new / file_name

    def load_file(x):
        logger.debug("Reading %s" % x)
        return pd.read_csv(x, index_col=0, parse_dates=True)

    try:
        df_hist = load_file(file_hist)
        df_new = load_file(file_new)
        old_rows = df_hist.index.difference(df_new.index)
        df_hist = pd.concat([df_new, df_hist.loc[old_rows, :]])
        n_new = len(df_new.index.difference(df_hist.index))
        n_updated = len(df_new) - n_new
    except FileNotFoundError:
        logger.debug("file_hist: %s" % file_hist)
        logger.debug("file_new: %s" % file_new)
        logger.info("No history file was found, starting a new one.")
        df_hist = load_file(file_new)
        n_new = len(df_hist)
        n_updated = n_new
    logger.debug("Added: %d / Updated: %d" % (n_new, n_updated))

    logger.debug("Sorting index")
    df_hist.sort_index(inplace=True)

    logger.debug("Saving history")
    df_hist.to_csv(file_hist)


def update_d3map(folder_in, folder_out, file_name, thresh_date="2000-01-01"):
    poll = ge.read_csv(folder_in / f"{file_name}_co2.csv")
    elec = ge.read_csv(folder_in / f"{file_name}_elec.csv")

    # Remove old map data
    shutil.rmtree(folder_out, ignore_errors=True)
    os.makedirs(folder_out, exist_ok=True)

    for ts in poll.df.loc[thresh_date:, :].index:
        _ = create_graph(poll, elec, ts, folder_out=folder_out, save_data=True)


if __name__ == "__main__":
    make_dataset()
