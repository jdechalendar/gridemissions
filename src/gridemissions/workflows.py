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
    folder_out: pathlib.Path,
    start=None,
    end=None,
    base_name: str = "data",
    folder_hist: pathlib.Path = pathlib.Path("NOFOLDER"),
    scrape: bool = True,
):
    """
    Make gridemissions dataset

    Parameters
    ----------
    folder_out: pathlib.Path
        folder where the dataset is made
    start: datetime-like, optional
        used when scraping new data
    end: datetime-like, optional
        used when scraping new data
    base_name: str, default "data"
        starting name for data files
    folder_hist: pathlib.Path, optional
        Historical data to use for `ge.RollingCleaner`.
        Looks for a file called `folder_hist / f"{base_name}_basic.csv"`
        Pass `None` to disable rolling window cleaning
    scrape: bool, default `True`

    Notes
    -----
    If `scrape`, pull fresh data from the EIA API between `start` and `end`.
    Otherwise, assume the starting file already exists and is called
    f"{base_name}_raw.csv".

    Rolling-window cleaning can be disabled, but note that we currently use the
    weights from rolling window cleaning in the reconciliation step.
    """
    start_time = time.time()

    folder_out.mkdir(exist_ok=True)
    file_name_raw = folder_out / f"{base_name}_raw.csv"
    file_name_basic = folder_out / f"{base_name}_basic.csv"

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
        cleaner.process(folder_hist / f"{base_name}_basic.csv")
        cleaner.out.to_csv(folder_out / f"{base_name}_rolling.csv")
        data = cleaner.out
        weights = cleaner.weights
        weights.to_csv(folder_out / f"{base_name}_weights.csv")
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
        cleaner.out.to_csv(folder_out / f"{base_name}_opt_no_src.csv")
        cleaner.CleaningObjective.to_csv(
            folder_out / f"{base_name}_objective_no_src.csv"
        )
        cleaner.deltas.to_csv(folder_out / f"{base_name}_deltas_no_src.csv")

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
    cleaner.out.to_csv(folder_out / f"{base_name}_opt.csv")
    cleaner.CleaningObjective.to_csv(folder_out / f"{base_name}_objective.csv")
    cleaner.deltas.to_csv(folder_out / f"{base_name}_deltas.csv")

    # Post-processing (none for now)
    cleaner.out.to_csv(folder_out / f"{base_name}_elec.csv")
    data = cleaner.out

    # Consumption-based emissions
    logger.info("Computing consumption-based emissions")
    co2_calc = ge.EmissionsCalc(data)
    co2_calc.process()
    co2_calc.poll_data.to_csv(folder_out / f"{base_name}_co2.csv")
    co2_calc.polli_data.to_csv(folder_out / f"{base_name}_co2i.csv")

    logger.info(f"make_dataset took {time.time() - start_time} seconds")


def update_dataset(folder_hist: pathlib.Path, folder_new: pathlib.Path, cutoff_date):
    """
    Update dataset in storage with new data.

    Parameters
    ----------
    folder_hist: pathlib.Path
    folder_new: pathlib.Path
    cutoff_date: datetime-like
        only keep history after this date
    """
    folder_hist.mkdir(exist_ok=True)
    for f in folder_new.iterdir():
        if f.name.endswith(".csv") and (folder_new / f.name).is_file():
            _update_dataset(f.name, folder_hist, folder_new, cutoff_date)


def _update_dataset(
    name: str, folder_hist: pathlib.Path, folder_new: pathlib.Path, cutoff_date
):
    """
    Helper for update_dataset

    Parameters
    ----------
    name : str
    folder_hist : pathlib.Path
    folder_new : pathlib.Path
    cutoff_date : datetime-like

    Notes
    -----
    We prioritize new rows over old rows, when merging the old dataframe
    with the new incoming data. Accordingly, we look for the rows in the old
    dataset that are not in the new dataset, and append them to the new dataset
    """
    file_hist = folder_hist / name
    file_new = folder_new / name

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
    df_hist.loc[cutoff_date:].to_csv(file_hist)


def update_d3map(
    folder_in: pathlib.Path,
    folder_out: pathlib.Path,
    base_name: str,
):
    poll = ge.read_csv(folder_in / f"{base_name}_co2.csv")
    elec = ge.read_csv(folder_in / f"{base_name}_elec.csv")

    # Remove old map data
    shutil.rmtree(folder_out, ignore_errors=True)
    os.makedirs(folder_out, exist_ok=True)

    for ts in poll.df.index:
        _ = create_graph(poll, elec, ts, folder_out=folder_out, save_data=True)
