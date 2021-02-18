"""
"""
import os
import shutil
from os.path import join
import time
import logging
import pandas as pd
from seed import config
from seed.load import BaData
from seed.eia_api import EBA_data_scraper, load_eia_columns
from seed.clean import BaDataBasicCleaner, BaDataCvxCleaner
from seed.clean import BaDataRollingCleaner
from seed.emissions import BaDataEmissionsCalc
from seed.viz.d3map import create_graph


# Optimization-based cleaning is different pre and post July 2018
THRESH_DATE = pd.to_datetime("20180701", utc=True)


def make_dataset(
    start="20191110T08Z",
    end="20191115T08Z",
    file_name="seed_data",
    tmp_folder="tmp",
    folder_hist=None,
    scrape=True,
):
    """
    Make dataset between two dates

    Pull fresh data from the EIA API between `start` and `end`, then run the
    data through the cleaning workflow before computing consumption emissions.

    Uses historical data if available.
    """
    start_time = time.time()

    logger = logging.getLogger("scraper")
    eia_columns = load_eia_columns()

    if scrape:  # else: assume that the file exists
        # Scrape EIA data
        logger.info("Scraping EIA data from %s to %s" % (start, end))
        scraper = EBA_data_scraper()
        df = scraper.scrape(eia_columns, start=start, end=end, split_calls=True)
        os.makedirs(tmp_folder, exist_ok=True)
        df.to_csv(join(tmp_folder, "%s_raw.csv" % file_name))

    # Basic data cleaning
    logger.info("Basic data cleaning")
    data = BaData(fileNm=join(tmp_folder, "%s_raw.csv" % file_name))
    cleaner = BaDataBasicCleaner(data)
    cleaner.process()
    cleaner.r.df.to_csv(join(tmp_folder, "%s_basic.csv" % file_name))
    data = cleaner.r

    weights = None
    if folder_hist is not None:  # Rolling-window-based data cleaning
        logger.info("Rolling window data cleaning")
        data = BaData(fileNm=join(tmp_folder, "%s_basic.csv" % file_name))
        cleaner = BaDataRollingCleaner(data)
        cleaner.process(file_name, folder_hist)
        cleaner.r.df.to_csv(join(tmp_folder, "%s_rolling.csv" % file_name))
        data = cleaner.r
        weights = cleaner.weights
        weights.to_csv(join(tmp_folder, "%s_weights.csv" % file_name))
    else:
        logger.warning("No rolling window data cleaning!")

    if len(data.df.loc[:THRESH_DATE, :]) > 0:
        logger.info(f"Optimization-based cleaning without src data: pre {THRESH_DATE}")
        ba_data = BaData(df=data.df.loc[:THRESH_DATE, :]) 
        if weights is not None:
            cleaner = BaDataCvxCleaner(ba_data, weights=weights.loc[:THRESH_DATE, :])
        else:
            cleaner = BaDataCvxCleaner(ba_data)
        cleaner.process(debug=False, with_ng_src=False)
        cleaner.r.df.to_csv(join(tmp_folder, "%s_opt_no_src.csv" % file_name))
        cleaner.CleaningObjective.to_csv(join(tmp_folder, "%s_objective_no_src.csv" % file_name))

    # Only keep going if we have data post THRESH_DATE
    if len(data.df.loc[THRESH_DATE:, :]) == 0:
        return

    logger.info(f"Optimization-based cleaning with src data: post {THRESH_DATE}")
    data.df = data.df.loc[THRESH_DATE:, :]
    if weights is not None:
        cleaner = BaDataCvxCleaner(data, weights=weights.loc[THRESH_DATE:, :])
    else:
        cleaner = BaDataCvxCleaner(data)
    cleaner.process(debug=False)
    cleaner.r.df.to_csv(join(tmp_folder, "%s_opt.csv" % file_name))
    cleaner.CleaningObjective.to_csv(join(tmp_folder, "%s_objective.csv" % file_name))

    # Post-processing (none for now)
    cleaner.r.df.to_csv(join(tmp_folder, "%s_elec.csv" % file_name))
    data = cleaner.r

    # Consumption-based emissions
    logger.info("Computing consumption-based emissions")
    co2_calc = BaDataEmissionsCalc(data)
    co2_calc.process()
    co2_calc.poll_data.df.to_csv(join(tmp_folder, "%s_co2.csv" % file_name))

    logger.info(
        "seed.workflows.make_dataset took %.2f seconds" % (time.time() - start_time)
    )


def update_dataset(
    folder_hist, file_names, folder_new="tmp", folder_2weeks=None
):
    """
    Update dataset in storage with new data.

    Assumes fresh data has just been pulled into a temporary working folder.
    Deletes that folder when updating is finished.
    """
    logger = logging.getLogger("scraper")
    os.makedirs(folder_hist, exist_ok=True)
    if folder_2weeks is not None:
        os.makedirs(folder_2weeks, exist_ok=True)
        logger.info(f"Creating {folder_2weeks}")
    for file_name in file_names:
        _update_dataset(folder_hist, file_name, folder_new)


    if folder_2weeks is not None:
        for file_name in file_names:
            _extract_last2weeks(folder_hist, file_name, folder_2weeks)

    # Remove
    logger.warning("Removal of temporary folder not yet implemented")


def _extract_last2weeks(folder_hist, file_name, folder_2weeks):
    """
    Helper for `update_dataset` to save last 2 weeks
    """
    logger = logging.getLogger("scraper")
    file_hist = join(folder_hist, file_name)
    file_new = join(folder_2weeks, file_name)

    def load_file(x):
        logger.info("Reading %s" % x)
        return pd.read_csv(x, index_col=0, parse_dates=True)

    logger.debug("Saving 2 weeks for %s" % file_name)
    df_hist = load_file(file_hist)
    df_hist.iloc[-24 * 14 :].to_csv(file_new)


def _update_dataset(folder_hist, file_name, folder_new="tmp"):
    """
    Helper for `udpate_dataset`

    Note: we prioritize new rows over old rows, when merging the old dataframe
    with the new incoming data. Accordingly, we look for the rows in the old
    dataset that are not in the new dataset, and append them to the new dataset
    """
    logger = logging.getLogger("scraper")
    # folder_name = join(config["DATA_PATH"], "analysis", "webapp")
    file_hist = join(folder_hist, file_name)
    file_new = join(folder_new, file_name)

    def load_file(x):
        logger.info("Reading %s" % x)
        return pd.read_csv(x, index_col=0, parse_dates=True)

    try:
        df_hist = load_file(file_hist)
        df_new = load_file(file_new)
        old_rows = df_hist.index.difference(df_new.index)
        df_hist = df_new.append(df_hist.loc[old_rows, :], sort=True)
        n_new = len(df_new.index.difference(df_hist.index))
        n_updated = len(df_new) - n_new
    except FileNotFoundError:
        logger.info("file_hist: %s" % file_hist)
        logger.info("file_new: %s" % file_new)
        logger.info("No history file was found, starting a new one.")
        df_hist = load_file(file_new)
        n_new = len(df_hist)
        n_updated = n_new
    logger.info("Added: %d / Updated: %d" % (n_new, n_updated))

    logger.debug("Sorting index")
    df_hist.sort_index(inplace=True)

    logger.debug("Saving history")
    df_hist.to_csv(file_hist)


def update_d3map(folder_in, folder_out, file_name, thresh_date="2000-01-01"):
    poll = BaData(fileNm=join(folder_in, f"{file_name}_co2.csv"), variable="CO2")
    elec = BaData(fileNm=join(folder_in, f"{file_name}_elec.csv"), variable="E")

    # Remove old map data
    shutil.rmtree(folder_out)
    os.makedirs(folder_out, exist_ok=True)

    for ts in poll.df.loc[thresh_date:,:].index:
        _ = create_graph(poll, elec, ts, folder_out=folder_out, save_data=True)


if __name__ == "__main__":
    make_dataset()
