import random
import logging
import time

from os.path import join, isdir
import os
import pandas as pd
import numpy as np
import gridemissions
from gridemissions.load import BaData
from gridemissions.workflows import make_dataset


def run_test(i="", level=0.2, debug=False):
    # Load raw data and restrict to a 2 day test period
    file_name_raw = join(gridemissions.config["APEN_PATH"], "data", "EBA_raw.csv")
    data_raw = BaData(fileNm=file_name_raw)

    start = pd.to_datetime("2020-11-01T00:00Z")
    end = pd.to_datetime("2020-11-03T00:00Z")
    data_raw.df = data_raw.df.loc[start:end]

    # Create a copy of the test dataset and modify it
    data_raw_copy = BaData(df=data_raw.df.copy(deep=True))
    data_raw_copy.df.loc[
        :, data_raw_copy.get_cols("CISO", "D")[0]
    ] *= np.random.uniform(1 - level, 1 + level, len(data_raw_copy.df))

    # Set up test folder and save data to the folder
    tmp_folder = join(gridemissions.config["APEN_PATH"], "si_test4", f"{i}", "tmp")
    os.makedirs(tmp_folder, exist_ok=True)
    data_raw_copy.df.to_csv(join(tmp_folder, "EBA_raw.csv"))

    # Load historical data and restrict to 15 days before when we are testing
    folder_hist = join(gridemissions.config["APEN_PATH"], "si_test4", "hist")
    if ~isdir(folder_hist):
        file_name_basic = join(
            gridemissions.config["APEN_PATH"], "data", "EBA_basic.csv"
        )
        data_basic = BaData(fileNm=file_name_basic)
        end_hist = start
        start_hist = end_hist - pd.Timedelta("15D")
        data_basic.df = data_basic.df.loc[start_hist:end_hist]

        os.makedirs(folder_hist, exist_ok=True)
        data_basic.df.to_csv(join(folder_hist, "EBA_basic.csv"))

    # Run workflow on fake dataset
    make_dataset(
        tmp_folder=tmp_folder,
        folder_hist=folder_hist,
        scrape=False,
    )

    # Reload results
    file_name = join(tmp_folder, "EBA_%s.csv")
    raw = BaData(fileNm=file_name % "raw")
    opt = BaData(fileNm=file_name % "opt")

    # Compute error
    d_col = raw.get_cols("CISO", "D")[0]
    error = (
        (data_raw.df.loc[start:end, d_col] - opt.df.loc[:, d_col]).abs()
        / data_raw.df.loc[start:end, d_col]
    ).mean()

    if debug:
        basic = BaData(fileNm=file_name % "basic")
        rolling = BaData(fileNm=file_name % "rolling")
        return error, raw, basic, rolling, opt, data_raw

    return error


if __name__ == "__main__":
    random.seed(8765)

    for lg in ["load", "scraper", "clean"]:
        logging.getLogger(lg).setLevel(logging.ERROR)

    results = {}
    levels = [0.05, 0.1, 0.2, 0.3, 0.4]

    for lev in levels:
        start_time = time.time()
        print(f"Processing level {lev}")

        results[lev] = [run_test(i, lev) for i in range(20)]
        print(f"{time.time()-start_time:.2f} seconds")

    results = pd.DataFrame(results)
    print(results)
    results.to_csv(join(gridemissions.config["APEN_PATH"], "si_test4", "results.csv"))
