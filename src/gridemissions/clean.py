"""
Tools to clean Balancing area data.
A data cleaning step is performed by an object that subclasses
the `Cleaner` class.
"""
import logging
import time
import re
from gridemissions.load import GraphData
from gridemissions import eia_api_v2 as eia
import pandas as pd
import numpy as np
from collections import defaultdict
import cvxpy as cp
import dask


A = 1e4  # MWh
GAMMA = 10  # MWh
EPSILON = 1  # MWh


def na_stats(data, title, cols):
    """
    Print NA statistics for a subset of a dataframe.
    """
    print(
        "%s:\t%.2f%%"
        % (
            title,
            (
                data.df.loc[:, cols].isna().sum().sum()
                / len(data.df)
                / len(data.df.loc[:, cols].columns)
                * 100
            ),
        )
    )


class Cleaner:
    """
    Template class for data cleaning.

    This is mostly just a shell to show how cleaning classes should operate.
    """

    def __init__(self, data: GraphData):
        """
        Parameters
        ----------
        ba_data : GraphData object
        """
        self.d = data
        self._LOGGER = logging.getLogger("gridemissions." + self.__class__.__name__)

    def process(self):
        pass


class BasicCleaner(Cleaner):
    """
    Basic data cleaning class.

    We run this as the first step of the cleaning process.
    - Add missing columns for demand
    - Add missing columns for generation
    - Add missing columns for total interchange
    - Add missing columns for interchange data
    - Filter very unrealistic values
    """

    def process(self):
        self._LOGGER.info(f"Running BasicCleaner for {len(self.d.df)} rows")
        start = time.time()
        data = self.d
        parsed_columns = data.parsed_columns.drop("variable", axis=1).set_index("field")
        BAs_with_NG = parsed_columns.loc["NG"].region.unique()
        if set(BAs_with_NG) != set(parsed_columns.loc["TI"].region):
            raise ValueError(
                "Unexpected: regions with generation do not match those with interchange"
            )

        # Create demand columns for BAs that have generation but not demand
        missing_D_cols = [
            col
            for col in BAs_with_NG
            if col not in parsed_columns.loc["D"].region.unique()
        ]
        self._LOGGER.info(f"Adding demand columns for {len(missing_D_cols)} bas")
        for ba in missing_D_cols:
            # Add 1 MWh for demand
            data.df.loc[:, data.KEY["D"] % ba] = 1.0
            data.df.loc[:, data.KEY["NG"] % ba] += 0.5
            data.df.loc[:, data.KEY["TI"] % ba] -= 0.5

        # Add columns for the BAs that are outside of the US
        foreign_bas = [
            col
            for col in parsed_columns.region2.dropna().unique()
            if col not in BAs_with_NG
        ]
        self._LOGGER.info(
            "Adding demand, generation and TI columns for %d foreign bas"
            % len(foreign_bas)
        )
        for ba in foreign_bas:
            trade_cols = [col for col in data.df.columns if f"{ba}_ID" in col]
            TI = -data.df.loc[:, trade_cols].sum(axis=1)
            data.df.loc[:, data.KEY["TI"] % ba] = TI
            exports = TI.apply(lambda x: max(x, 0))
            imports = TI.apply(lambda x: min(x, 0))
            data.df.loc[:, data.KEY["D"] % ba] = -imports
            data.df.loc[:, data.KEY["NG"] % ba] = exports
            if ba in ["BCHA", "HQT", "MHEB"]:
                # Assume for these Canadian BAs generation is hydro
                data.df.loc[:, data.KEY["WAT"] % ba] = exports
            else:
                # And all others are OTH (other)
                data.df.loc[:, data.KEY["OTH"] % ba] = exports
            for col in trade_cols:
                ba2 = re.split(r"\.|-|_", col)[1]
                data.df.loc[:, data.KEY["ID"] % (ba, ba2)] = -data.df.loc[:, col]

        # Make sure that trade columns exist both ways
        for col in data.get_cols(field="ID"):
            ba = re.split(r"\.|-|_", col)[1]
            ba2 = re.split(r"\.|-|_", col)[2]
            othercol = data.KEY["ID"] % (ba2, ba)
            if othercol not in data.df.columns:
                self._LOGGER.info("Adding %s" % othercol)
                data.df.loc[:, othercol] = -data.df.loc[:, col]

        # Reinitialize object to reinitialize fields
        data = GraphData(df=data.df)

        # Filter unrealistic values using self.reject_dict
        self._create_reject_dict()
        cols = (
            data.get_cols(field="D")
            + data.get_cols(field="NG")
            + data.get_cols(field="TI")
            + data.get_cols(field="ID")
        )
        for col in cols:
            s = data.df.loc[:, col]
            data.df.loc[:, col] = s.where(
                (s >= self.reject_dict[col][0]) & (s <= self.reject_dict[col][1])
            )

        # Do the same for the generation by source columns
        # If there is no generation by source, add one that is OTH
        # Edge case for solar:
        # There are a lot of values at -50 MWh or so during the night. We want
        # to set those to 0, but consider that very negative values (below
        # -1GW) are rejected
        for ba in data.regions:
            missing = True
            for fuel in eia.FUELS:
                col = data.KEY[fuel] % ba
                if col in data.df.columns:
                    missing = False
                    s = data.df.loc[:, col]
                    if fuel == "SUN":
                        self.reject_dict[col] = (-1e3, 200e3)
                    data.df.loc[:, col] = s.where(
                        (s >= self.reject_dict[col][0])
                        & (s <= self.reject_dict[col][1])
                    )
                    if fuel == "SUN":
                        data.df.loc[:, col] = data.df.loc[:, col].apply(
                            lambda x: max(x, 0)
                        )
            if missing:
                data.df.loc[:, data.KEY["OTH"] % ba] = data.df.loc[
                    :, data.KEY["NG"] % ba
                ]

        # Reinitialize fields
        self._LOGGER.info("Reinitializing fields")
        data = GraphData(df=data.df)

        self.out = data

        self._LOGGER.info("Basic cleaning took %.2f seconds" % (time.time() - start))

    def _create_reject_dict(self):
        """
        Create a defaultdict to store ranges outside of which values are
        considered unrealistic.

        The default range is (-1., 200e3) MW. Manual ranges can be set for
        specific columns here if that range is not strict enough.
        """
        reject_dict = defaultdict(lambda: (-1.0, 200e3))
        for col in self.d.get_cols(field="TI"):
            reject_dict[col] = (-100e3, 100e3)
        for col in self.d.get_cols(field="ID"):
            reject_dict[col] = (-100e3, 100e3)
        reject_dict["E_AZPS_D"] = (1.0, 30e3)
        reject_dict["E_BANC_D"] = (1.0, 6.5e3)
        reject_dict["E_BANC_TI"] = (-5e3, 5e3)
        reject_dict["E_CISO_NG"] = (5e3, 60e3)
        self.reject_dict = reject_dict


def rolling_window_filter(
    df,
    offset=10 * 24,
    min_periods=100,
    center=True,
    replace_nan_with_mean=True,
    return_mean=False,
):
    """
    Apply a rolling window filter to a dataframe.

    Filter using dynamic bounds: reject points that are farther than 4 standard
    deviations from the mean, using a rolling window to compute the mean and
    standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to filter
    offset : int
        Passed on to pandas' rolling function
    min_periods : int
        Passed on to pandas' rolling function
    center : bool
        Passed on to pandas' rolling function
    replace_nan_with_mean : bool
        Whether to replace NaNs with the mean of the timeseries at the end of
        the procedure

    Notes
    -----
    Keeps at least 200 MWh around the mean as an acceptance range.
    """
    for col in df.columns:
        rolling_ = df[col].rolling(offset, min_periods=min_periods, center=center)
        mean_ = rolling_.mean()
        std_ = rolling_.std().apply(lambda x: max(100, x))
        ub = mean_ + 4 * std_
        lb = mean_ - 4 * std_
        idx_reject = (df[col] >= ub) | (df[col] <= lb)
        df.loc[idx_reject, col] = np.nan
        if replace_nan_with_mean:
            # First try interpolating linearly, but only for up to 3 hours
            df.loc[:, col] = df.loc[:, col].interpolate(limit=3)
            # If there is more than 3 hours of missing data, use rolling mean
            df.loc[df[col].isnull(), col] = mean_.loc[df[col].isnull()]

    if return_mean:
        mean_ = df.rolling(offset, min_periods=min_periods, center=center).mean()
        return (df, mean_)
    return df


class RollingCleaner(Cleaner):
    """
    Rolling window cleaning.

    This applies the `rolling_window_filter` function to the dataset. In order
    to apply this properly to the beginning of the dataset, we load past data
    that will be used for the cleaning - that is then dropped.
    """

    def process(self, file_name_hist="", nruns=2):
        """
        Processor function for the cleaner object.

        Parameters
        ----------
        file_name_hist : str
            Base name of the file from which to read historical data.
            Typically, this is old output from `BasicCleaner`
        nruns : int
            Number of times to apply the rolling window procedure

        Notes
        -----
        If we are not processing a large amount of data at a time, we may not
        have enough data to appropriately estimate the rolling mean and
        standard deviation for the rolling window procedure. If a value is
        given for `file_name_hist`, data will be read from a
        historical dataset to estimate the rolling mean and standard deviation.
        If there are very large outliers, they can 'mask' smaller outliers.
        Running the rolling window procedure a couple of times helps with this
        issue.
        """
        self._LOGGER.info("Running RollingCleaner (%d runs)" % nruns)
        start = time.time()
        data = self.d

        # Remember what part we are cleaning
        idx_cleaning = data.df.index

        try:
            # Load the data we already have in memory
            df_hist = pd.read_csv(file_name_hist, index_col=0, parse_dates=True)

            # Only take the last 1,000 rows
            # Note that if df_hist has less than 1,000 rows,
            # pandas knows to select df_hist without raising an error.
            df_hist = df_hist.iloc[-1000:]

            # Overwrite with the new data
            old_rows = df_hist.index.difference(data.df.index)
            df_hist = pd.concat([data.df, df_hist.loc[old_rows, :]])
            df_hist.sort_index(inplace=True)

        except FileNotFoundError:
            self._LOGGER.info("No history file")
            df_hist = data.df

        # Apply rolling horizon threshold procedure
        # 20200206 update: don't try replacing NaNs anymore, leave that to the
        # next step
        for _ in range(nruns):
            df_hist = rolling_window_filter(df_hist, replace_nan_with_mean=False)

        # Deal with NaNs
        # First deal with NaNs by taking the average of the previous day and
        # next day. In general we observe strong daily patterns so this seems
        # to work well. Limit the filling procedure to one day at a time. If
        # there are multiple missing days, this makes for a smoother transition
        # between the two valid days. If we had to do this more than 4 times,
        # give up and use forward and backward fills without limits
        for col in df_hist.columns:
            npasses = 0
            while (df_hist.loc[:, col].isna().sum() > 0) and (npasses < 4):
                npasses += 1
                df_hist.loc[:, col] = pd.concat(
                    [
                        df_hist.loc[:, col].groupby(df_hist.index.hour).ffill(limit=1),
                        df_hist.loc[:, col].groupby(df_hist.index.hour).bfill(limit=1),
                    ],
                    axis=1,
                ).mean(axis=1)
            if npasses == 4:
                self._LOGGER.debug("A lot of bad data for %s" % col)
                df_hist.loc[:, col] = pd.concat(
                    [
                        df_hist.loc[:, col].groupby(df_hist.index.hour).ffill(),
                        df_hist.loc[:, col].groupby(df_hist.index.hour).bfill(),
                    ],
                    axis=1,
                ).mean(axis=1)

            # All bad data columns
            if df_hist.loc[:, col].isna().sum() == len(df_hist):
                df_hist.loc[:, col] = 0.0

        # Some NaNs will still remain - try using the rolling mean average
        df_hist, mean_ = rolling_window_filter(
            df_hist, replace_nan_with_mean=True, return_mean=True
        )

        if df_hist.isna().sum().sum() > 0:
            self._LOGGER.warning("There are still some NaNs. Unexpected")

        # Just keep the indices we are working on currently
        data = GraphData(df=df_hist.loc[idx_cleaning, :])

        self.out = data
        self.weights = mean_.loc[idx_cleaning, :].applymap(lambda x: A / max(GAMMA, abs(x)))

        self._LOGGER.info(
            "Rolling window cleaning took %.2f seconds" % (time.time() - start)
        )


class CvxCleaner(Cleaner):
    """
    Optimization-based cleaning class.

    Uses cvxpy.
    """

    def __init__(self, ba_data, weights=None):
        super().__init__(ba_data)
        self.weights = weights
        if weights is not None:
            self.d.df = pd.concat(
                [self.d.df, weights.rename(lambda x: x + "_W", axis=1)], axis=1
            )

    def process(self, debug=False, with_ng_src=True):
        key = eia.KEYS["E"]
        start = time.time()
        self._LOGGER.info("Running CvxCleaner for %d rows" % len(self.d.df))
        self.d.df = self.d.df.fillna(0)

        results = []

        def cvx_solve(row, regions, debug=False):
            if row.isna().sum() > 0:
                raise ValueError("Cannot call this method on data with NaNs")

            n_regions = len(regions)

            D = row[[key["D"] % r for r in regions]].values
            D_W = [
                el**0.5 for el in row[[key["D"] % r + "_W" for r in regions]].values
            ]
            NG = row[[key["NG"] % r for r in regions]].values
            NG_W = [
                el**0.5 for el in row[[key["NG"] % r + "_W" for r in regions]].values
            ]
            TI = row[[key["TI"] % r for r in regions]].values
            TI_W = [
                el**0.5 for el in row[[key["TI"] % r + "_W" for r in regions]].values
            ]

            delta_D = cp.Variable(n_regions, name="delta_D")
            delta_NG = cp.Variable(n_regions, name="delta_NG")
            delta_TI = cp.Variable(n_regions, name="delta_TI")

            obj = (
                cp.sum_squares(cp.multiply(D_W, delta_D))
                + cp.sum_squares(cp.multiply(NG_W, delta_NG))
                + cp.sum_squares(cp.multiply(TI_W, delta_TI))
            )

            ID = {}
            ID_W = {}
            for i, ri in enumerate(regions):
                for j, rj in enumerate(regions):
                    if key["ID"] % (ri, rj) in row.index:
                        ID[(ri, rj)] = row[key["ID"] % (ri, rj)]
                        ID_W[(ri, rj)] = row[key["ID"] % (ri, rj) + "_W"]
            delta_ID = {k: cp.Variable(name=f"{k}") for k in ID}
            constraints = [
                D + delta_D >= 1.0,
                NG + delta_NG >= 1.0,
                D + delta_D + TI + delta_TI - NG - delta_NG == 0.0,
            ]

            if with_ng_src:
                NG_SRC = {}
                NG_SRC_W = {}

                for i, src in enumerate(eia.FUELS):
                    for j, r in enumerate(regions):
                        if key[src] % r in row.index:
                            NG_SRC[(src, r)] = row[key[src] % r]
                            NG_SRC_W[(src, r)] = row[key[src] % r + "_W"]
                delta_NG_SRC = {k: cp.Variable(name=f"{k}") for k in NG_SRC}

                for k in NG_SRC:
                    constraints += [NG_SRC[k] + delta_NG_SRC[k] >= 1.0]
                    obj += NG_SRC_W[k] * delta_NG_SRC[k] ** 2

            # Add the antisymmetry constraints twice is less efficient but not a huge deal.
            for ri, rj in ID:  # then (rj, ri) must also be in ID
                constraints += [
                    ID[(ri, rj)]
                    + delta_ID[(ri, rj)]
                    + ID[(rj, ri)]
                    + delta_ID[(rj, ri)]
                    == 0.0
                ]
                obj += ID_W[(ri, rj)] * delta_ID[(ri, rj)] ** 2

            for i, ri in enumerate(regions):
                if with_ng_src:
                    constraints += [
                        NG[i]
                        + delta_NG[i]
                        - cp.sum(
                            [
                                NG_SRC[(src, ri)] + delta_NG_SRC[(src, ri)]
                                for src in eia.FUELS
                                if (src, ri) in NG_SRC
                            ]
                        )
                        == 0.0
                    ]
                constraints += [
                    TI[i]
                    + delta_TI[i]
                    - cp.sum(
                        [
                            ID[(ri, rj)] + delta_ID[(ri, rj)]
                            for rj in regions
                            if (ri, rj) in ID
                        ]
                    )
                    == 0.0
                ]
            objective = cp.Minimize(obj)

            prob = cp.Problem(objective, constraints)
            prob.solve()

            if with_ng_src:
                r = pd.concat(
                    [
                        pd.Series(
                            NG + delta_NG.value,
                            index=[key["NG"] % r for r in regions],
                        ),
                        pd.Series(
                            D + delta_D.value,
                            index=[key["D"] % r for r in regions],
                        ),
                        pd.Series(
                            TI + delta_TI.value,
                            index=[key["TI"] % r for r in regions],
                        ),
                        pd.Series(
                            {key["ID"] % k: ID[k] + delta_ID[k].value for k in ID}
                        ),
                        pd.Series(
                            {
                                key[s] % r: NG_SRC[(s, r)] + delta_NG_SRC[(s, r)].value
                                for (s, r) in NG_SRC
                            }
                        ),
                        pd.Series({"CleaningObjective": prob.value}),
                    ]
                )
            else:
                r = pd.concat(
                    [
                        pd.Series(
                            NG + delta_NG.value,
                            index=[key["NG"] % r for r in regions],
                        ),
                        pd.Series(
                            D + delta_D.value,
                            index=[key["D"] % r for r in regions],
                        ),
                        pd.Series(
                            TI + delta_TI.value,
                            index=[key["TI"] % r for r in regions],
                        ),
                        pd.Series(
                            {key["ID"] % k: ID[k] + delta_ID[k].value for k in ID}
                        ),
                        pd.Series({"CleaningObjective": prob.value}),
                    ]
                )

            if not debug:
                return r

            if with_ng_src:
                deltas = pd.concat(
                    [
                        pd.Series(
                            delta_NG.value, index=[key["NG"] % r for r in regions]
                        ),
                        pd.Series(delta_D.value, index=[key["D"] % r for r in regions]),
                        pd.Series(
                            delta_TI.value, index=[key["TI"] % r for r in regions]
                        ),
                        pd.Series({key["ID"] % k: delta_ID[k].value for k in ID}),
                        pd.Series(
                            {
                                key[s] % r: delta_NG_SRC[(s, r)].value
                                for (s, r) in NG_SRC
                            }
                        ),
                    ]
                )
            else:
                deltas = pd.concat(
                    [
                        pd.Series(
                            delta_NG.value, index=[key["NG"] % r for r in regions]
                        ),
                        pd.Series(delta_D.value, index=[key["D"] % r for r in regions]),
                        pd.Series(
                            delta_TI.value, index=[key["TI"] % r for r in regions]
                        ),
                        pd.Series({key["ID"] % k: delta_ID[k].value for k in ID}),
                    ]
                )
            return pd.concat([r, deltas.rename(lambda x: x + "_Delta")])

        cvx_solve = dask.delayed(cvx_solve)
        for idx, row in self.d.df.iterrows():
            results.append(cvx_solve(row, self.d.regions, debug=debug))
        results = dask.compute(*results, scheduler="processes")
        df = pd.DataFrame(results, index=self.d.df.index)

        self.out = df.loc[
            :,
            [
                c
                for c in df.columns
                if "Delta" not in c and "CleaningObjective" not in c
            ],
        ]
        self.CleaningObjective = df.CleaningObjective
        self.deltas = df.loc[:, [c for c in df.columns if "Delta" in c]]

        # Make sure the cleaning step performed as expected
        self.out = GraphData(df=self.out)
        self._LOGGER.info("Checking BAs...")
        self.out.check_all()
        self._LOGGER.info("Execution took %.2f seconds" % (time.time() - start))
