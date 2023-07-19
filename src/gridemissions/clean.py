"""
Tools to clean Balancing area data.
A data cleaning step is performed by an object that subclasses
the `BaDataCleaner` class.
"""
import os
import logging
import time
import re
from gridemissions.load import BaData, GraphData
from gridemissions.eia_api import SRC, KEYS
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


class BaDataCleaner(object):
    """
    Template class for data cleaning.

    This is mostly just a shell to show how cleaning classes should operate.
    """

    def __init__(self, ba_data):
        """
        Parameters
        ----------
        ba_data : BaData object
        """
        self.d = ba_data
        self.logger = logging.getLogger("clean")

    def process(self):
        pass


class BaDataBasicCleaner(BaDataCleaner):
    """
    Basic data cleaning class.

    We run this as the first step of the cleaning process.
    """

    def process(self):
        self.logger.info("Running BaDataBasicCleaner")
        start = time.time()
        data = self.d
        missing_D_cols = [col for col in data.NG_cols if col not in data.D_cols]
        self.logger.info("Adding demand columns for %d bas" % len(missing_D_cols))
        for ba in missing_D_cols:
            data.df.loc[:, data.KEY["D"] % ba] = 1.0
            data.df.loc[:, data.KEY["NG"] % ba] -= 1.0
            data.df.loc[:, data.KEY["TI"] % ba] -= 1.0

        # AVRN only exports to BPAT - this is missing for now
        if "AVRN" not in data.ID_cols:
            self.logger.info("Adding trade columns for AVRN")
            ba = "AVRN"
            ba2 = "BPAT"
            data.df.loc[:, data.KEY["ID"] % (ba, ba2)] = (
                data.df.loc[:, data.KEY["NG"] % ba] - 1.0
            )
            data.df.loc[:, data.KEY["ID"] % (ba2, ba)] = (
                -data.df.loc[:, data.KEY["NG"] % ba] + 1.0
            )

        # Add columns for biomass and geothermal for CISO
        # We are assuming constant generation for each of these sources
        # based on historical data. Before updating this, need to
        # contact the EIA API maintainers to understand why this isn't
        # reported and where to find it
        self.logger.info("Adding GEO and BIO columns for CISO")
        data.df.loc[:, "EBA.CISO-ALL.NG.GEO.H"] = 900.0
        data.df.loc[:, "EBA.CISO-ALL.NG.BIO.H"] = 600.0
        #         data.df.loc[:, "EBA.CISO-ALL.NG.H"] += 600.0 + 900.0

        # Add columns for the BAs that are outside of the US
        foreign_bas = list(
            set([col for col in data.ID_cols2 if col not in data.NG_cols])
        )
        self.logger.info(
            "Adding demand, generation and TI columns for %d foreign bas"
            % len(foreign_bas)
        )
        for ba in foreign_bas:
            trade_cols = [col for col in data.df.columns if "%s.ID.H" % ba in col]
            TI = -data.df.loc[:, trade_cols].sum(axis=1)
            data.df.loc[:, data.KEY["TI"] % ba] = TI
            exports = TI.apply(lambda x: max(x, 0))
            imports = TI.apply(lambda x: min(x, 0))
            data.df.loc[:, data.KEY["D"] % ba] = -imports
            data.df.loc[:, data.KEY["NG"] % ba] = exports
            if ba in ["BCHA", "HQT", "MHEB"]:
                # Assume for these Canadian BAs generation is hydro
                data.df.loc[:, data.KEY["SRC_WAT"] % ba] = exports
            else:
                # And all others are OTH (other)
                data.df.loc[:, data.KEY["SRC_OTH"] % ba] = exports
            for col in trade_cols:
                ba2 = re.split(r"\.|-|_", col)[1]
                data.df.loc[:, data.KEY["ID"] % (ba, ba2)] = -data.df.loc[:, col]

        # Make sure that trade columns exist both ways
        for col in data.get_cols(field="ID"):
            ba = re.split(r"\.|-|_", col)[1]
            ba2 = re.split(r"\.|-|_", col)[2]
            othercol = data.KEY["ID"] % (ba2, ba)
            if othercol not in data.df.columns:
                self.logger.info("Adding %s" % othercol)
                data.df.loc[:, othercol] = -data.df.loc[:, col]

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
            for src in SRC:
                col = data.KEY["SRC_%s" % src] % ba
                if col in data.df.columns:
                    missing = False
                    s = data.df.loc[:, col]
                    if src == "SUN":
                        self.reject_dict[col] = (-1e3, 200e3)
                    data.df.loc[:, col] = s.where(
                        (s >= self.reject_dict[col][0])
                        & (s <= self.reject_dict[col][1])
                    )
                    if src == "SUN":
                        data.df.loc[:, col] = data.df.loc[:, col].apply(
                            lambda x: max(x, 0)
                        )
            if missing:
                data.df.loc[:, data.KEY["SRC_OTH"] % ba] = data.df.loc[
                    :, data.KEY["NG"] % ba
                ]

        # Reinitialize fields
        self.logger.info("Reinitializing fields")
        data = BaData(df=data.df)

        self.r = data

        self.logger.info("Basic cleaning took %.2f seconds" % (time.time() - start))

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
        reject_dict["EBA.AZPS-ALL.D.H"] = (1.0, 30e3)
        reject_dict["EBA.BANC-ALL.D.H"] = (1.0, 6.5e3)
        reject_dict["EBA.BANC-ALL.TI.H"] = (-5e3, 5e3)
        reject_dict["EBA.CISO-ALL.NG.H"] = (5e3, 60e3)
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


class BaDataRollingCleaner(BaDataCleaner):
    """
    Rolling window cleaning.

    This applies the `rolling_window_filter` function to the dataset. In order
    to apply this properly to the beginning of the dataset, we load past data
    that will be used for the cleaning - that is then dropped.
    """

    def process(self, file_name="", folder_hist="", nruns=2):
        """
        Processor function for the cleaner object.

        Parameters
        ----------
        file_name : str
            Base name of the file from which to read historical data.
            Data is read from "%s_basic.csv" % file_name
        folder_hist : str
            Folder from which to read historical data
        nruns : int
            Number of times to apply the rolling window procedure

        Notes
        -----
        If we are not processing a large amount of data at a time, we may not
        have enough data to appropriately estimate the rolling mean and
        standard deviation for the rolling window procedure. If values are
        given for `file_name` and `folder_hist`, data will be read from a
        historical dataset to estimate the rolling mean and standard deviation.
        If there are very large outliers, they can 'mask' smaller outliers.
        Running the rolling window procedure a couple of times helps with this
        issue.
        """
        self.logger.info("Running BaDataRollingCleaner (%d runs)" % nruns)
        start = time.time()
        data = self.d

        # Remember what part we are cleaning
        idx_cleaning = data.df.index

        try:
            # Load the data we already have in memory
            df_hist = pd.read_csv(
                os.path.join(folder_hist, "%s_basic.csv" % file_name),
                index_col=0,
                parse_dates=True,
            )

            # Only take the last 1,000 rows
            # Note that if df_hist has less than 1,000 rows,
            # pandas knows to select df_hist without raising an error.
            df_hist = df_hist.iloc[-1000:]

            # Overwrite with the new data
            old_rows = df_hist.index.difference(data.df.index)
            df_hist = data.df.append(df_hist.loc[old_rows, :], sort=True)
            df_hist.sort_index(inplace=True)

        except FileNotFoundError:
            self.logger.info("No history file")
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
                self.logger.debug("A lot of bad data for %s" % col)
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
            self.logger.warning("There are still some NaNs. Unexpected")

        # Just keep the indices we are working on currently
        data = BaData(df=df_hist.loc[idx_cleaning, :])

        self.r = data
        self.weights = mean_.loc[idx_cleaning, :].applymap(
            lambda x: A / max(GAMMA, abs(x))
        )

        self.logger.info(
            "Rolling window cleaning took %.2f seconds" % (time.time() - start)
        )


class BaDataPyoCleaningModel(object):
    """
    Create an AbstractModel() for the cleaning problem.

    No data is passed into this model at this point, it is
    simply written in algebraic form.
    """

    def __init__(self):
        m = pyo.AbstractModel()

        # Sets
        m.regions = pyo.Set()
        m.srcs = pyo.Set()
        m.regions2 = pyo.Set(within=m.regions * m.regions)
        m.regions_srcs = pyo.Set(within=m.regions * m.srcs)

        # Parameters
        m.D = pyo.Param(m.regions, within=pyo.Reals)
        m.NG = pyo.Param(m.regions, within=pyo.Reals)
        m.TI = pyo.Param(m.regions, within=pyo.Reals)
        m.ID = pyo.Param(m.regions2, within=pyo.Reals)
        m.NG_SRC = pyo.Param(m.regions_srcs, within=pyo.Reals)
        m.D_W = pyo.Param(m.regions, default=1.0, within=pyo.Reals)
        m.NG_W = pyo.Param(m.regions, default=1.0, within=pyo.Reals)
        m.TI_W = pyo.Param(m.regions, default=1.0, within=pyo.Reals)
        m.ID_W = pyo.Param(m.regions2, default=1.0, within=pyo.Reals)
        m.NG_SRC_W = pyo.Param(m.regions_srcs, default=1.0, within=pyo.Reals)

        # Variables
        # delta_NG_aux are aux variable for the case where there
        # are no SRC data. In that case, the NG_sum constraint would
        # only have: m.NG + m.delta_NG = 0.
        m.delta_D = pyo.Var(m.regions, within=pyo.Reals)
        m.delta_NG = pyo.Var(m.regions, within=pyo.Reals)
        m.delta_TI = pyo.Var(m.regions, within=pyo.Reals)
        m.delta_ID = pyo.Var(m.regions2, within=pyo.Reals)
        m.delta_NG_SRC = pyo.Var(m.regions_srcs, within=pyo.Reals)
        #         m.delta_NG_aux = pyo.Var(m.regions, within=pyo.Reals)

        # Constraints
        m.D_positive = pyo.Constraint(m.regions, rule=self.D_positive)
        m.NG_positive = pyo.Constraint(m.regions, rule=self.NG_positive)
        m.NG_SRC_positive = pyo.Constraint(m.regions_srcs, rule=self.NG_SRC_positive)
        m.energy_balance = pyo.Constraint(m.regions, rule=self.energy_balance)
        m.antisymmetry = pyo.Constraint(m.regions2, rule=self.antisymmetry)
        m.trade_sum = pyo.Constraint(m.regions, rule=self.trade_sum)
        m.NG_sum = pyo.Constraint(m.regions, rule=self.NG_sum)

        # Objective
        m.total_penalty = pyo.Objective(rule=self.total_penalty, sense=pyo.minimize)

        self.m = m

    def D_positive(self, model, i):
        return (model.D[i] + model.delta_D[i]) >= EPSILON

    def NG_positive(self, model, i):
        return (model.NG[i] + model.delta_NG[i]) >= EPSILON

    def NG_SRC_positive(self, model, k, s):
        return model.NG_SRC[k, s] + model.delta_NG_SRC[k, s] >= EPSILON

    def energy_balance(self, model, i):
        return (
            model.D[i]
            + model.delta_D[i]
            + model.TI[i]
            + model.delta_TI[i]
            - model.NG[i]
            - model.delta_NG[i]
        ) == 0.0

    def antisymmetry(self, model, i, j):
        return (
            model.ID[i, j]
            + model.delta_ID[i, j]
            + model.ID[j, i]
            + model.delta_ID[j, i]
            == 0.0
        )

    def trade_sum(self, model, i):
        return (
            model.TI[i]
            + model.delta_TI[i]
            - sum(
                model.ID[k, l] + model.delta_ID[k, l]
                for (k, l) in model.regions2
                if k == i
            )
        ) == 0.0

    def NG_sum(self, model, i):
        return (
            model.NG[i]
            + model.delta_NG[i]  # + model.delta_NG_aux[i]
            - sum(
                model.NG_SRC[k, s] + model.delta_NG_SRC[k, s]
                for (k, s) in model.regions_srcs
                if k == i
            )
        ) == 0.0

    def total_penalty(self, model):
        return (
            sum(
                (
                    model.D_W[i] * model.delta_D[i] ** 2
                    + model.NG_W[i] * model.delta_NG[i] ** 2
                    #                      + model.delta_NG_aux[i]**2
                    + model.TI_W[i] * model.delta_TI[i] ** 2
                )
                for i in model.regions
            )
            + sum(
                model.ID_W[i, j] * model.delta_ID[i, j] ** 2
                for (i, j) in model.regions2
            )
            + sum(
                model.NG_SRC_W[i, s] * model.delta_NG_SRC[i, s] ** 2
                for (i, s) in model.regions_srcs
            )
        )


class BaDataPyoCleaner(BaDataCleaner):
    """
    Optimization-based cleaning class.

    Uses pyomo to build the model and Gurobi as the default solver.
    """

    def __init__(self, ba_data, weights=None, solver="gurobi"):
        super().__init__(ba_data)

        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory

        self.m = BaDataPyoCleaningModel().m
        self.opt = SolverFactory(solver)
        self.weights = weights
        if weights is not None:
            self.d.df = pd.concat(
                [self.d.df, weights.rename(lambda x: x + "_W", axis=1)], axis=1
            )

    def process(self, debug=False):
        start = time.time()
        self.logger.info("Running BaDataPyoCleaner for %d rows" % len(self.d.df))
        self.d.df = self.d.df.fillna(0)
        if not debug:
            self.r = self.d.df.apply(self._process, axis=1)
        else:
            r_list = []
            delta_list = []
            for idx, row in self.d.df.iterrows():
                _, r, deltas = self._process(row, debug=True)
                r_list.append(r)
                delta_list.append(deltas)
            self.r = pd.concat(r_list, axis=1).transpose()
            self.deltas = pd.concat(delta_list, axis=1).transpose()
            self.deltas.index = self.d.df.index

        self.r.index = self.d.df.index

        # Make sure the cleaning step performed as expected
        self.r = BaData(df=self.r)
        self.logger.info("Checking BAs...")
        for ba in self.r.regions:
            self.r.checkBA(ba)
        self.logger.info("Execution took %.2f seconds" % (time.time() - start))

    def _process(self, row, debug=False):
        if row.isna().sum() > 0:
            raise ValueError("Cannot call this method on data with NaNs")
        i = self._create_instance(row)
        self.opt.solve(i)

        r = pd.concat(
            [
                pd.Series(
                    {
                        self.d.KEY["NG"] % k: (i.NG[k] + pyo.value(i.delta_NG[k]))
                        for k in i.regions
                    }
                ),
                pd.Series(
                    {
                        self.d.KEY["D"] % k: (i.D[k] + pyo.value(i.delta_D[k]))
                        for k in i.regions
                    }
                ),
                pd.Series(
                    {
                        self.d.KEY["TI"] % k: (i.TI[k] + pyo.value(i.delta_TI[k]))
                        for k in i.regions
                    }
                ),
                pd.Series(
                    {
                        self.d.KEY["ID"]
                        % (k1, k2): (i.ID[k1, k2] + pyo.value(i.delta_ID[k1, k2]))
                        for (k1, k2) in i.regions2
                    }
                ),
                pd.Series(
                    {
                        self.d.KEY["SRC_%s" % s]
                        % k: (i.NG_SRC[k, s] + pyo.value(i.delta_NG_SRC[k, s]))
                        for (k, s) in i.regions_srcs
                    }
                ),
            ]
        )

        deltas = pd.concat(
            [
                pd.Series(
                    {
                        self.d.KEY["NG"] % k: (pyo.value(i.delta_NG[k]))
                        for k in i.regions
                    }
                ),
                pd.Series(
                    {self.d.KEY["D"] % k: (pyo.value(i.delta_D[k])) for k in i.regions}
                ),
                pd.Series(
                    {
                        self.d.KEY["TI"] % k: (pyo.value(i.delta_TI[k]))
                        for k in i.regions
                    }
                ),
                pd.Series(
                    {
                        self.d.KEY["ID"] % (k1, k2): (pyo.value(i.delta_ID[k1, k2]))
                        for (k1, k2) in i.regions2
                    }
                ),
                pd.Series(
                    {
                        self.d.KEY["SRC_%s" % s] % k: (pyo.value(i.delta_NG_SRC[k, s]))
                        for (k, s) in i.regions_srcs
                    }
                ),
            ]
        )

        if not debug:
            return r

        return i, r, deltas

    def _create_instance(self, row):
        def append_W(x):
            return [c + "_W" for c in x]

        NG_SRC_data = self._get_ng_src(row)
        NG_SRC_data_W = self._get_ng_src(row, weights=True)
        opt_data = {
            None: {
                "regions": {None: self.d.regions},
                "srcs": {None: SRC},
                "regions2": {
                    None: list(
                        set(
                            [
                                (re.split(r"\.|-|_", el)[1], re.split(r"\.|-|_", el)[2])
                                for el in self.d.df.columns
                                if "ID" in re.split(r"\.|-|_", el)
                            ]
                        )
                    )
                },
                "regions_srcs": {None: list(NG_SRC_data.keys())},
                "D": self._reduce_cols(row.loc[self.d.get_cols(field="D")].to_dict()),
                "NG": self._reduce_cols(row.loc[self.d.get_cols(field="NG")].to_dict()),
                "TI": self._reduce_cols(row.loc[self.d.get_cols(field="TI")].to_dict()),
                "ID": self._reduce_cols(
                    row.loc[self.d.get_cols(field="ID")].to_dict(), nfields=2
                ),
                "NG_SRC": NG_SRC_data,
            }
        }

        if self.weights is not None:
            opt_data[None]["D_W"] = self._reduce_cols(
                row.loc[append_W(self.d.get_cols(field="D"))].to_dict()
            )
            opt_data[None]["NG_W"] = self._reduce_cols(
                row.loc[append_W(self.d.get_cols(field="NG"))].to_dict()
            )
            opt_data[None]["TI_W"] = self._reduce_cols(
                row.loc[append_W(self.d.get_cols(field="TI"))].to_dict()
            )
            opt_data[None]["ID_W"] = self._reduce_cols(
                row.loc[append_W(self.d.get_cols(field="ID"))].to_dict(), nfields=2
            )
            opt_data[None]["NG_SRC_W"] = NG_SRC_data_W

        instance = self.m.create_instance(opt_data)
        return instance

    def _reduce_cols(self, mydict, nfields=1):
        """
        Helper function to simplify the names in a dictionary
        """
        newdict = {}
        for k in mydict:
            if nfields == 1:
                newk = re.split(r"\.|-|_", k)[1]
            elif nfields == 2:
                newk = (re.split(r"\.|-|_", k)[1], re.split(r"\.|-|_", k)[2])
            else:
                raise ValueError("Unexpected argument")
            newdict[newk] = mydict[k]
        return newdict

    def _get_ng_src(self, r, weights=False):
        """
        Helper function to get the NG_SRC data.
        """
        mydict = {}
        for ba in self.d.regions:
            for src in SRC:
                col = self.d.KEY["SRC_%s" % src] % ba
                if weights:
                    col += "_W"
                if col in self.d.df.columns:
                    mydict[(ba, src)] = r[col]
        return mydict


class BaDataCvxCleaner(BaDataCleaner):
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
        start = time.time()
        self.logger.info("Running BaDataCvxCleaner for %d rows" % len(self.d.df))
        self.d.df = self.d.df.fillna(0)

        results = []

        def cvx_solve(row, regions, debug=False):
            if row.isna().sum() > 0:
                raise ValueError("Cannot call this method on data with NaNs")

            n_regions = len(regions)

            D = row[[KEYS["E"]["D"] % r for r in regions]].values
            D_W = [
                el**0.5
                for el in row[[KEYS["E"]["D"] % r + "_W" for r in regions]].values
            ]
            NG = row[[KEYS["E"]["NG"] % r for r in regions]].values
            NG_W = [
                el**0.5
                for el in row[[KEYS["E"]["NG"] % r + "_W" for r in regions]].values
            ]
            TI = row[[KEYS["E"]["TI"] % r for r in regions]].values
            TI_W = [
                el**0.5
                for el in row[[KEYS["E"]["TI"] % r + "_W" for r in regions]].values
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
                    if KEYS["E"]["ID"] % (ri, rj) in row.index:
                        ID[(ri, rj)] = row[KEYS["E"]["ID"] % (ri, rj)]
                        ID_W[(ri, rj)] = row[KEYS["E"]["ID"] % (ri, rj) + "_W"]
            delta_ID = {k: cp.Variable(name=f"{k}") for k in ID}
            constraints = [
                D + delta_D >= 1.0,
                NG + delta_NG >= 1.0,
                D + delta_D + TI + delta_TI - NG - delta_NG == 0.0,
            ]

            if with_ng_src:
                NG_SRC = {}
                NG_SRC_W = {}

                for i, src in enumerate(SRC):
                    for j, r in enumerate(regions):
                        if KEYS["E"][f"SRC_{src}"] % r in row.index:
                            NG_SRC[(src, r)] = row[KEYS["E"][f"SRC_{src}"] % r]
                            NG_SRC_W[(src, r)] = row[KEYS["E"][f"SRC_{src}"] % r + "_W"]
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
                                for src in SRC
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
                            index=[KEYS["E"]["NG"] % r for r in regions],
                        ),
                        pd.Series(
                            D + delta_D.value,
                            index=[KEYS["E"]["D"] % r for r in regions],
                        ),
                        pd.Series(
                            TI + delta_TI.value,
                            index=[KEYS["E"]["TI"] % r for r in regions],
                        ),
                        pd.Series(
                            {KEYS["E"]["ID"] % k: ID[k] + delta_ID[k].value for k in ID}
                        ),
                        pd.Series(
                            {
                                KEYS["E"][f"SRC_{s}"] % r: NG_SRC[(s, r)]
                                + delta_NG_SRC[(s, r)].value
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
                            index=[KEYS["E"]["NG"] % r for r in regions],
                        ),
                        pd.Series(
                            D + delta_D.value,
                            index=[KEYS["E"]["D"] % r for r in regions],
                        ),
                        pd.Series(
                            TI + delta_TI.value,
                            index=[KEYS["E"]["TI"] % r for r in regions],
                        ),
                        pd.Series(
                            {KEYS["E"]["ID"] % k: ID[k] + delta_ID[k].value for k in ID}
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
                            delta_NG.value, index=[KEYS["E"]["NG"] % r for r in regions]
                        ),
                        pd.Series(
                            delta_D.value, index=[KEYS["E"]["D"] % r for r in regions]
                        ),
                        pd.Series(
                            delta_TI.value, index=[KEYS["E"]["TI"] % r for r in regions]
                        ),
                        pd.Series({KEYS["E"]["ID"] % k: delta_ID[k].value for k in ID}),
                        pd.Series(
                            {
                                KEYS["E"][f"SRC_{s}"] % r: delta_NG_SRC[(s, r)].value
                                for (s, r) in NG_SRC
                            }
                        ),
                    ]
                )
            else:
                deltas = pd.concat(
                    [
                        pd.Series(
                            delta_NG.value, index=[KEYS["E"]["NG"] % r for r in regions]
                        ),
                        pd.Series(
                            delta_D.value, index=[KEYS["E"]["D"] % r for r in regions]
                        ),
                        pd.Series(
                            delta_TI.value, index=[KEYS["E"]["TI"] % r for r in regions]
                        ),
                        pd.Series({KEYS["E"]["ID"] % k: delta_ID[k].value for k in ID}),
                    ]
                )
            return pd.concat([r, deltas.rename(lambda x: x + "_Delta")])

        cvx_solve = dask.delayed(cvx_solve)
        for idx, row in self.d.df.iterrows():
            results.append(cvx_solve(row, self.d.regions, debug=debug))
        results = dask.compute(*results, scheduler="processes")
        df = pd.DataFrame(results, index=self.d.df.index)

        self.r = df.loc[
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
        self.r = BaData(df=self.r)
        self.logger.info("Checking BAs...")
        for ba in self.r.regions:
            self.r.checkBA(ba)
        self.logger.info("Execution took %.2f seconds" % (time.time() - start))
