import pandas as pd
import time
import re
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from gridemissions import eia_api_v2 as eia
from .clean_v2 import Cleaner, EPSILON
from .load import BaData


class PyoCleaningModel(object):
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


class PyoCleaner(Cleaner):
    """
    Optimization-based cleaning class.

    Uses pyomo to build the model and Gurobi as the default solver.
    """

    def __init__(self, ba_data, weights=None, solver="gurobi"):
        super().__init__(ba_data)
        self.m = PyoCleaningModel().m
        self.opt = SolverFactory(solver)
        self.weights = weights
        if weights is not None:
            self.d.df = pd.concat(
                [self.d.df, weights.rename(lambda x: x + "_W", axis=1)], axis=1
            )

    def process(self, debug=False):
        start = time.time()
        self.logger.info("Running PyoCleaner for %d rows" % len(self.d.df))
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
                        self.d.KEY[s]
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
                        self.d.KEY[s] % k: (pyo.value(i.delta_NG_SRC[k, s]))
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
                "srcs": {None: eia.FUELS},
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
            for fuel in eia.FUELS:
                col = self.d.KEY[fuel] % ba
                if weights:
                    col += "_W"
                if col in self.d.df.columns:
                    mydict[(ba, fuel)] = r[col]
        return mydict
