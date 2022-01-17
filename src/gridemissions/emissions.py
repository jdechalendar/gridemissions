import sys
import logging
import time
import numpy as np
from gridemissions.eia_api import SRC, KEYS
from gridemissions.load import BaData


# Default emissions factors - can also supply custom EFs to BaDataEmissionsCalc
# CO2
# UNK is 2017 average US power grid intensity according to Schivley 2018
# unit is kg / MWh
EMISSIONS_FACTORS = {
    "CO2": {
        "WAT": 4,
        "NUC": 16,
        "SUN": 46,
        "NG": 469,
        "WND": 12,
        "COL": 1000,
        "OIL": 840,
        "OTH": 439,
        "UNK": 439,
        "BIO": 230,
        "GEO": 42,
    }
}


def consumption_emissions(F, P, ID):
    """
    Form and solve linear system to compute consumption emissions

    Parameters
    ----------
    F: np.array
        emissions
    P: np.array
        production
    ID: np.array
        exchanges

    Notes
    -----
    Create linear system to calculate consumption emissions
    - Create import matrix
    - Create linear system and solve:
    f_i^c*(d_i+sum_j t_{ji}) - sum_j t_{ij}*f_j^c = F_i^p
    where:
        f_i^c: consumption emissions at node i
        d_i: demand at node i
        t: trade matrix - t_{ij} is from node i to j
        F_i^p: emissions produced at node i
    Note: np version must be high enough, otherwise np.linalg.cond fails
    on a matrix with only zeros.
    """
    from distutils.version import LooseVersion

    assert LooseVersion(np.__version__) >= LooseVersion("1.15.1")

    # Create and solve linear system
    Imp = (-ID).clip(min=0)  # trade matrix reports exports - we want imports
    I_tot = Imp.sum(axis=1)  # sum over columns
    A = np.diag(P + I_tot) - Imp
    b = F

    perturbed = []
    if np.linalg.cond(A) > (1.0 / sys.float_info.epsilon):
        # matrix is ill-conditioned
        for i in range(len(A)):
            if (np.abs(A[:, i]).sum() == 0.0) & (np.abs(A[i, :]).sum() == 0.0):
                A[i, i] = 1.0  # slightly perturb that element
                perturbed += [i]
                # force this to be zero so the linear system makes sense
                b[i] = 0.0

    X = np.linalg.solve(A, b)

    for j in perturbed:
        if X[j] != 0.0:
            print(b[j])
            print(np.abs(A[j, :]).sum())
            print(np.abs(A[:, j]).sum())
            raise ValueError("X[%d] is %.2f instead of 0" % (j, X[j]))

    return X, len(perturbed)


class BaDataEmissionsCalc(object):
    def __init__(self, ba_data, poll="CO2", EF=None):
        self.logger = logging.getLogger("clean")
        self.ba_data = ba_data
        self.df = ba_data.df.copy(deep=True)
        self.regions = ba_data.regions
        self.poll = poll
        self.KEY_E = KEYS["E"]
        self.KEY_poll = KEYS[poll]
        if EF is None:
            self.EF = EMISSIONS_FACTORS[poll]
        else:
            self.EF = EF

    def process(self):
        """
        Compute emissions production, consumption and flows.

        Compute (i) production emissions, and (ii) consumption-based emissions
        factors
        Then recreate a BaData object for emissions and check physical
        balances.
        """
        self.logger.info("Running BaDataEmissionsCalc for %d rows" % len(self.df))
        cnt_na = self.df.isna().any().sum()
        if cnt_na > 0:
            self.logger.warning(f"Setting {cnt_na} NaNs to zero")
            self.logger.debug(f"Dumping cols with NaNs: {self.df.columns[self.df.isna().any()]}")
        self._add_production_emissions()
        self._add_consumption_efs()

        # Create columns for demand
        for ba in self.regions:
            self.df.loc[:, "%s_%s_D" % (self.poll, ba)] = (
                self.df.loc[:, "%si_%s_D" % (self.poll, ba)]
                * self.df.loc[:, self.ba_data.get_cols(r=ba, field="D")[0]]
            )

        # Create columns for pairwise trade
        for ba in self.regions:
            for ba2 in self.ba_data.get_trade_partners(ba):
                imp = self.df.loc[:, self.KEY_E["ID"] % (ba, ba2)].apply(
                    lambda x: min(x, 0)
                )
                exp = self.df.loc[:, self.KEY_E["ID"] % (ba, ba2)].apply(
                    lambda x: max(x, 0)
                )
                self.df.loc[:, self.KEY_poll["ID"] % (ba, ba2)] = (
                    imp * self.df.loc[:, "%si_%s_D" % (self.poll, ba2)]
                    + exp * self.df.loc[:, "%si_%s_D" % (self.poll, ba)]
                )

        # Create columns for total trade
        for ba in self.regions:
            self.df.loc[:, self.KEY_poll["TI"] % ba] = self.df.loc[
                :,
                [
                    self.KEY_poll["ID"] % (ba, ba2)
                    for ba2 in self.ba_data.get_trade_partners(ba)
                ],
            ].sum(axis=1)

        # Create BaData object for pollutant
        self.poll_data = BaData(
            df=self.df.loc[
                :, [col for col in self.df.columns if "%s_" % self.poll in col]
            ],
            variable=self.poll,
        )

        # Check balances
        self.logger.warn("Consumption calcs - unimplemented balance check!")

    def _add_production_emissions(self):
        """

        Unit for emissions is kg.
        Assumes elec data comes in MWh and EF data in kg/MWh.
        """
        for ba in self.regions:
            gen_cols = [(src, self.KEY_E["SRC_%s" % src] % ba) for src in SRC]
            gen_cols = [(src, col) for src, col in gen_cols if col in self.df.columns]
            self.df.loc[:, self.KEY_poll["NG"] % ba] = self.df.apply(
                lambda x: sum(self.EF[src] * x[col] for src, col in gen_cols), axis=1
            )

    def _add_consumption_efs(self):
        """

        Unit for emissions is kg.
        Assumes elec data comes in MWh and EF data in kg/MWh.
        """
        self.logger.info("Calculating consumption emissions...")
        start_time = time.time()
        newcols = self.df.columns.tolist() + [
            "%si_%s_D" % (self.poll, ba) for ba in self.regions
        ]

        self.df = self.df.reindex(columns=newcols)
        self.df = self.df.apply(self._apply_consumption_calc, axis=1)
        end_time = time.time()
        self.logger.info("Elapsed time was %g seconds" % (end_time - start_time))

    def _apply_consumption_calc(self, row):
        """
        Extract data for a row in the dataframe with EBA and AMPD data and call
        the consumption emissions function
        """
        P = row[[KEYS["E"]["NG"] % r for r in self.regions]].values
        ID = np.zeros((len(self.regions), len(self.regions)))
        for i, ri in enumerate(self.regions):
            for j, rj in enumerate(self.regions):
                if KEYS["E"]["ID"] % (ri, rj) in row.index:
                    ID[i][j] = row[KEYS["E"]["ID"] % (ri, rj)]

        F = row[[("%s_%s_NG") % (self.poll, ba) for ba in self.regions]].values
        X = [np.nan for ba in self.regions]
        try:
            X, pert = consumption_emissions(F, P, ID)
        except np.linalg.LinAlgError:
            pass
        except ValueError:
            raise
        row[[("%si_%s_D") % (self.poll, ba) for ba in self.regions]] = X
        return row
