import sys
import logging
import time
import numpy as np
import pandas as pd
from gridemissions import eia_api_v2 as eia_api
from gridemissions.eia_api_v2 import FUELS, KEYS
from gridemissions.load import GraphData
from packaging.version import Version

assert Version(np.__version__) >= Version(
    "1.15.1"
), "consumption_emissions breaks for numpy versions < 1.15.1"

# Default emissions factors - can also supply custom EFs to EmissionsCalc
# CO2
# UNK is 2017 average US power grid intensity according to Schivley 2018
# unit is kg / MWh
EMISSIONS_FACTORS = {
    "CO2": {
        "WAT": 4,
        "NUC": 16,
        "SUN": 46,
        "GAS": 469,
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


class EmissionsCalc(object):
    def __init__(self, ba_data: GraphData, poll: str = "CO2", EF=None):
        self.logger = logging.getLogger("gridemissions." + self.__class__.__name__)
        self.ba_data = ba_data
        self.df = ba_data.df.copy(deep=True)
        self.regions = ba_data.regions
        self.poll = poll
        self.KEY_E = eia_api.get_key("E")
        self.KEY_poll = eia_api.get_key(poll)
        self.KEY_polli = eia_api.get_key(poll + "i")
        self.EF = EF if EF else EMISSIONS_FACTORS[poll]

    def process(self):
        """
        Compute emissions production, consumption and flows.

        Compute (i) production emissions, and (ii) consumption-based emissions
        factors. Then recreate a GraphData object for emissions and check physical
        balances.
        """
        self.logger.info(f"Running EmissionsCalc for {len(self.df)} rows")
        cnt_na = np.sum(self.df.isna().values)
        if cnt_na > 0:
            self.logger.warning(f"Setting {cnt_na} NaNs to zero")
            self.logger.debug(
                f"Dumping cols with NaNs: {self.df.columns[self.df.isna().any()]}"
            )
        self._add_production_emissions()
        self._add_consumption_efs()

        # Create columns for demand
        for ba in self.regions:
            self.df.loc[:, self.KEY_poll["D"] % ba] = self.df.loc[
                :, self.KEY_polli["D"] % ba
            ] * self.ba_data.get_data(region=ba, field="D")

        # Create columns for pairwise trade
        new_cols = {}
        for ba in self.regions:
            for ba2 in self.ba_data.partners[ba]:
                imp = self.df.loc[:, self.KEY_E["ID"] % (ba, ba2)].apply(
                    lambda x: min(x, 0)
                )
                exp = self.df.loc[:, self.KEY_E["ID"] % (ba, ba2)].apply(
                    lambda x: max(x, 0)
                )
                new_cols[self.KEY_poll["ID"] % (ba, ba2)] = (
                    imp * self.df.loc[:, self.KEY_polli["D"] % ba2]
                    + exp * self.df.loc[:, self.KEY_polli["D"] % ba]
                )
        self.df = pd.concat(
            [self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1
        )

        new_cols = {}
        # Create columns for total trade
        for ba in self.regions:
            new_cols[self.KEY_poll["TI"] % ba] = self.df.loc[
                :,
                [self.KEY_poll["ID"] % (ba, ba2) for ba2 in self.ba_data.partners[ba]],
            ].sum(axis=1)
        self.df = pd.concat(
            [self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1
        )

        # Create GraphData object for pollutant
        self.poll_data = GraphData(
            df=self.df.loc[
                :, [col for col in self.df.columns if "%s_" % self.poll in col]
            ]
        )

        # Create GraphData object for pollutant intensity
        self.polli_data = GraphData(
            df=self.df.loc[
                :, [col for col in self.df.columns if "%si_" % self.poll in col]
            ]
        )

        # Check balances
        self.logger.warning("Consumption calcs - unimplemented balance check!")
        self.poll_data.check_all()

    def _add_production_emissions(self):
        """

        Unit for emissions is kg.
        Assumes elec data comes in MWh and EF data in kg/MWh.
        """
        for ba in self.regions:
            gen_cols = [(fuel, self.KEY_E[fuel] % ba) for fuel in FUELS]
            gen_cols = [(fuel, col) for fuel, col in gen_cols if col in self.df.columns]
            self.df.loc[:, self.KEY_poll["NG"] % ba] = self.df.apply(
                lambda x: sum(self.EF[fuel] * x[col] for fuel, col in gen_cols), axis=1
            )

    def _add_consumption_efs(self):
        """

        Unit for emissions is kg.
        Assumes elec data comes in MWh and EF data in kg/MWh.
        """
        self.logger.info("Calculating consumption emissions...")
        start_time = time.time()
        newcols = self.df.columns.tolist() + [
            self.KEY_polli["D"] % ba for ba in self.regions
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

        F = row[[self.KEY_poll["NG"] % ba for ba in self.regions]].values
        X = [np.nan for ba in self.regions]
        try:
            X, pert = consumption_emissions(F, P, ID)
        except np.linalg.LinAlgError:
            pass
        except ValueError:
            raise
        row[[self.KEY_polli["D"] % ba for ba in self.regions]] = X
        return row
