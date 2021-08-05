"""
This file contains code to load EBA, eGrid and AMPD datasets. The file provides
one class for each data set. Data can either be raw (of the type that is output
from the parse.py script) or cleaned (outputs from clean.py).

The data handler classes provide methods to access the data in different ways
and perform some checks.
"""
from os.path import join
import pandas as pd
import logging
import json
import re
from gridemissions import config
from gridemissions.eia_api import KEYS, BAs, EIA_ALLOWED_SERIES_ID


class BaData(object):
    """Class to handle BA-level data. The EBA class provides generation,
    consumption, the trade matrix and total interchange either at the BA or at
    the regional (IEA-defined) level. User guide:
        https://www.eia.gov/realtime_grid/docs/userguide-knownissues.pdf

    The class is a light wrapped around the pd.DataFrame object which is the
    format in which the underlying data are stored.

    The main purpose of the class is to provide convenience functions that make
    handling the data easier.

    Timestamps are in UTC.

    EBA data columns
    ----------------
    D: Demand
    NG: Net Generation
    TI: Total Interchange - (positive if exports)
    ID: Interchange with directly connected balancing authorities - (positive
        if exports)

    Consistency requirements
    ------------------------
    - Interchange data is antisymmetric: ID[i,j] == -ID[j,i]
    - Total trade with interchange data: TI == sum(ID[i,:])
    - Balance equation for total trade, demand, generation: TI + D == NG

    Methods
    -------
    get_cols(self, r) : generate column names for regions r for a given field.

    Attributes
    ----------
    regions : are in alphabetical order
    df : raw dataframe
    """
    def __init__(self, fileNm=None, df=None, variable="E", dataset="EBA", step=None):
        """
        Initialize the BaData object

        There are two preferred ways to initialize this object: by passing a `pd.DataFrame`,
        or by passing a file name from which to read the data using `pd.read_csv`.


        Parameters
        ----------
        fileNm: str, default None
            fileNm from which to read the data
        df: pd.DataFrame, default None
            data 
        dataset: str, default "EBA"
            base name for file in which data are stored. Parameter will be deprecated soon and
            should not be used.
        step: int, default None
            processing step at which to load the data. Parameter will be deprecated soon and
            should not be used.

        """
        self.logger = logging.getLogger("load")

        if df is not None:
            self.df = df
        else:
            if step is not None:
                fileNm = join(
                    config["DATA_PATH"], "analysis", "%s_%d.csv" % (dataset, step)
                )
            if fileNm is None:
                fileNm = join(config["DATA_PATH"], "analysis", "EBA_0.csv")
            self.df = pd.read_csv(fileNm, index_col=0, parse_dates=True)

        self.variable = variable
        self.regions = self._parse_data_cols()
        self.fileNm = fileNm
        self.KEY = KEYS[variable]

    def get_cols(self, r=None, field="D"):
        """
        Retrieve column name(s) corresponding to region(s) and a field

        Parameters
        ----------
        r: str or list of str, default None
            regions. If None, data is returned for all regions
        field: str
            field for which to load columns. Used to index in self.KEY

        Returns
        -------
        cols: list of str
        """
        if field not in self.KEY:
            raise ValueError(f"{field} not in str(list(self.KEY.keys()))")
        if field != "ID":
            if r is None:
                r = self.regions
            if isinstance(r, str):
                r = [r]
            return [self.KEY[field] % ir for ir in r]
        else:
            if r is None:
                r = self.regions
            if isinstance(r, str):
                r = [r]
            return [
                self.KEY[field] % (ir, ir2)
                for ir in r
                for ir2 in self.regions
                if self.KEY[field] % (ir, ir2) in self.df.columns
            ]

    def get_trade_partners(self, r):
        """
        Return list of regions that trade with a given region

        Parameter
        ---------
        r: str
            region for which to search
        """
        partners = []
        for r2 in self.regions:
            if (self.KEY["ID"] % (r, r2) in self.df.columns) and (
                self.KEY["ID"] % (r2, r) in self.df.columns
            ):
                partners += [r2]
        return partners

    def _parse_data_cols(self):
        """
        Checks:
        - Consistent number of regions for demand / generation / total
            interchange / trade matrix
        Returns the list of regions
        """
        regions = set([re.split(r"\.|-|_", el)[1] for el in self.df.columns])
        D_cols = [
            re.split(r"\.|-|_", el)[1]
            for el in self.df.columns
            if "D" in re.split(r"\.|-|_", el)
        ]
        # The check in [3, 5] was added to filter out the generation columns
        # by source in electricity
        NG_cols = [
            re.split(r"\.|-|_", el)[1]
            for el in self.df.columns
            if (
                ("NG" in re.split(r"\.|-|_", el))
                and (len(re.split(r"\.|-|_", el)) in [3, 5])
            )
        ]
        TI_cols = [
            re.split(r"\.|-|_", el)[1]
            for el in self.df.columns
            if "TI" in re.split(r"\.|-|_", el)
        ]
        ID_cols = [
            re.split(r"\.|-|_", el)[1]
            for el in self.df.columns
            if "ID" in re.split(r"\.|-|_", el)
        ]
        ID_cols2 = [
            re.split(r"\.|-|_", el)[2]
            for el in self.df.columns
            if "ID" in re.split(r"\.|-|_", el)
        ]

        self.D_cols = D_cols
        self.NG_cols = NG_cols
        self.TI_cols = TI_cols
        self.ID_cols = ID_cols
        self.ID_cols2 = ID_cols2

        if len(NG_cols) != len(D_cols):
            self.logger.warn("Inconsistent columns: len(NG_cols) != len(D_cols)")
        if set(NG_cols) != regions:
            self.logger.warn("Inconsistent columns: set(NG_cols) != regions")

        if not ("i" in self.variable):
            if len(NG_cols) != len(TI_cols):
                self.logger.warn("Inconsistent columns: len(NG_cols) != len(TI_cols)")
            if set(NG_cols) != set(ID_cols):
                self.logger.warn("Inconsistent columns: set(NG_cols) != set(ID_cols)")
            if set(NG_cols) != set(ID_cols2):
                self.logger.warn("Inconsistent columns: set(NG_cols) != set(ID_cols2)")

        return sorted(list(regions))

    def get_trade_out(self, r=None):
        if r is None:
            r = self.regions
        if isinstance(r, str):
            r = [r]
        cols = []
        for ir2 in self.regions:
            cols += [self.KEY["ID"] % (ir, ir2) for ir in r]
        return [c for c in cols if c in self.df.columns]

    def checkBA(self, ba, tol=1e-2, log_level=logging.INFO):
        """
        Sanity check function
        TODO: add check for different generation sources, if data is present
        """
        logger = self.logger
        log_level_old = logger.level
        logger.setLevel(log_level)
        logger.debug("Checking %s" % ba)
        partners = self.get_trade_partners(ba)

        # NaNs
        for field in ["D", "NG", "TI"]:
            ind_na = self.df.loc[:, self.get_cols(r=ba, field=field)[0]].isna()
            cnt_na = ind_na.sum()
            if cnt_na != 0:
                logger.error(
                    "There are still %d nans for %s field %s" % (cnt_na, ba, field)
                )

        for ba2 in partners:
            cnt_na = self.df.loc[:, self.KEY["ID"] % (ba, ba2)].isna().sum()
            if cnt_na != 0:
                logger.error("There are still %d nans for %s-%s" % (cnt_na, ba, ba2))

        # TI+D == NG
        res1 = self.df.loc[:, self.get_cols(r=ba, field="NG")[0]] - (
            self.df.loc[:, self.get_cols(r=ba, field="D")[0]]
            + self.df.loc[:, self.get_cols(r=ba, field="TI")[0]]
        )
        if (res1.abs() > tol).sum() != 0:
            logger.error("%s: TI+D == NG violated" % ba)

        # TI == ID.sum()
        res2 = self.df.loc[:, self.get_cols(r=ba, field="TI")[0]] - self.df.loc[
            :, [self.KEY["ID"] % (ba, ba2) for ba2 in partners]
        ].sum(axis=1)
        if (res2.abs() > tol).sum() != 0:
            logger.error("%s: TI == ID.sum()violated" % ba)

        # ID[i,j] == -ID[j,i]
        for ba2 in partners:
            res3 = (
                self.df.loc[:, self.KEY["ID"] % (ba, ba2)]
                + self.df.loc[:, self.KEY["ID"] % (ba2, ba)]
            )
            if (res3.abs() > tol).sum() != 0:
                logger.error("%s-%s: ID[i,j] == -ID[j,i] violated" % (ba, ba2))

        # D and NG negative
        for field in ["D", "NG"]:
            ind_neg = self.df.loc[:, self.get_cols(r=ba, field=field)[0]] < 0
            cnt_neg = ind_neg.sum()
            if cnt_neg != 0:
                logger.error(
                    "%s: there are %d <0 values for field %s" % (ba, cnt_neg, field)
                )
        logger.setLevel(log_level_old)


def convert_raw_eba(file_name, file_name_out=None):
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)
    logger.debug("Loading raw JSON")
    with open(f"{file_name}.txt") as fr:
        lines = fr.readlines()

    # convert json - each line is a dictionary
    data = []
    for l in lines:
        data += [json.loads(l)]

    # separate id data from ts data
    ts_data = [d for d in data if len(d.keys()) == 10]
    # id_list = [d for d in data if len(d.keys()) == 5]


    def choose(el, ba_list):
        series_id = el["series_id"]
        if ".HL" in series_id:
            return False
        if ".ID.H" in series_id:
            return (re.split(r"\.|-", series_id)[1] in ba_list) and (
                re.split(r"\.|-", series_id)[2] in ba_list
            )
        else:
            return re.split(r"\.|-", series_id)[1] in ba_list

    logger.debug("Converting to dataframe")
    df = pd.concat(
        [
            pd.DataFrame(el["data"], columns=["datetime", el["series_id"]]).set_index(
                "datetime", drop=True
            )
            for el in ts_data
            if el["series_id"] in EIA_ALLOWED_SERIES_ID
        ],
        axis=1,
    )

    logger.debug("Converting to datetime and sorting")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    raw = BaData(df=df)

    if file_name_out is not None:
        raw.df.to_csv(file_name_out)
    else:
        return raw
