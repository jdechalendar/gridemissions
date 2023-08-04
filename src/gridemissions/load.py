"""
This file contains code to load EBA, eGrid and AMPD datasets. The file provides
one class for each data set. Data can either be raw (of the type that is output
from the parse.py script) or cleaned (outputs from clean.py).

The data handler classes provide methods to access the data in different ways
and perform some checks.
"""
from typing import List, Union
import itertools
from os.path import join
from os import PathLike
import pandas as pd
import numpy as np
import logging
import json
import re
import warnings
from gridemissions import config, eia_api, eia_api_v2
from gridemissions.eia_api import KEYS, EIA_ALLOWED_SERIES_ID

RTOL = 1e-05
ATOL = 1e-08


class GraphData(object):
    """
    Important note: this class will progressively replace the `BaData` class, which will
    be deprecated.

    Abstraction to represent timeseries data on a graph. This class is a light wrapper
    around a pd.DataFrame, with convenience functions for accessing data. In the
    underlying pd.DataFrame, the index represents time (UTC) and columns represent
    data for different fields on the graph.

    A graph consists of regions (nodes) and trade links (edges). Trade links are
    defined as ordered region pairs.

    The class supports holding data for one variable and multiple fields. Examples
    of variables are electricity, co2, so2, etc. Examples of fields are demand,
    generation, interchange. Field data can be for regions or for links.

    The regions, variable, and fields are inferred from the underlying data columns.

    The `.check_` functions can be used to check certain constraints are met for
    different fields. By convention, if one of the fields is missing, the check
    is True.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.logger = logging.getLogger("gridemissions." + self.__class__.__name__)
        self.atol = ATOL
        self.rtol = RTOL
        self.df = df
        self._parse_info()

    def _parse_info(self) -> None:
        """
        Helper function for the constructor

        Sets the following attributes:
        - variable
        - regions
        - region_fields
        - link_fields
        - fields

        Issues a warning if columns are not consistent, eg if:
        - the regions inferred for the different fields do not match
        - link data is supplied for (r1, r2) but not (r2, r1)
        """
        parsed_columns = pd.DataFrame(
            data=list(self.df.columns.map(eia_api_v2.parse_column))
        )
        if "region2" not in parsed_columns.columns:
            parsed_columns["region2"] = np.nan

        # Store parsed columns as an attribute
        self.parsed_columns = parsed_columns

        variables = parsed_columns.variable.unique()
        assert (
            len(variables) == 1
        ), f"GraphData only support one variable! Got {variables}"
        self.variable = variables[0]
        self.KEY = eia_api_v2.get_key(self.variable)
        self.regions = list(
            parsed_columns[["region", "region2"]].stack().dropna().unique()
        )

        has_no_region2 = parsed_columns["region2"].isna()
        self.region_fields = list(parsed_columns[has_no_region2].field.unique())
        self.link_fields = list(parsed_columns[~has_no_region2].field.unique())
        self.fields = self.region_fields + self.link_fields

        # Get list of trading partners for each region
        self.partners = {}
        for region in self.regions:
            sel = (parsed_columns.region == region) & (parsed_columns.field == "ID")
            self.partners[region] = list(parsed_columns[sel].region2.values)

        # Check consistency of regions for each field
        for f in self.fields:
            if f in eia_api_v2.FUELS:  # Skip this for generation by source columns
                continue
            mismatch = _compare_lists(
                parsed_columns[parsed_columns.field == f].region.unique(), self.regions
            )
            if len(mismatch) > 0:
                self.logger.warning(f"Regions for {f} do not match overall regions!")
                self.logger.debug(mismatch)

        # Check consistency of trading partners
        mismatches = []
        for region in self.regions:
            for region2 in self.partners[region]:
                if region not in self.partners[region2]:
                    mismatches.append(f"{region2}-{region}")
        if len(mismatches) > 0:
            self.logger.warning(
                "Inconsistencies in trade reporting - DEBUG has missing links"
            )
            self.logger.debug(",".join(mismatches))

        # Make sure all columns are accounted for
        mismatch = _compare_lists(self.get_cols(), self.df.columns)
        if len(mismatch) > 0:
            self.logger.debug(self.fields)
            self.logger.debug(mismatch)
            raise ValueError("Unaccounted columns when parsing info!")

    def get_cols(
        self,
        region: Union[str, List[str], None] = None,
        field: Union[str, List[str], None] = None,
        region2: Union[str, List[str], None] = None,
    ) -> List[str]:
        """
        Retrieve column name(s) corresponding to given region(s) and field(s)

        Not passing an argument is equivalent to passing all the available options
        for that argument and a call to `self.get_cols()` should return all of the
        underlying pd.DataFrame's columns.

        The `region2` argument is only used for fields that are defined on links,
        e.g. for fields in `self.link_fields`.

        For links, we use the cartesian product of region and region2.

        Parameters
        ----------
        region: str | List[str] | None
        field: str | List[str] | None
        region2: str | List[str] | None

        Returns
        -------
        columns: List[str]
        """
        # Standardize arguments
        regions = self._parse_args_get_cols(region, self.regions)
        fields = self._parse_args_get_cols(field, self.fields)
        regions2 = self._parse_args_get_cols(region2, self.regions)

        # List all possible columns that match
        candidate_columns = []
        for f in fields:
            if f in self.region_fields:
                candidate_columns += [self.KEY[f] % r for r in regions]
            elif f in self.link_fields:
                candidate_columns += [
                    self.KEY[f] % (r, r2)
                    for r, r2 in itertools.product(regions, regions2)
                ]
            else:
                raise ValueError("Unexpected case!")

        # Filter to get the ones we actually have data for
        return [c for c in candidate_columns if c in self.df.columns]

    def _parse_args_get_cols(
        self, arg: Union[str, List[str], None], default_values: List[str]
    ) -> List[str]:
        """
        Helper function for standardizing arguments to get_cols
        """
        if arg is None:
            return default_values
        if isinstance(arg, str):
            arg = [arg]
        for a in arg:
            if a not in default_values:
                raise ValueError(f"Incorrect argument!\n\t{a} not in {default_values}")
        return arg

    def get_data(
        self, *args, squeeze: bool = True, **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Convenience function to get the data from a call to `get_cols`

        squeeze: bool, default True
            If there is only one column and squeeze is True, ensure we return a pd.Series
            Otherwise return a pd.DataFrame

        Notes
        -----
        If you plan to call `.sum(axis=1)` on the result from this function, the `squeeze`
        argument should probably be `False` to ensure you have a pd.DataFrame
        """
        res = self.df.loc[:, self.get_cols(*args, **kwargs)]
        if squeeze:
            return res.squeeze(axis=1)
        return res

    def to_csv(
        self, path: Union[str, "PathLike[str]", None] = None, **kwargs
    ) -> Union[str, None]:
        return self.df.to_csv(path, **kwargs)

    def to_parquet(self, path: Union[str, "PathLike[str]", None]):
        pass

    def check_all(self) -> bool:
        """"""
        res = True
        for region in self.regions:
            res = (
                res
                & self.check_nans(region)
                & self.check_balance(region)
                & self.check_interchange(region)
                & self.check_antisymmetric(region)
                & self.check_positive(region)
                & self.check_generation_by_source(region)
            )

        if res:
            self.logger.info("All checks passed!")
        return res

    def check_nans(self, region: str) -> None:
        """"""
        res = True
        for field in self.fields:
            if not self.has_field(field, region):
                continue
            ind_na = self.get_data(region=region, field=field).isna()
            cnt_na = np.sum(ind_na.values)
            if cnt_na != 0:
                self.logger.error(f"{region}: {cnt_na} NaNs for {field}")
                res = False
        return res

    def has_field(self, field: Union[str, List[str]], region: str = None) -> bool:
        """"""
        if isinstance(field, str):
            field = [field]
        if region is None:
            for f in field:
                if f not in self.fields:
                    return False
            return True
        else:
            for f in field:
                if (f not in self.fields) or (
                    len(self.get_cols(region=region, field=field)) == 0
                ):
                    return False
            return True

    def check_balance(self, region: str) -> bool:
        """
        TI+D == NG
        """
        res = True
        if not self.has_field(["TI", "D", "NG"], region):
            return res

        left = self.get_data(region=region, field="D") + self.get_data(
            region=region, field="TI"
        )
        right = self.get_data(region=region, field="NG")
        failures = ~np.isclose(left, right, rtol=self.rtol, atol=self.atol)
        cnt = failures.sum()
        if cnt != 0:
            self.logger.error(f"{region}: {cnt} TI+D != NG")
            self.logger.debug(self.get_data(region, field=["TI", "D", "NG"])[failures])
            res = False
        return res

    def check_interchange(self, region: str) -> bool:
        """
        TI == sum(ID)
        """
        out = True
        if not self.has_field(["TI", "ID"], region):
            return out
        left = self.get_data(region=region, field="TI")
        right = self.get_data(region=region, field="ID", squeeze=False).sum(axis=1)
        failures = ~np.isclose(left, right, rtol=self.rtol, atol=self.atol)
        cnt = failures.sum()
        if cnt != 0:
            out = False
            self.logger.error(f"{region}: {cnt} TI != sum(ID)")
            d = self.get_data(region=region, field="TI")[failures]
            self.logger.debug(f"Detail for {region}: TI != sum(ID)\n{d}")
        return out

    def check_antisymmetric(self, region: str) -> bool:
        """
        ID[i,j] == -ID[j,i]
        """
        out = True
        if not self.has_field(["ID"], region):
            return out
        for region2 in self.partners[region]:
            left = self.get_data(region=region, region2=region2, field="ID")
            right = -self.get_data(region=region2, region2=region, field="ID")
            failures = ~np.isclose(left, right, rtol=self.rtol, atol=self.atol)
            cnt = failures.sum()
            if cnt != 0:
                self.logger.error(f"{region}-{region2}: {cnt} ID[i,j] != -ID[j,i]")
                self.logger.debug(pd.concat([left, -right], axis=1))
                out = False
        return out

    def check_positive(self, region: str) -> bool:
        """
        D > 0
        NG > 0
        Generation by fuel > 0
        """
        out = True
        for field in ["D", "NG"] + eia_api_v2.FUELS:
            if not self.has_field([field], region):
                continue
            ind_neg = self.get_data(region=region, field=field) < 0
            cnt_neg = ind_neg.sum()
            if (len(ind_neg) > 0) and (cnt_neg != 0):
                self.logger.error(f"{region}: {cnt_neg} <0 for {field}")
                out = False
        return out

    def check_generation_by_source(self, region: str) -> None:
        """
        NG == sum(Generation by fuel)
        """
        out = True
        src_cols = [
            col for col in eia_api_v2.FUELS if self.has_field(col, region=region)
        ]
        if not (self.has_field(["NG"]) and len(src_cols) > 0):
            return out
        left = self.get_data(region=region, field="NG")
        right = self.get_data(region=region, field=src_cols, squeeze=False).sum(axis=1)
        failures = ~np.isclose(left, right, rtol=self.rtol, atol=self.atol)
        cnt = failures.sum()
        if cnt != 0:
            self.logger.error(f"{region}: {cnt} NG != sum(Generation by fuel)")
            out = False
        return out


def read_csv(path: Union[str, "PathLike[str]"]) -> GraphData:
    return GraphData(pd.read_csv(path, index_col=0, parse_dates=True))


def read_parquet(path: Union[str, "PathLike[str]"]) -> GraphData:
    pass


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
        warnings.warn("The BaData class should be replaced with the GraphData class")
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

        # Infer variable from first data column
        self.variable = eia_api.column_name_to_variable(self.df.columns[0])
        self.regions = self._parse_data_cols()
        self.fileNm = fileNm
        if variable in KEYS:
            self.KEY = KEYS[variable]
        else:
            self.KEY = eia_api.generic_key(variable)

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
    for ll in lines:
        data += [json.loads(ll)]

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


def _compare_lists(ll, rr):
    left = set(ll)
    right = set(rr)
    return list(left - right) + list(right - left)
