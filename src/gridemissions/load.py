"""
This file contains code to load EBA, eGrid and AMPD datasets. The file provides
one class for each data set. Data can either be raw (of the type that is output
from the parse.py script) or cleaned (outputs from clean.py).

The data handler classes provide methods to access the data in different ways
and perform some checks.
"""

from typing import List, Union, Any
import itertools
from os import PathLike
import pandas as pd
import numpy as np
import logging
from gridemissions import config, eia_api_v2

RTOL = 1e-05
ATOL = 1e-08


class GraphData(object):
    """
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

    def __init__(self, df: pd.DataFrame, api_module: Any = None) -> None:
        self.logger = logging.getLogger("gridemissions." + self.__class__.__name__)
        self.atol = ATOL
        self.rtol = RTOL
        self.df = df
        self.api_module = eia_api_v2 if api_module is None else api_module
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
            data=list(self.df.columns.map(self.api_module.parse_column))
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
        self.KEY = self.api_module.get_key(self.variable)
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


def read_csv(path: Union[str, "PathLike[str]"], api_module: Any = None) -> GraphData:
    return GraphData(
        pd.read_csv(path, index_col=0, parse_dates=True), api_module=api_module
    )


def read_parquet(path: Union[str, "PathLike[str]"]) -> GraphData:
    pass


def _compare_lists(ll, rr):
    left = set(ll)
    right = set(rr)
    return list(left - right) + list(right - left)


def load_bulk(which: str = "elec") -> GraphData:
    """ """
    if which not in ["elec", "co2", "co2i", "raw", "basic", "rolling", "opt"]:
        raise ValueError(f"Unexpected value for which: {which}")
    folder = config["DATA_PATH"] / "EIA_Grid_Monitor" / "processed"
    files = [f for f in folder.iterdir() if f.name.endswith(f"{which}.csv")]
    gd = GraphData(
        pd.concat(
            [pd.read_csv(path, index_col=0, parse_dates=True) for path in files],
            axis=0,
        )
    )
    gd.df.sort_index(inplace=True)
    gd.df = gd.df[~gd.df.index.duplicated(keep="last")]

    return gd
