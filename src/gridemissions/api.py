import io
import logging
from typing import List, Union
import requests
import pandas as pd

import gridemissions
from gridemissions import eia_api

DATETIME_FMT = "%Y%m%dT%H%MZ"
DATASET_TO_VARIABLE = {"raw": "E", "co2": "CO2", "elec": "E", "co2i": "CO2i"}

logger = logging.getLogger(__name__)


def retrieve(dataset, start=None, end=None, return_type="dataframe", **kwargs):
    """
    Retrieve data from the gridemissions API. Additional kwargs will be passed to
    GraphData.get_cols on the server side of the API.

    Parameters
    ----------
    dataset : str in DATASET_TO_VARIABLE.keys(). Currently, ["co2", "raw", "elec", "co2i"]
    start : str | datetime-like | None
        must be parseable by pd.to_datetime. Assumes UTC
    end : str | datetime-like | None
        must be parseable by pd.to_datetime. Assumes UTC
    return_type: str {"dataframe", "text"}

    Other Parameters
    ----------------
    region: str | List[str] | None
    field: str | List[str] | None
    region2: str | List[str] | None


    Notes
    -----
    For documentation of the "Other Parameters", refer to `GraphData.get_cols`.

    If `dataset == "co2i"`, the function behavior is slightly different. The API does
    not provide raw carbon intensity data, only consumption and production data, so in this
    case we make two calls to the API and recompute the carbon intensities. Current
    implementation only provides access to either consumption or production based carbon
    intensities. Column names are regions for which data are requested.

    If start or end are None, retrieves the last 24 hours of available data

    Examples
    --------

    To download co2 consumption data for CISO for the year of 2019:

    >>> from gridemissions import api
    >>> api.retrieve(dataset="CO2", region="CISO", start="20190101",
    ...              end="20200101", field="D")

    To download electricity generation data for ERCOT and BPAT for a year:

    >>> api.retrieve(dataset="E", region=["ERCOT", "BPAT"],
    ...              start="20190101", end="20200101", field="NG")
    """
    _check_arg_in_list(dataset, DATASET_TO_VARIABLE.keys())
    _check_arg_in_list(return_type, ["dataframe", "text"])

    if dataset == "co2i":
        if ("field" not in kwargs) or (kwargs["field"] not in ["D", "NG"]):
            raise ValueError("Dataset is co2i, field should be D or NG")

        co2 = retrieve(
            dataset="co2", start=start, end=end, return_type="dataframe", **kwargs
        )
        elec = retrieve(
            dataset="elec", start=start, end=end, return_type="dataframe", **kwargs
        )
        key_E = eia_api.get_key("E")
        key_CO2 = eia_api.get_key("CO2")
        key_CO2i = eia_api.get_key("CO2i")
        field = kwargs["field"]
        co2.columns = co2.columns.map(
            lambda x: eia_api.column_name_to_region(x, key_CO2[field])
        )
        elec.columns = elec.columns.map(
            lambda x: eia_api.column_name_to_region(x, key_E[field])
        )
        co2i = co2 / elec
        co2i.columns = co2i.columns.map(lambda x: key_CO2i[field] % x)

        if return_type == "dataframe":
            return co2i
        else:
            return co2i.to_csv()

    params = {"dataset": dataset}
    if (start is None) or (end is None):
        params["past24"] = "yes"
    else:
        params["start"] = pd.to_datetime(start).strftime(DATETIME_FMT)
        params["end"] = pd.to_datetime(end).strftime(DATETIME_FMT)
    for name in ["region", "field", "region2"]:
        if (name in kwargs) and (kwargs[name] is not None):
            params[name] = kwargs[name]

    url = gridemissions.config["API_URL"] + "/data"

    try:
        response = requests.get(url, params=params)
    except requests.exceptions.ConnectionError:
        logger.error(f"ConnectionError when connecting to {url}")
        if return_type == "dataframe":
            return pd.DataFrame()
        return ""

    if response.status_code == requests.codes.ok:
        if return_type == "dataframe":
            return pd.read_csv(
                io.StringIO(response.text), index_col=0, parse_dates=True
            )
        return response.text
    else:
        logger.error(f"{response.status_code} error: DEBUG for more info")
        logger.debug(f"{response.text}")
        if return_type == "dataframe":
            return pd.DataFrame()
        return ""


def _check_arg_in_list(
    arg: Union[str, List[str]], allowed_list: List[str], split: Union[str, None] = None
):
    """
    Helper function to sanitize inputs to the retrieve function
    """
    if type(arg) == list:
        for subarg in arg:
            _check_arg_in_list(subarg, allowed_list, split=split)
        return
    if split is not None:
        for subarg in arg.split(split):
            _check_arg_in_list(subarg, allowed_list)
    if arg not in allowed_list:
        raise ValueError(f"Incorrect argument passed: {arg}")
