import io
import requests
import pandas as pd

import gridemissions
from gridemissions import eia_api

DATETIME_FMT = "%Y%m%dT%H%M"


def retrieve(
    variable="co2",
    ba="CISO",
    start="20200101",
    end="20200102",
    field="D",
    ba2=None,
    return_type="dataframe",
):
    """
    Retrieve data from the gridemissions API.

    Parameters
    ----------
    variable : str, one of "co2", "elec", "elec_raw", "co2i"
    ba : str or list of str
        Which balancing area(s) to pull data for
    start : str
    end : str
    field : str {"D", "NG", "TI", "ID"}
    ba2: str
        Second balancing area - if field is "ID"
    return_type: str {"dataframe", "text"}

    Notes
    -----
    `start` and `end` must be parseable by `pandas.to_datetime`. We assume they
    are specified in UTC.
    If field is None or ba is [], then data for all balancing areas and all fields
    is queried.
    If `variable == "co2i"`, the function behavior is slightly different. The API does
    not provide raw carbon intensity data, only consumption and production data, so in this
    case we make two calls to the API and recompute the carbon intensities. Current implementation
    only provides access to either consumption or production based carbon intensities. Column names
    are balancing areas for which data are requested.

    Examples
    --------

    To download co2 consumption data for CISO for the year of 2019:

    >>> from gridemissions import api
    >>> api.retrieve(variable="co2", ba="CISO", start="20190101",
    ...              end="20200101", field="D")

    To download electricity generation data for ERCOT and BPAT for a year:

    >>> api.retrieve(variable="elec", ba=["ERCOT", "BPAT"],
    ...              start="20190101", end="20200101", field="NG")
    """
    if return_type not in ["dataframe", "text"]:
        raise ValueError(f"Incorrect value for return_type: {return_type}")

    if variable == "co2i":
        if field not in ["D", "NG"]:
            raise ValueError(f"Variable is co2i, incorrect value for field: {field}")

        co2 = retrieve(
            variable="co2",
            ba=ba,
            start=start,
            end=end,
            field=field,
            ba2=ba2,
            return_type="dataframe",
        )
        elec = retrieve(
            variable="elec",
            ba=ba,
            start=start,
            end=end,
            field=field,
            ba2=ba2,
            return_type="dataframe",
        )
        co2.columns = co2.columns.map(
            lambda x: eia_api.column_name_to_ba(x, eia_api.KEYS["CO2"][field])
        )
        elec.columns = elec.columns.map(
            lambda x: eia_api.column_name_to_ba(x, eia_api.KEYS["E"][field])
        )
        co2i = co2 / elec

        if return_type == "dataframe":
            return co2i
        else:
            return co2i.to_csv()

    start = pd.to_datetime(start).strftime(DATETIME_FMT)
    end = pd.to_datetime(end).strftime(DATETIME_FMT)

    params = {
        "start": start,
        "end": end,
        "variable": variable,
        "ba": ba,
    }
    if field is not None:
        params["field"] = field

    url = gridemissions.base_url + "/data"

    response = requests.get(url, params=params)
    if response.status_code == requests.codes.ok:
        if return_type == "dataframe":
            return pd.read_csv(
                io.StringIO(response.text), index_col=0, parse_dates=True
            )
        return response.text
    else:
        print("Bad http status code: %d" % response.status_code)
        return ""
