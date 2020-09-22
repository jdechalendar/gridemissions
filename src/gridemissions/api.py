import gridemissions
import requests
import pandas as pd

DATETIME_FMT = "%Y%m%dT%H%M"


def retrieve(
    variable="co2", ba="CISO", start="20200101", end="20200102", field="D", ba2=None
):
    """
    Retrieve data from the gridemissions API.

    Parameters
    ----------
    variable : str, one of "co2", "elec", "elec_raw"
    ba : str or list of str
        Which balancing area(s) to pull data for
    start : str
    end : str
    field : str {"D", "NG", "TI", "ID"}
    ba2: str
        Second balancing area - if field is "ID"

    Notes
    -----
    `start` and `end` must be parseable by `pandas.to_datetime`. We assume they
    are specified in UTC.

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

    start = pd.to_datetime(start).strftime(DATETIME_FMT)
    end = pd.to_datetime(end).strftime(DATETIME_FMT)

    params = {
        "start": start,
        "end": end,
        "variable": variable,
        "ba": ba,
        "field": field,
    }

    url = gridemissions.base_url + "/data"

    response = requests.get(url, params=params)
    if response.status_code == requests.codes.ok:
        return response.text
    else:
        print("Bad http status code: %d" % response.status_code)
