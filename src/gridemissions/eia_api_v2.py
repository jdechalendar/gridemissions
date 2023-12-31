"""
Tools to interact with version 2 of the EIA API at https://www.eia.gov/opendata/
"""
from typing import List
import logging
import requests
from pprint import pformat
import re
from urllib.parse import urljoin
import pandas as pd
import gridemissions as ge


DEFAULT_TIMEOUT_SECONDS = 200.0
BASE_URL = "https://api.eia.gov/v2"
DEFAULT_ROUTE = "electricity/rto/"
ROUTES = ["fuel-type-data", "region-data", "interchange-data"]
MAX_ROWS = 5000  # Max number of rows we can request from the API in one call
INF = 1e20
EIA_DATETIME_FORMAT = "%Y-%m-%dT%H"
FUELS = ["COL", "GAS", "NUC", "OIL", "OTH", "SUN", "UNK", "WAT", "WND", "GEO", "BIO"]
KEYS = {}

BALANCING_AREAS = [
    "AEC",
    "AECI",
    "AESO",
    "AVA",
    "AVRN",
    "AZPS",
    "BANC",
    "BCHA",
    "BPAT",
    "CEN",
    "CHPD",
    "CISO",
    "CPLE",
    "CPLW",
    "DEAA",
    "DOPD",
    "DUK",
    "EEI",
    "EPE",
    "ERCO",
    "FMPP",
    "FPC",
    "FPL",
    "GCPD",
    "GRIF",
    "GRID",
    "GRMA",
    "GVL",
    "GWA",
    "HGMA",
    "HQT",
    "HST",
    "IESO",
    "IID",
    "IPCO",
    "ISNE",
    "JEA",
    "LDWP",
    "LGEE",
    "MISO",
    "MHEB",
    "NBSO",
    "NEVP",
    "NSB",
    "NWMT",
    "NYIS",
    "OVEC",
    "PACE",
    "PACW",
    "PGE",
    "PJM",
    "PNM",
    "PSCO",
    "PSEI",
    "SC",
    "SCEG",
    "SCL",
    "SEC",
    "SEPA",
    "SOCO",
    "SPA",
    "SPC",
    "SRP",
    "SWPP",
    "TAL",
    "TEC",
    "TEPC",
    "TIDC",
    "TPWR",
    "TVA",
    "WACM",
    "WALC",
    "WAUW",
    "WWA",
    "YAD",
]

# Dates at which balancing areas were retired. Source:
# https://www.eia.gov/electricity/gridmonitor/about
RETIREMENTS = {
    "AEC": "2021-09-01",
    "CFE": "2018-07-01",
    "EEI": "2020-02-29",
    "GLHB": "2022-09-01",
    "GRMA": "2018-05-03",
    "NSB": "2020-01-08",
    "OVEC": "2018-12-01",
}


def _is_retired(ba, date):
    if ba not in RETIREMENTS:
        return False
    return pd.to_datetime(RETIREMENTS[ba]) < pd.to_datetime(date)


class EIASession:
    """
    A client session for communicating with the EIA server.

    Parameters
    ----------
    api_key: str, default None
        If not provided, use the key from the config
    base_route: str, default None
        If not provided, use `DEFAULT_API_ROUTE`
    """

    def __init__(self, api_key=None, base_route=None):
        self._LOGGER = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._http: BaseUrlTimeoutSession = BaseUrlTimeoutSession(
            base_url=f"{BASE_URL}/{base_route or DEFAULT_ROUTE}",
            timeout=DEFAULT_TIMEOUT_SECONDS,
            api_key=api_key or ge.config["EIA_API_KEY"],
        )

    def compute_facet_options(self, routes: List[str]) -> dict:
        """
        Compute different options for facets for each provided route

        Parameters
        ----------
        routes: List[str]

        Notes
        -----
        This is used in a test to store the different options available in the API and
        can be used periodically to check if the API has changed.
        See tests/test_eia_api_v2_snapshots.py

        Returns
        -------
        dictionary
            {route => { facet_id => list of options }}
        """
        options = {}
        for route in routes:
            options[route] = {}
            r = self._http.get(route)
            facet_ids = (
                [item["id"] for item in r["response"]["facets"]]
                if "facets" in r["response"]
                else []
            )
            for facet_id in facet_ids:
                r = self._http.get(f"{route}/facet/{facet_id}")
                options[route][facet_id] = r["response"]["facets"]

        return options

    def get_data(
        self,
        route,
        start,
        end=None,
        length=None,
        params=None,
        parse=False,
        drop_duplicates=True,
    ) -> pd.DataFrame:
        """
        Retrieve all data for a route between two dates

        Parameters
        ----------
        route: str
        start: datetime-like
        end: datetime-like, optional
        length: int, default None
        params: dict, default {}
        parse: bool, default False
        drop_duplicates: bool, default True
        """
        length = length or MAX_ROWS
        offset = 0
        rows = []
        total = INF
        params = params or {}

        ba_identifier = "fromba" if route == "interchange-data" else "respondent"
        params.update(
            {
                "start": _ensure_dt_fmt(start),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "sort[1][column]": ba_identifier,
                "sort[1][direction]": "asc",
                "data[]": "value",
                "length": length,
            }
        )

        # Manually list the balancing areas for which we want data
        ba_in_params = (f"facets[{ba_identifier}][0]" in params) or (
            f"facets[{ba_identifier}][]" in params
        )
        if not ba_in_params:
            i = 0
            for ba in BALANCING_AREAS:
                if not _is_retired(ba, start):
                    params.update({f"facets[{ba_identifier}][{i}]": ba})
                i += 1

        # Extra params depending on the route
        if route == "fuel-type-data":
            params.update(
                {
                    "sort[2][column]": "fueltype",
                    "sort[2][direction]": "asc",
                }
            )
        elif route == "region-data":
            # No need to get demand forecast. Only retrieve demand, net generation, interchange
            params.update(
                {
                    "facets[type][0]": "D",
                    "facets[type][1]": "NG",
                    "facets[type][2]": "TI",
                    "sort[2][column]": "type",
                    "sort[2][direction]": "asc",
                }
            )
        elif route == "interchange-data":
            params.update(
                {
                    "sort[2][column]": "toba",
                    "sort[2][direction]": "asc",
                }
            )
        if end is not None:
            params["end"] = _ensure_dt_fmt(end)

        while len(rows) < total:
            params.update({"offset": offset})
            r = self._http.get(f"{route}/data", params=params)
            if "response" not in r:
                self._LOGGER.debug("Dumping params...")
                self._LOGGER.debug(pformat(params))
                raise ValueError(
                    "No response for this query! Increase log level for more info"
                )
            rows += r["response"]["data"]
            if (total != r["response"]["total"]) and (total != INF):
                raise ValueError(
                    "# of rows responsive to the request changed between two subsequent calls!"
                )
            total = r["response"]["total"]
            offset += length

        assert len(rows) == total, f"Got {len(rows)} rows but expected {total}!"
        self._LOGGER.debug(f"{route}: Downloaded {total} rows")

        df = pd.DataFrame.from_records(rows)
        if drop_duplicates:
            nprev = len(df)
            df = df.drop_duplicates()
            if nprev != len(df):
                self._LOGGER.warning(
                    f"{route}: Removed {nprev - len(df)} duplicate rows"
                )

        return _parse(df, route) if parse else df


class BaseUrlTimeoutSession(requests.Session):
    """
    A Session with a base URL, API key, default timeouts and custom logging

    Parameters
    ----------
    base_url: str
    timeout: float
    api_key: str
    """

    def __init__(self, base_url: str, timeout: float, api_key: str) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key
        self._LOGGER = logging.getLogger(__name__ + "." + self.__class__.__name__)
        super(BaseUrlTimeoutSession, self).__init__()

    def request(
        self, method, url: str, *args, timeout=None, params={}, **kwargs
    ) -> dict:
        """
        Send the request after generating the complete URL, including the API key.

        Returns
        -------
        dict returned by Response.json()
        """
        complete_url: str = urljoin(self.base_url, url)
        self._LOGGER.debug(f"{method}: {complete_url}")
        params.update({"api_key": self.api_key})
        response = super(BaseUrlTimeoutSession, self).request(
            method,
            complete_url,
            *args,
            timeout=(timeout or self.timeout),
            params=params,
            **kwargs,
        )
        if response.status_code != 200:
            self._LOGGER.error(
                f"Got status code {response.status_code} when calling {url}"
            )
            self._LOGGER.debug(f"URL for error: {response.url}")
            self._LOGGER.debug(pformat(response.json()))
        return response.json()


def _ensure_dt_fmt(ts, format=None) -> str:
    """Ensure timestamp conforms to format"""
    return pd.to_datetime(ts).strftime(format=(format or EIA_DATETIME_FORMAT))


def _parse(df: pd.DataFrame, route: str) -> pd.DataFrame:
    if df.empty:
        return df
    key = get_key("E")
    if route == "fuel-type-data":
        # Rename "NG" to "GAS" to avoid confusion with net generation column
        df.fueltype = df.fueltype.map(lambda x: "GAS" if x == "NG" else x)
        df["columns"] = df.apply(lambda x: key[x["fueltype"]] % x["respondent"], axis=1)
    if route == "region-data":
        df["columns"] = df.apply(lambda x: key[x["type"]] % x["respondent"], axis=1)
    if route == "interchange-data":
        df["columns"] = df.apply(lambda x: key["ID"] % (x["fromba"], x["toba"]), axis=1)
    df_tmp = df.pivot(index="period", columns="columns", values="value")
    df_tmp.columns.name = None
    return df_tmp


def scrape(start, end=None):
    """
    Scrape data from the EIA API between a start and an end date

    Arguments
    ---------
    start
    end
    """
    session = EIASession()
    return pd.concat(
        [session.get_data(route, start=start, end=end, parse=True) for route in ROUTES],
        axis=1,
    ).sort_index(axis=1)


def get_key(variable):
    if variable in KEYS:
        return KEYS[variable]
    key = {
        "D": f"{variable}_%s_D",
        "NG": f"{variable}_%s_NG",
    }
    if variable.endswith("i"):
        # e.g. carbon intensity is CO2i
        return key

    key["TI"] = f"{variable}_%s_TI"
    key["ID"] = f"{variable}_%s-%s_ID"
    if variable == "E":
        # Electricity also has generation by source
        for fuel in FUELS:
            key[fuel] = f"{variable}_%s_{fuel}"

    KEYS[variable] = key

    return key


def parse_column(column: str) -> dict:
    """
    Extract variable, field, region, optionally region2 from column name
    """
    # Split column names on special character: "_"
    split = re.split(r"-|_", column)
    assert len(split) in [3, 4], f"Unknown column format: {column}"
    data = {"variable": split[0], "region": split[1], "field": split[-1]}

    if data["field"] == "ID":
        data["region2"] = split[2]

    return data


if __name__ == "__main__":
    ge.configure_logging("INFO")
    logging.getLogger("__main__.EIASession").setLevel("DEBUG")
    start = "2023-06-10"
    end = "2023-06-20"
    df = scrape(start, end)
    df.to_csv(f"E_raw_{start}_{end}.csv")
    # session = EIASession()
    # df = session.get_data("interchange-data", start, end, drop_duplicates=False)
    # print(f"{df.duplicated().sum()} duplicate rows...")
    # print(df[df.duplicated()])
