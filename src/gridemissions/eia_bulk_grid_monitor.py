"""
Parsers for the six-month files available from the "Download" menu at
https://www.eia.gov/electricity/gridmonitor/dashboard/

Data are converted to the format that is output by `eia_api_v2.scrape`

Note: there do not seem to be data on Geothermal or Biomass
"""
import pandas as pd
from gridemissions.eia_api_v2 import get_key, EIA_DATETIME_FORMAT

EIA_BULK_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def parse_balance_file(filename):
    df = pd.read_csv(filename, thousands=",")
    col_renamer = {
        "Balancing Authority": "BA",
        "Demand (MW)": "D",
        "Net Generation (MW)": "NG",
        "Net Generation (MW) from All Petroleum Products": "OIL",
        "Net Generation (MW) from Coal": "COL",
        "Net Generation (MW) from Hydropower and Pumped Storage": "WAT",
        "Net Generation (MW) from Natural Gas": "GAS",
        "Net Generation (MW) from Nuclear": "NUC",
        "Net Generation (MW) from Other Fuel Sources": "OTH",
        "Net Generation (MW) from Solar": "SUN",
        "Net Generation (MW) from Unknown Fuel Sources": "UNK",
        "Net Generation (MW) from Wind": "WND",
        "Total Interchange (MW)": "TI",
        "UTC Time at End of Hour": "period",
    }
    df = df[col_renamer.keys()].rename(col_renamer, axis=1)
    df = df.melt(id_vars=["period", "BA"])
    key = get_key("E")
    df["column"] = df.apply(lambda x: key[x.variable] % x.BA, axis=1)
    df.period = pd.to_datetime(df.period, format=EIA_BULK_DATETIME_FORMAT).dt.strftime(
        EIA_DATETIME_FORMAT
    )
    df = df.pivot(index="period", columns="column", values="value")
    return df


def parse_interchange_file(filename):
    df = pd.read_csv(filename, thousands=",")
    col_renamer = {
        "Interchange (MW)": "value",
        "UTC Time at End of Hour": "period",
        "Balancing Authority": "BA",
        "Directly Interconnected Balancing Authority": "BA2",
    }
    df = df[col_renamer.keys()].rename(col_renamer, axis=1)
    key = get_key("E")["ID"]
    df["column"] = df.apply(lambda x: key % (x.BA, x.BA2), axis=1)
    df.period = pd.to_datetime(df.period, format=EIA_BULK_DATETIME_FORMAT).dt.strftime(
        EIA_DATETIME_FORMAT
    )
    df = df.pivot(index="period", columns="column", values="value")
    return df
