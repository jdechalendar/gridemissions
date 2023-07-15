import pandas as pd
import io
from unittest import mock
import pytest

from gridemissions import eia_api_v2 as eia
from .eia_samples import RETURN_VALUE_1, RETURN_VALUE_2, get_path

OPTIONS = {
    "respondent": [{"id": f"BA{i}"} for i in range(15)],
    "fueltype": [{"id": f"fuel{i}"} for i in range(8)],
}


def test_dt_format():
    assert eia._ensure_dt_fmt("2023-01-01") == "2023-01-01T00"
    assert eia._ensure_dt_fmt("2023-01-01T22H00") == "2023-01-01T22"
    assert eia._ensure_dt_fmt("2023-01-01T22H10") == "2023-01-01T22"


@mock.patch("requests.sessions.Session.request")
def test_get_data(mock_request):
    """
    Usage for Mock `side_effect`:
    https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect
    If you pass in an iterable, it is used to retrieve an iterator which must yield a
    value on every call. This value can either be an exception instance to be raised,
    or a value to be returned from the call to the mock (DEFAULT handling is identical
    to the function case).
    """
    session = eia.EIASession(api_key="DUMMY_KEY")
    mock_response = mock.Mock()
    mock_response.json.side_effect = [RETURN_VALUE_1, RETURN_VALUE_2]
    mock_response.status_code = 200
    mock_request.return_value = mock_response

    data = session.get_data(
        "fuel-type-data",
        start="2023-01-01",
        end="2023-01-01T23",
        length=16,
        params={"facets[respondent][]": "AECI", "facets[fueltype][]": "COL"},
    )

    # This call to EIASession.get_data should make 2 calls to request
    assert len(mock_request.call_args_list) == 2

    # We expect 24 data points in total
    assert len(data) == 24


DATA_PARSE = {
    "fuel-type-data": (
        """
period,respondent,respondent-name,fueltype,type-name,value,value-units\n
2023-07-17T00,PNM,Public Service Company of New Mexico,WND,Wind,40,megawatthours\n
2023-07-17T00,MIDW,Midwest,NUC,Nuclear,11675,megawatthours\n""",
        """
period,E_MIDW_NUC,E_PNM_WND\n
2023-07-17T00,11675,40\n""",
    ),
    "region-data": (
        """
period,respondent,respondent-name,type,type-name,value,value-units\n
2023-07-17T00,SOCO,"Southern Company Services, Inc. - Trans",D,Demand,36006.0,megawatthours\n
2023-07-17T00,CISO,California Independent System Operator,TI,Total interchange,2108.0,megawatthours\n""",
        """
period,E_CISO_TI,E_SOCO_D\n
2023-07-17T00,2108.0,36006.0\n""",
    ),
    "interchange-data": (
        """
period,fromba,fromba-name,toba,toba-name,value,value-units\n
2023-07-17T00,AECI,"Associated Electric Cooperative, Inc.",MISO,"Midcontinent Independent System Operator, Inc.",-430,megawatthours\n
2023-07-17T00,YAD,"Alcoa Power Generating, Inc. - Yadkin Division",DUK,Duke Energy Carolinas,80,megawatthours\n""",
        """
period,E_AECI-MISO_ID,E_YAD-DUK_ID\n
2023-07-17T00,-430,80\n""",
    ),
}


def test_parse():
    for route, data in DATA_PARSE.items():
        print(route)
        print(eia._parse(pd.read_csv(io.StringIO(data[0])), route))
        pd.testing.assert_frame_equal(
            eia._parse(pd.read_csv(io.StringIO(data[0])), route),
            pd.read_csv(io.StringIO(data[1]), index_col=0),
        )


test_data = [
    ("X_MISO_D", {"variable": "X", "region": "MISO", "field": "D"}),
    (
        "E_MISO-PJM_ID",
        {"variable": "E", "region": "MISO", "field": "ID", "region2": "PJM"},
    ),
    (
        "X_MISO-PJM_ID",
        {"variable": "X", "region": "MISO", "field": "ID", "region2": "PJM"},
    ),
    ("E_MISO_GAS", {"variable": "E", "region": "MISO", "field": "GAS"}),
]


@pytest.mark.parametrize("input,expected", test_data, ids=range(len(test_data)))
def test_parse_column(input, expected):
    assert eia.parse_column(input) == expected


def test_regenerate_column():
    # Regenerate column using the parsed data and the key and check we get the same thing
    columns = pd.read_csv(get_path("E_raw_2021-07-10_2021-07-20.csv")).columns
    columns = [c for c in columns if c != "period"]
    for column in columns:
        data = eia.parse_column(column)
        key = eia.get_key(data["variable"])

        if "region2" in data:
            regenerated_column = key[data["field"]] % (data["region"], data["region2"])
        else:
            regenerated_column = key[data["field"]] % data["region"]

        if column != regenerated_column:
            raise ValueError(
                f"Error parsing {column}... Regenerated: {regenerated_column} with {data}"
            )


def test_retirements():
    # "AEC": "2021-09-01",
    # "CFE": "2018-07-01",
    assert not eia._is_retired("AECI", "1992-01-01")
    assert eia._is_retired("AEC", "2022-01-01")
    assert not eia._is_retired("AEC", "2021-09-01")
    assert eia._is_retired("CFE", "2018-07-02")
    assert not eia._is_retired("CFE", "2018-06-30")
