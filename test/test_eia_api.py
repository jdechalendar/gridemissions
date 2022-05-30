import pytest

from gridemissions.eia_api import (
    column_name_to_region,
    column_name_to_variable,
    parse_column,
)

key = "EBA.%s-ALL.D.H"
ba = "MISO"
column_name = key % ba
test_data = [(column_name, key, ba)]

key = "EBA.%s-%s.ID.H"
ba1 = "MISO"
ba2 = "PJM"
ba = f"{ba1}-{ba2}"
column_name = key % (ba1, ba2)
test_data.append((column_name, key, ba))


@pytest.mark.parametrize("column_name,key,ba", test_data, ids=range(len(test_data)))
def test_column_name_to_region(column_name, key, ba):
    assert column_name_to_region(column_name, key) == ba


test_data = [("EBA.MISO-ALL.D.H", "E"), ("X_MISO_D", "X")]


@pytest.mark.parametrize("column_name,variable", test_data, ids=range(len(test_data)))
def test_column_name_to_variable(column_name, variable):
    assert column_name_to_variable(column_name) == variable


test_data = [
    ("EBA.MISO-ALL.D.H", {"variable": "E", "region": "MISO", "field": "D"}),
    ("X_MISO_D", {"variable": "X", "region": "MISO", "field": "D"}),
    (
        "EBA.MISO-PJM.ID.H",
        {"variable": "E", "region": "MISO", "field": "ID", "region2": "PJM"},
    ),
    (
        "X_MISO-PJM_ID",
        {"variable": "X", "region": "MISO", "field": "ID", "region2": "PJM"},
    ),
    ("EBA.MISO-ALL.NG.NG.H", {"variable": "E", "region": "MISO", "field": "SRC_NG"}),
]


def test_parse_column_fail():
    with pytest.raises(ValueError):
        parse_column("EBA.C-ALL.SRC.COL.H")


@pytest.mark.parametrize("input,expected", test_data, ids=range(len(test_data)))
def test_parse_column(input, expected):
    assert parse_column(input) == expected
