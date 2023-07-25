import pytest
import numpy as np
import pandas as pd
import pandas.testing as tm

import gridemissions as ge


DF1 = pd.DataFrame(data={"CO2_A_D": [np.nan], "CO2_B_D": [1.0]})
DF2 = pd.DataFrame(columns=["CO2_A_D", "CO2_B_D", "CO2_A-B_ID"])
DF2bis = pd.DataFrame(columns=["CO2_A_D", "CO2_B_D", "CO2_A-B_ID", "CO2_A_NG"])
DF2ter = pd.DataFrame(columns=["CO2_A_D", "SO2_B_D"])
DF3 = pd.DataFrame(columns=["CO2_A_D", "CO2_B_D", "CO2_A-B_ID", "CO2_B-A_ID"])
DF4 = pd.DataFrame(
    data={
        "E_A_D": [1.0],
        "E_A_NG": [10.0],
        "E_A_TI": [0.0],
        "E_A_BIO": [1.0],
        "E_B_D": [1.0],
        "E_B_NG": [10.0],
        "E_B_TI": [9.0],
        "E_A-B_ID": [-1.0],
        "E_B-A_ID": [1.0],
        "E_C-D_ID": [0.0],
        "E_D-C_ID": [1.0],
        "E_C_TI": [0.0],
        "E_C_D": [-1.0],
        "E_C_NG": [-1.0],
        "E_C_COL": [-1.0],
    }
)

DF5 = pd.DataFrame(
    data={
        "E_A-B_ID": [-1.0],
        "E_A-C_ID": [0.0],
    }
)


# Reminder that we do not allow pd.DataFrame(), but we do allow dataframes with no data
# Columns need to be supplied for the constructor to work
def test_empty():
    with pytest.raises(AttributeError):
        ge.GraphData(pd.DataFrame())


def test_parse_info(caplog):
    gdata = ge.GraphData(df=DF1)
    assert gdata.variable == "CO2"
    assert gdata.regions == ["A", "B"]
    assert gdata.region_fields == ["D"]
    assert gdata.link_fields == []
    assert gdata.fields == ["D"]
    assert gdata.partners == {"A": [], "B": []}
    assert (
        "Inconsistencies in trade reporting - DEBUG has missing links"
        not in caplog.text
    )

    ge.configure_logging("DEBUG")
    gdata = ge.GraphData(df=DF2)
    assert gdata.regions == ["A", "B"]
    assert gdata.region_fields == ["D"]
    assert gdata.variable == "CO2"
    assert gdata.link_fields == ["ID"]
    assert gdata.fields == ["D", "ID"]
    assert gdata.partners == {"A": ["B"], "B": []}
    assert "Inconsistencies in trade reporting - DEBUG has missing links" in caplog.text
    assert "B-A" in caplog.text


def test_parse_info2(caplog):
    gdata = ge.GraphData(df=DF3)
    assert gdata.regions == ["A", "B"]
    assert gdata.region_fields == ["D"]
    assert gdata.variable == "CO2"
    assert gdata.link_fields == ["ID"]
    assert gdata.fields == ["D", "ID"]
    assert gdata.partners == {"A": ["B"], "B": ["A"]}
    assert (
        "Inconsistencies in trade reporting - DEBUG has missing links"
        not in caplog.text
    )


def test_parse_info3(caplog):
    ge.configure_logging("DEBUG")
    gdata = ge.GraphData(DF4)
    assert gdata.regions == ["A", "B", "C", "D"]
    assert gdata.region_fields == ["D", "NG", "TI", "BIO", "COL"]
    assert gdata.variable == "E"
    assert gdata.link_fields == ["ID"]
    assert gdata.fields == ["D", "NG", "TI", "BIO", "COL", "ID"]
    assert gdata.partners == {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    assert (
        "Inconsistencies in trade reporting - DEBUG has missing links"
        not in caplog.text
    )


def test_parse_info4(caplog):
    ge.GraphData(DF2bis)
    assert "Regions for NG do not match overall regions!" in caplog.text
    assert "Regions for ID do not match overall regions!" in caplog.text


def test_parse_info5():
    with pytest.raises(AssertionError):
        ge.GraphData(DF2ter)


def test_get_cols():
    gdata = ge.GraphData(df=DF1)
    assert gdata.get_cols() == list(gdata.df.columns)
    assert gdata.get_cols(region="A") == ["CO2_A_D"]
    assert gdata.get_cols(field="D") == ["CO2_A_D", "CO2_B_D"]
    assert gdata.get_cols(region=["A"], field=["D"]) == ["CO2_A_D"]

    gdata = ge.GraphData(df=DF2)
    assert gdata.get_cols(region="A") == ["CO2_A_D", "CO2_A-B_ID"]
    assert gdata.get_cols(field="ID") == ["CO2_A-B_ID"]
    assert gdata.get_cols(region="B", field="ID") == []

    gdata = ge.GraphData(df=DF3)
    assert gdata.get_cols(region="A") == ["CO2_A_D", "CO2_A-B_ID"]
    assert gdata.get_cols(field="ID") == ["CO2_A-B_ID", "CO2_B-A_ID"]
    assert gdata.get_cols(region="B", field="ID") == ["CO2_B-A_ID"]


def test_malformed_columns(caplog):
    ge.GraphData(df=DF2)
    assert "Regions for ID do not match overall regions!" in caplog.text


def test_get_data():
    gdata = ge.GraphData(df=DF1)

    # check that get_data squeezes output correctly
    print(DF1.loc[:, "CO2_A_D"])
    print(gdata.get_data(region="A", field="D"))
    tm.assert_series_equal(DF1.loc[:, "CO2_A_D"], gdata.get_data(region="A", field="D"))

    # No arguments should return the entire dataset
    tm.assert_frame_equal(DF1, gdata.get_data())

    # There is only one field in DF1, so this should also return the entire dataset
    tm.assert_frame_equal(DF1, gdata.get_data(field="D"))


def test_has_fields():
    gdata = ge.GraphData(DF1)
    assert gdata.has_field(["D"])
    assert not gdata.has_field(["bla"])

    gdata = ge.GraphData(DF2bis)
    assert gdata.has_field(["NG"])
    assert gdata.has_field(["NG"], region="A")
    assert not gdata.has_field(["NG"], region="B")


def test_check_nans1(caplog):
    gdata = ge.GraphData(DF1)
    assert not gdata.check_nans("A")
    assert gdata.check_nans("B")
    assert "A: 1 NaNs for D" in caplog.text
    assert "B: 1 NaNs for D" not in caplog.text


def test_check_nans2():
    gdata = ge.GraphData(DF4)
    assert gdata.check_nans("A")


def test_check_nans3():
    gdata = ge.GraphData(DF5)
    assert gdata.check_nans("A")


def test_check_balance(caplog):
    gdata = ge.GraphData(DF4)
    assert not gdata.check_balance("A")
    assert gdata.check_balance("B")
    assert "A: 1 TI+D != NG" in caplog.text
    assert "B: 1 TI+D != NG" not in caplog.text


def test_check_interchange(caplog):
    gdata = ge.GraphData(df=DF4)
    assert not gdata.check_interchange("A")
    assert gdata.check_interchange("C")
    assert "A: 1 TI != sum(ID)" in caplog.text
    assert "C: 1 TI != sum(ID)" not in caplog.text


def test_check_antisymmetric(caplog):
    gdata = ge.GraphData(df=DF4)
    assert gdata.check_antisymmetric("A")
    assert not gdata.check_antisymmetric("C")
    assert "A-B: 1 ID[i,j] != -ID[j,i]" not in caplog.text
    assert "C-D: 1 ID[i,j] != -ID[j,i]" in caplog.text


def test_check_positive(caplog):
    gdata = ge.GraphData(df=DF4)
    assert gdata.check_positive("A")
    assert not gdata.check_positive("C")
    assert "A: 1 <0 for D" not in caplog.text
    assert "C: 1 <0 for NG" in caplog.text
    assert "C: 1 <0 for COL" in caplog.text


def test_check_generation_by_source(caplog):
    gdata = ge.GraphData(df=DF4)
    assert not gdata.check_generation_by_source("A")
    assert gdata.check_generation_by_source("C")
    assert "A: 1 NG != sum(Generation by fuel)" in caplog.text
    assert "C: 1 NG != sum(Generation by fuel)" not in caplog.text


def test_check_generation_by_source2(caplog):
    assert ge.GraphData(df=DF1).check_generation_by_source("A")
    assert ge.GraphData(df=DF2).check_generation_by_source("A")
