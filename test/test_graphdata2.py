import pathlib
import gridemissions as ge

ge.configure_logging("DEBUG")


def test_read_csv():
    DATA_PATH = pathlib.Path(__file__).parent.absolute() / "data"
    ge.read_csv(DATA_PATH / "EBA_raw.csv")
    ge.read_csv(DATA_PATH / "EBA_co2.csv")
    ge.read_csv(DATA_PATH / "EBA_elec.csv")

    # Need to finish this test - for now this was more of a manual test
