import logging
import gridemissions as ge
from gridemissions import eia_api, emissions

from .eia_samples import get_path

ge.configure_logging("INFO")
logging.getLogger("gridemissions.load").setLevel("DEBUG")


def test_compat():
    # Check we can still load data in the old format (EIA v1 API)
    # I still have some of those files lying around from old papers...
    elec = ge.read_csv(get_path("EBA_elec.csv"), api_module=eia_api)
    EF = emissions.EMISSIONS_FACTORS["CO2"].copy()
    EF["NG"] = EF["GAS"]
    del EF["GAS"]

    co2_calc = ge.EmissionsCalc(elec, api_module=eia_api, EF=EF)
    co2_calc.process()
