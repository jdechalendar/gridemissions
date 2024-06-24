import logging
import pandas as pd
import gridemissions as ge
from .eia_samples import get_path

ge.configure_logging("INFO")
logging.getLogger("gridemissions.load").setLevel("DEBUG")
snapshot_float_format = "%.10g"


def test_other_pollutants(snapshot):
    elec = ge.read_csv(get_path("E_opt_2021-07-10_2021-07-20.csv"))
    EFs = pd.read_csv(get_path("TechnologyEFs.csv"), index_col=0).fillna(0.0)
    EFs.loc["UNK"] = EFs.loc["OTH"]
    for poll in EFs.columns:
        print(f"Processing {poll}")
        co2_calc = ge.EmissionsCalc(elec, poll=poll, EF=EFs[poll])
        co2_calc.process()
        assert co2_calc.poll_data.to_csv(
            float_format=snapshot_float_format
        ) == snapshot(name=poll)
