import logging
import gridemissions as ge
from gridemissions import eia_api

from .eia_samples import get_path

ge.configure_logging("INFO")
logging.getLogger("gridemissions.load").setLevel("DEBUG")


def test_load():
    # Check we can still load data in the old format (EIA v1 API)
    # I still have some of those files lying around from old papers...
    ge.read_csv(get_path("EBA_elec.csv"), api_module=eia_api)
