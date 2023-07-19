import logging
import os
import pytest
import gridemissions as ge
from gridemissions import eia_api_v2

pytestmark = [
    pytest.mark.skipif(
        "EIA_API_KEY" not in ge.config, reason="CI platforms lack API key"
    ),
    pytest.mark.skipif(
        bool(os.environ.get("SKIP_SLOW_TESTS")), reason="Deliberately skip slow tests"
    ),
]


def test_scrape_by_ba(snapshot):
    ge.configure_logging("INFO")
    logging.getLogger("gridemissions.eia_api_v2").setLevel("DEBUG")
    start = "2022-07-10"
    end = "2022-07-11"
    session = eia_api_v2.EIASession()
    for route in eia_api_v2.ROUTES:
        facet_name = "fromba" if route == "interchange-data" else "respondent"
        for ba in eia_api_v2.BALANCING_AREAS:
            params = {}
            # Manually list the balancing areas for which we want data
            if not eia_api_v2._is_retired(ba, start):
                params.update({f"facets[{facet_name}][]": ba})
                assert session.get_data(
                    route, start, end, params=params
                ).to_csv() == snapshot(name=f"{route}-{ba}-{start}-{end}")
