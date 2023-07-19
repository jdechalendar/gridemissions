import json
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


def test_compute_calls(snapshot):
    assert json.dumps(
        eia_api_v2.EIASession().compute_facet_options(eia_api_v2.ROUTES)
    ) == snapshot(name="Output from compute_facet_options")


def test_scrape(snapshot):
    assert eia_api_v2.scrape("2022-07-10", "2022-07-11").to_csv() == snapshot(
        name="Output of scrape for July 10th 2022"
    )


# Before some of the retirements
def test_scrape2(snapshot):
    assert eia_api_v2.scrape("2020-07-10", "2020-07-11").to_csv() == snapshot(
        name="Output of scrape for July 10th 2020"
    )


# Before generation by source data was available
def test_scrape3(snapshot):
    assert eia_api_v2.scrape("2016-07-10", "2016-07-11").to_csv() == snapshot(
        name="Output of scrape for July 10th 2016"
    )
