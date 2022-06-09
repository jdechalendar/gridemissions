from unittest import mock
import pytest

import gridemissions
from gridemissions import api

FAIL = [
    {"retrieve_kwargs": {"dataset": "C"}, "msg": "Incorrect argument passed: C"},
    # Commenting the tests below as these cases will cause an invalid response, but not cause an error
    #    {
    #        "retrieve_kwargs": {"field": "Demand"},
    #        "msg": "Incorrect argument passed: Demand",
    #    },
    #    # Generation by source is only available for electricity
    #    {
    #        "retrieve_kwargs": {"dataset": "co2", "field": "SRC_COL"},
    #        "msg": "Incorrect argument passed: SRC_COL",
    #    },
    #    # The California ISO is called CISO, not CAISO in the EIA dataset
    #    {"retrieve_kwargs": {"region": "CAISO"}, "msg": "Incorrect argument passed: CAISO"},
    #    # Delimiter should be "-", not ","
    #    {
    #        "retrieve_kwargs": {"region": "CAISO,BPAT"},
    #        "msg": "Incorrect argument passed: CAISO,BPAT",
    #    },
]

SUCCESS = [
    {
        "retrieve_kwargs": {"dataset": "co2"},
        "http_params": {
            "dataset": "co2",
            "past24": "yes",
        },
    },
    {
        "retrieve_kwargs": {"dataset": "co2", "region": "CISO", "field": "D"},
        "http_params": {
            "dataset": "co2",
            "region": "CISO",
            "past24": "yes",
            "field": "D",
        },
    },
    {
        "retrieve_kwargs": {
            "dataset": "co2",
            "region": ["CISO", "MISO"],
            "field": "D",
            "start": "20180701",
            "end": "20180702",
        },
        "http_params": {
            "dataset": "co2",
            "region": ["CISO", "MISO"],
            "start": "20180701T0000Z",
            "end": "20180702T0000Z",
            "field": "D",
        },
    },
]


@pytest.mark.parametrize("test_kwargs", FAIL, ids=range(len(FAIL)))
def test_incorrect_params(test_kwargs):
    with pytest.raises(ValueError, match=test_kwargs["msg"]):
        api.retrieve(**test_kwargs["retrieve_kwargs"])


@mock.patch("requests.get")
@pytest.mark.parametrize("test_kwargs", SUCCESS, ids=range(len(SUCCESS)))
def test_http_request(mock_request, test_kwargs):
    api.retrieve(**test_kwargs["retrieve_kwargs"])

    calls = mock_request.call_args_list
    assert len(calls) == 1
    http_args, http_params = calls[0]
    assert http_args == (gridemissions.config["API_URL"] + "/data",)
    assert http_params == {"params": test_kwargs["http_params"]}


# Manual test below needs to be formalized
# test_kwargs = SUCCESS[2]
# print(api.retrieve(**test_kwargs["retrieve_kwargs"]))
