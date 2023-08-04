import logging
import json
import os
import pandas as pd
import pytest

import gridemissions as ge
from gridemissions.clean import BasicCleaner, RollingCleaner, CvxCleaner
from gridemissions.emissions import EmissionsCalc
from .eia_samples import get_path
from gridemissions.viz.d3map import create_graph

FILENAMES = ["E_%s_2021-07-10_2021-07-20.csv", "E_%s_2023-06-10_2023-06-20.csv"]
ge.configure_logging("INFO")
logging.getLogger("gridemissions.load").setLevel("DEBUG")
snapshot_float_format = "%.10g"

# If API key is not present, assume we are in the CI environment
IS_CI = "EIA_API_KEY" not in ge.config


def test_basic_cleaner(snapshot):
    for filename in FILENAMES:
        data = ge.read_csv(get_path(filename % "raw"))
        print(data.KEY)
        cleaner = BasicCleaner(data)
        cleaner.process()
        assert cleaner.out.to_csv(float_format=snapshot_float_format) == snapshot(
            name=f"Basic cleaner output: {filename}"
        )

        if not IS_CI:
            # Store result for the next test
            cleaner.out.to_csv(get_path(filename % "basic"))


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_SLOW_TESTS")), reason="Deliberately skip slow tests"
)
def test_rolling_window_cleaner(snapshot):
    for filename in FILENAMES:
        data = ge.read_csv(get_path(filename % "basic"))
        cleaner = RollingCleaner(data)

        # TODO: Test the version with a folder_hist and a file_name
        cleaner.process()
        assert cleaner.out.to_csv(float_format=snapshot_float_format) == snapshot(
            name=f"Rolling window cleaner output: {filename}"
        )
        assert cleaner.weights.to_csv(float_format=snapshot_float_format) == snapshot(
            name=f"Rolling window cleaner weights: {filename}"
        )

        if not IS_CI:
            # Store result for the next test
            cleaner.out.to_csv(get_path(filename % "rolling"))
            cleaner.weights.to_csv(get_path(filename % "weights"))

    # Test version with history file
    for filename in FILENAMES:
        data = ge.read_csv(get_path(filename % "basic"))
        cleaner = RollingCleaner(data)

        # TODO: Test the version with a folder_hist and a file_name
        cleaner.process(file_name_hist=get_path(filename % "basic"))
        assert cleaner.out.to_csv(float_format=snapshot_float_format) == snapshot(
            name=f"Rolling window cleaner output (w hist): {filename}"
        )
        assert cleaner.weights.to_csv(float_format=snapshot_float_format) == snapshot(
            name=f"Rolling window cleaner weights (w hist): {filename}"
        )


@pytest.mark.skipif(
    bool(os.environ.get("SKIP_SLOW_TESTS")), reason="Deliberately skip slow tests"
)
def test_opt_cleaner(snapshot):
    for filename in FILENAMES:
        n = 20  # Only use the first 20 rows to make this test faster
        data = ge.read_csv(get_path(filename % "rolling"))
        weights = pd.read_csv(
            get_path(filename % "weights"), index_col=0, parse_dates=True
        )
        data.df = data.df.iloc[0:n]
        weights = weights.iloc[0:n]
        cleaner = CvxCleaner(data, weights=weights)
        cleaner.process(debug=True)
        assert cleaner.out.check_all()

        # Note: tolerance on comparison is lower here than for the other tests
        # Optimization results are not always exactly the same
        assert cleaner.out.to_csv(float_format="%.2g") == snapshot(
            name=f"Cvx cleaner: {filename}"
        )
        if not IS_CI:
            # Store result for the next test
            cleaner.out.to_csv(get_path(filename % "opt"))


def test_emissions_calc(snapshot):
    for filename in FILENAMES:
        co2_calc = EmissionsCalc(ge.read_csv(get_path(filename % "opt")))
        co2_calc.process()
        assert co2_calc.poll_data.check_all()
        assert co2_calc.poll_data.to_csv(float_format="%.3g") == snapshot(
            name=f"CO2 data: {filename}"
        )
        assert co2_calc.polli_data.to_csv(float_format="%.3g") == snapshot(
            name=f"CO2i data: {filename}"
        )

        if not IS_CI:
            # Store result for the next test
            co2_calc.poll_data.to_csv(get_path(filename[2:] % "CO2"))
            co2_calc.polli_data.to_csv(get_path(filename[2:] % "CO2i"))


def test_create_graph(snapshot):
    for filename in FILENAMES:
        poll = ge.read_csv(get_path(filename[2:] % "CO2"))
        elec = ge.read_csv(get_path(filename % "opt"))

        name = filename[2:-4] % "d3map"
        folder_out = get_path(name)
        folder_out.mkdir(exist_ok=True)

        for ts in poll.df.index[0:2]:
            g = create_graph(poll, elec, ts, folder_out=folder_out, save_data=True)
            assert json.dumps(g) == snapshot(name=f"{name} {ts}")
