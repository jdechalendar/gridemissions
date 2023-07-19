import gridemissions as ge
from .eia_samples import get_path

ge.configure_logging("DEBUG")


def test_read_csv(snapshot):
    # This test also serves to check metadata for the dataset at different dates
    # For example, two balancing areas were retired between the first set of dates
    # and the second one

    # The file below was downloaded with:
    # start = "2021-07-10"
    # end = "2021-07-20"
    # df = scrape(start, end)
    # df.to_csv(f"E_raw_{start}_{end}.csv")

    for i, filename in enumerate(
        [
            "E_raw_2021-07-10_2021-07-20.csv",
            "E_raw_2023-06-10_2023-06-20.csv",
        ]
    ):
        data = ge.read_csv(get_path(filename))
        assert len(data.df) == 241
        assert len(data.regions) == snapshot(name=f"Number of regions [{i}]")
        assert data.regions == snapshot(name=f"Regions [{i}]")
        assert data.region_fields == snapshot(name=f"Region fields [{i}]")
        assert data.link_fields == snapshot(name=f"Link fields [{i}]")
        assert data.fields == snapshot(name=f"Fields [{i}]")

        data = ge.read_csv(get_path("E_raw_2023-06-10_2023-06-20.csv"))
