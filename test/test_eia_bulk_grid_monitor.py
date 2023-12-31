import pathlib
from gridemissions.eia_bulk_grid_monitor import (
    parse_balance_file,
    parse_interchange_file,
)

from .eia_samples import get_path


folder = pathlib.Path("eia_bulk_grid_monitor")


def test_parse_balance_file(snapshot):
    df = parse_balance_file(get_path(folder / "balance_file.csv"))
    assert df.to_csv() == snapshot(name="Balance file")


def test_parse_interchange_file(snapshot):
    df = parse_interchange_file(get_path(folder / "interchange_file.csv"))
    assert df.to_csv() == snapshot(name="Interchange file")
