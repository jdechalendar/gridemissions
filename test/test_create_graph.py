import json

import gridemissions as ge
from gridemissions.viz.d3map import create_graph

# from gridemissions.viz.d3map import create_graph2

from .test_workflows import FILENAMES
from .eia_samples import get_path


def test_create_graph(snapshot):
    for filename in FILENAMES:
        poll = ge.read_csv(get_path(filename[2:] % "CO2"))
        elec = ge.read_csv(get_path(filename % "opt"))

        name = filename[2:-4] % "d3map"
        folder_out = get_path(name)
        folder_out.mkdir(exist_ok=True)

        for ts in poll.df.index[0:2]:
            g = create_graph(poll, elec, ts, folder_out=folder_out, save_data=False)
            assert json.dumps(g) == snapshot(name=f"{name} {ts}")

            # assert g == create_graph2(poll, elec, ts, folder_out=folder_out)
