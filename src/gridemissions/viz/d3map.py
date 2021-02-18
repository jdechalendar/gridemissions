import os
import json
import numpy as np
from seed import config


data_path = os.path.join(config["DATA_PATH"], "analysis", "d3map")

WECC_BAs = [
    "AVA",
    "AZPS",
    "BANC",
    "BPAT",
    "CHPD",
    "CISO",
    "DEAA",
    "DOPD",
    "EPE",
    "GCPD",
    "GRMA",
    "GWA",
    "HGMA",
    "IID",
    "IPCO",
    "LDWP",
    "NEVP",
    "NWMT",
    "PACE",
    "PACW",
    "PGE",
    "PNM",
    "PSCO",
    "PSEI",
    "SCL",
    "SRP",
    "TEPC",
    "TIDC",
    "TPWR",
    "WACM",
    "WALC",
    "WAUW",
    "WWA",
]

legCircleTitles = {
    "E": "Electricity consumption",
    "CO2": "Carbon consumption",
    "SO2": "SO2 consumption",
    "NOX": "NOX consumption",
}

legLineTitles = {
    "E": "Electricity trade",
    "CO2": "Carbon trade",
    "SO2": "SO2 trade",
    "NOX": "NOx trade",
}

legColorTitles = {
    "CO2": "Consumption-based carbon intensity (kg/MWh)",
    "SO2": "Consumption-based SO2 intensity (ton/MWh)",
    "NOX": "Consumption-based NOX intensity (ton/MWh)",
}

titles = {
    "E": "ELECTRICITY",
    "CO2": "CARBON",
    "SO2": "SULFUR DIOXIDE",
    "NOX": "NITROGEN OXIDES",
}


def resetCoords():
    xyCoordsPath = os.path.join(data_path, "xycoords.json")
    with open(xyCoordsPath, "r") as fr:
        xycoords = json.load(fr)
    xyCoordsLabPath = os.path.join(data_path, "xycoords_lab.json")
    with open(xyCoordsLabPath, "r") as fr:
        xycoords_lab = json.load(fr)

    baseGraphPath = os.path.join(data_path, "graph.json")
    with open(baseGraphPath, "r") as fr:
        graph = json.load(fr)
    newnodes = []
    labels = []
    for el in graph["nodes"]:
        el["coords"] = xycoords[el["shortNm"]]
        newnodes.append(el)
        if el["shortNm"] in xycoords_lab:
            labels.append(
                {"shortNm": el["shortNm"], "coords": xycoords_lab[el["shortNm"]]}
            )
    graph["nodes"] = newnodes
    graph["labels"] = labels

    graphPath_out = os.path.join(data_path, "graph2.json")
    with open(graphPath_out, "w") as fw:
        json.dump(graph, fw)


def add_data_nodes(graph, field, data, key="nodes"):
    for el in graph[key]:
        if el["shortNm"] in data:
            el[field] = data[el["shortNm"]]
    return graph


def load_base_graph():
    file_name = os.path.join(data_path, "graph2.json")
    with open(file_name, "r") as fr:
        graph = json.load(fr)

    return graph


def replace_with_none(data, condition):
    for k in data:
        if condition(data[k]):
            data[k] = None
    return data


def create_graph(
    poll,
    elec,
    idx,
    folder_out,
    poll_scaling=1e-6,
    elec_scaling=1e-3,
    polli_scaling=1,
    unit="ktons",
    variable_name="CO2",
    poll_name="CO2",
    save_data=False,
):
    """
    Todo
    ----
    remove commented code
    add foreign bas, these are ignored for now
    links: check that if I go get the poll transfers directly I get the same
    result...

    Default scaling values are set for hourly data

    Assumes elec data comes in MWh and poll data comes in kg
    """
    # Unpack options
    legCircleTitle = legCircleTitles[variable_name] + " (%s)" % unit
    legLineTitle = legLineTitles[variable_name] + " (%s)" % unit
    title = titles[variable_name]
    legColorTitle = legColorTitles[poll_name]

    # Prepare data
    cols_poll = poll.get_cols(field="D")
    cols_elec = elec.get_cols(field="D")
    row_poll = poll.df.loc[idx, cols_poll].fillna(0.0).values
    row_elec = elec.df.loc[idx, cols_elec].fillna(0.0).values
    intensity_data = row_poll / row_elec * polli_scaling
    row_poll *= poll_scaling
    row_elec *= elec_scaling

    intensity_data = dict(zip(elec.regions, intensity_data))
    row_poll = dict(zip(elec.regions, row_poll))
    row_elec = dict(zip(elec.regions, row_elec))
    graph = load_base_graph()

    # Add node demand data
    #     for ba in elec.regions:
    #         el = row_poll.pop(ba)
    #         if el == 0.:
    #             el = None
    #         row_poll[ba] = el

    row_poll = replace_with_none(row_poll, lambda x: x == 0.0)
    row_elec = replace_with_none(row_elec, lambda x: x == 0.0)

    graph = add_data_nodes(graph, "poll_D", row_poll)
    graph = add_data_nodes(graph, "elec_D", row_elec)

    # Todo: labels field can probably be removed if a coords_labels
    # field is added to the dicts in the nodes field
    graph = add_data_nodes(graph, "poll_D", row_poll, "labels")
    graph = add_data_nodes(graph, "elec_D", row_elec, "labels")

    # Add node intensity data
    #     for ba in elec.regions:
    #         el = data.pop(ba)
    #         if np.isnan(el):
    #             el = None
    #         data[ba] = el
    intensity_data = replace_with_none(intensity_data, lambda x: np.isnan(x))
    graph = add_data_nodes(graph, "poll_Di", intensity_data)

    # Add node interconnect label
    data = {}
    for ba in elec.regions:
        if ba in WECC_BAs:
            data[ba] = "wecc"
        elif ba == "ERCO":
            data[ba] = "erco"
        else:
            data[ba] = "eic"
    graph = add_data_nodes(graph, "interconnect", data)

    # Add data for the links
    node_list = [el["shortNm"] for el in graph["nodes"]]
    shortNm2ind = {node["shortNm"]: i for i, node in enumerate(graph["nodes"])}

    links = []

    regions = node_list

    for i in range(len(regions)):
        for j in range(i, len(regions)):
            from_ba = regions[i]
            to_ba = regions[j]
            if (elec.KEY["ID"] % (from_ba, to_ba) in elec.df.columns) & (
                elec.KEY["ID"] % (to_ba, from_ba) in elec.df.columns
            ):
                elec_transfer = elec.df.loc[
                    :, elec.KEY["ID"] % (from_ba, to_ba)
                ].values[0]

                if elec_transfer < 0:  # Have this be positive
                    from_ba = regions[j]
                    to_ba = regions[i]
                    elec_transfer = elec.df.loc[
                        :, elec.KEY["ID"] % (from_ba, to_ba)
                    ].values[0]
                if intensity_data[from_ba] is None:
                    poll_TI = None
                else:
                    poll_TI = elec_transfer * intensity_data[from_ba] * poll_scaling
                links += [
                    {
                        "source": shortNm2ind[from_ba],
                        "target": shortNm2ind[to_ba],
                        "elec_TI": elec_transfer * elec_scaling,
                        "poll_TI": poll_TI,
                        "TI_i": intensity_data[from_ba],
                    }
                ]

    graph["links"] = links

    graph["meta"] = {
        "colorModeAuto": False,
        "fieldRadius": "poll_D",
        "fieldLineWidth": "poll_TI",
        "fieldCircle": "poll_Di",
        "fieldLineColor": "TI_i",
        "legColorTitle": legColorTitle,
        "legCircleTitle": legCircleTitle,
        "legLineTitle": legLineTitle,
        "unit": unit,
        "title": title,
        "timestamp": idx.tz_convert("US/Mountain").strftime("%Y%m%dT%H MT"),
    }

    if save_data:
        file_name = os.path.join(
            folder_out,
            "%s_%s_%si.json" % (idx.strftime("%Y%m%dT%HZ"), variable_name, poll_name),
        )
        with open(file_name, "w") as fw:
            json.dump(graph, fw)

    return graph
