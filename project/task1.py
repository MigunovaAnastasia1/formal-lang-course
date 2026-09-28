from dataclasses import dataclass
from pathlib import Path

import cfpq_data
from networkx import MultiDiGraph
from networkx.drawing.nx_pydot import to_pydot


@dataclass
class GraphInfo:
    number_of_nodes: int
    number_of_edges: int
    labels: set[str]


def get_graph_info(name: str) -> GraphInfo:
    graph_path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(graph_path)
    labels = {data["label"] for _, _, data in graph.edges(data=True) if "label" in data}
    return GraphInfo(
        number_of_nodes=graph.number_of_nodes(),
        number_of_edges=graph.number_of_edges(),
        labels=labels,
    )


def build_and_save_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
    path: str | Path,
) -> None:
    graph: MultiDiGraph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels=labels
    )
    to_pydot(graph).write_raw(str(path))
