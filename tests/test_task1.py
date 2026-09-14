import pydot

from project.task1 import build_and_save_two_cycles_graph, get_graph_info


def test_get_graph_info():
    info = get_graph_info("bzip")

    assert info.number_of_nodes == 632
    assert info.number_of_edges == 556
    assert info.labels == {"a", "d"}


def test_build_and_save_two_cycles_graph(tmp_path):
    path = tmp_path / "two_cycles.dot"

    build_and_save_two_cycles_graph(3, 2, ("a", "b"), path)

    (graph,) = pydot.graph_from_dot_file(path)
    nodes = {node.get_name() for node in graph.get_nodes()}
    edges = graph.get_edges()
    edge_labels = {edge.get_label().strip('"') for edge in edges}

    # 3 + 2 cycle nodes + 1 shared node
    assert len(nodes) == 6
    # (3 + 1) edges in the first cycle + (2 + 1) edges in the second
    assert len(edges) == 7
    assert edge_labels == {"a", "b"}
