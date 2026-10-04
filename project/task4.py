import scipy.sparse as sp
from networkx import MultiDiGraph

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersection = intersect_automata(regex_fa, graph_fa)

    combined = sp.csr_matrix(
        (intersection.states_count, intersection.states_count), dtype=bool
    )
    for matrix in intersection.matrices.values():
        combined = combined + matrix

    def flatten(regex_index: int, graph_index: int) -> int:
        return regex_index * graph_fa.states_count + graph_index

    regex_start = next(iter(regex_fa.start_states))
    graph_start_indices = list(graph_fa.start_states)
    sources_count = len(graph_start_indices)

    frontier = sp.dok_matrix((sources_count, intersection.states_count), dtype=bool)
    for row, graph_start in enumerate(graph_start_indices):
        frontier[row, flatten(regex_start, graph_start)] = True
    frontier = frontier.tocsr()

    visited = frontier
    for _ in range(max(1, intersection.states_count)):
        frontier = frontier @ combined
        new_visited = visited + frontier
        if (new_visited != visited).nnz == 0:
            break
        visited = new_visited

    result = set()
    for row, graph_start in enumerate(graph_start_indices):
        graph_start_node = graph_fa.index_to_state[graph_start].value
        for regex_final in regex_fa.final_states:
            for graph_final in graph_fa.final_states:
                if visited[row, flatten(regex_final, graph_final)]:
                    graph_final_node = graph_fa.index_to_state[graph_final].value
                    result.add((graph_start_node, graph_final_node))

    return result
