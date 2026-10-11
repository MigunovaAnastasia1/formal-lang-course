from collections.abc import Iterable

import scipy.sparse as sp
from networkx import MultiDiGraph
from numpy import zeros
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol

from project.task2 import graph_to_nfa, regex_to_dfa


def _transitive_closure(
    matrices: dict[Symbol, sp.csr_matrix], states_count: int
) -> sp.csr_matrix:
    reachability = sp.eye(states_count, dtype=bool, format="csr")
    for matrix in matrices.values():
        reachability = reachability + matrix

    steps = max(1, (states_count - 1).bit_length())
    for _ in range(steps):
        reachability = reachability @ reachability

    return reachability


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton):
        states = list(automaton.states)
        self.states_count = len(states)
        self.index_to_state = states
        self.state_to_index = {state: index for index, state in enumerate(states)}

        self.start_states = {self.state_to_index[s] for s in automaton.start_states}
        self.final_states = {self.state_to_index[s] for s in automaton.final_states}

        self.matrices: dict[Symbol, sp.csr_matrix] = {}
        transitions = automaton.to_dict()
        for symbol in automaton.symbols:
            matrix = sp.dok_matrix((self.states_count, self.states_count), dtype=bool)
            for source, symbol_to_targets in transitions.items():
                targets = symbol_to_targets.get(symbol)
                if targets is None:
                    continue
                if not isinstance(targets, set):
                    targets = {targets}
                for target in targets:
                    matrix[self.state_to_index[source], self.state_to_index[target]] = (
                        True
                    )
            self.matrices[symbol] = matrix.tocsr()

    @classmethod
    def _from_parts(
        cls,
        states_count: int,
        matrices: dict[Symbol, sp.csr_matrix],
        start_states: set[int],
        final_states: set[int],
    ) -> "AdjacencyMatrixFA":
        instance = cls.__new__(cls)
        instance.states_count = states_count
        instance.matrices = matrices
        instance.start_states = start_states
        instance.final_states = final_states
        return instance

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current = zeros(self.states_count, dtype=bool)
        current[list(self.start_states)] = True

        for symbol in word:
            matrix = self.matrices.get(symbol)
            if matrix is None:
                return False
            current = current @ matrix
            if not current.any():
                return False

        return bool(current[list(self.final_states)].any())

    def is_empty(self) -> bool:
        if self.start_states & self.final_states:
            return False
        if self.states_count == 0:
            return True

        reachability = _transitive_closure(self.matrices, self.states_count)

        for start in self.start_states:
            for final in self.final_states:
                if reachability[start, final]:
                    return False

        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    states_count = automaton1.states_count * automaton2.states_count
    common_symbols = automaton1.matrices.keys() & automaton2.matrices.keys()

    matrices = {
        symbol: sp.kron(
            automaton1.matrices[symbol], automaton2.matrices[symbol], format="csr"
        )
        for symbol in common_symbols
    }

    def flatten(index1: int, index2: int) -> int:
        return index1 * automaton2.states_count + index2

    start_states = {
        flatten(i, j) for i in automaton1.start_states for j in automaton2.start_states
    }
    final_states = {
        flatten(i, j) for i in automaton1.final_states for j in automaton2.final_states
    }

    return AdjacencyMatrixFA._from_parts(
        states_count, matrices, start_states, final_states
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersection = intersect_automata(regex_fa, graph_fa)
    reachability = _transitive_closure(intersection.matrices, intersection.states_count)

    result = set()
    for regex_start in regex_fa.start_states:
        for regex_final in regex_fa.final_states:
            for graph_start_index in graph_fa.start_states:
                for graph_final_index in graph_fa.final_states:
                    start = regex_start * graph_fa.states_count + graph_start_index
                    final = regex_final * graph_fa.states_count + graph_final_index
                    if reachability[start, final]:
                        graph_start_node = graph_fa.index_to_state[
                            graph_start_index
                        ].value
                        graph_final_node = graph_fa.index_to_state[
                            graph_final_index
                        ].value
                        result.add((graph_start_node, graph_final_node))

    return result
