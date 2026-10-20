from collections import defaultdict, deque

from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Production, Terminal


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    productions = set(cfg.to_normal_form().productions)
    if cfg.start_symbol in cfg.get_nullable_symbols():
        productions.add(Production(cfg.start_symbol, []))
    return CFG(start_symbol=cfg.start_symbol, productions=productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    weak_cfg = cfg_to_weak_normal_form(cfg)

    start_is_nullable = False
    terminal_productions = defaultdict(set)
    productions_by_first = defaultdict(set)
    productions_by_second = defaultdict(set)
    for production in weak_cfg.productions:
        body = production.body
        if len(body) == 0:
            start_is_nullable = True
        elif len(body) == 1:
            terminal_productions[body[0]].add(production.head)
        else:
            first, second = body
            productions_by_first[first].add((production.head, second))
            productions_by_second[second].add((production.head, first))

    starts_from = defaultdict(lambda: defaultdict(set))
    ends_at = defaultdict(lambda: defaultdict(set))
    queue = deque()

    def add(v, variable, u):
        if u not in starts_from[v][variable]:
            starts_from[v][variable].add(u)
            ends_at[u][variable].add(v)
            queue.append((v, variable, u))

    if start_is_nullable:
        for node in graph.nodes:
            add(node, weak_cfg.start_symbol, node)

    for v, u, label in graph.edges(data="label"):
        for variable in terminal_productions[Terminal(label)]:
            add(v, variable, u)

    while queue:
        v, variable, u = queue.popleft()

        for head, second in productions_by_first[variable]:
            for w in list(starts_from[u][second]):
                add(v, head, w)

        for head, first in productions_by_second[variable]:
            for w in list(ends_at[v][first]):
                add(w, head, u)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    return {
        (v, u)
        for v in start_nodes
        for u in starts_from[v][weak_cfg.start_symbol]
        if u in final_nodes
    }
