from typing import Optional, List

from ..core import Graph, Node


def dfs(graph: Graph, src_node: Node, trg_node: Node) -> Optional[List[Node]]:
    """Run depth-first search and return a path as nodes if reachable."""
    assert (
        src_node in graph.nodes and trg_node in graph.nodes
    ), "target and source node need to be in the same graph"

    if src_node.id == trg_node.id:
        return [src_node]

    adjacency = graph.adjacency
    id_to_node = {node.id: node for node in graph.nodes}

    visited = set()
    stack: List[tuple[str, List[str]]] = [(src_node.id, [src_node.id])]

    while stack:
        current_id, path = stack.pop()
        if current_id in visited:
            continue
        visited.add(current_id)

        if current_id == trg_node.id:
            return [id_to_node[node_id] for node_id in path]

        for neighbor_id in reversed(adjacency.get(current_id, [])):
            if neighbor_id in visited:
                continue
            stack.append((neighbor_id, path + [neighbor_id]))

    return None
