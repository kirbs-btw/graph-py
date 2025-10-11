from collections import deque
from typing import Optional, List

from ..core import Graph, Node


def bfs(graph: Graph, src_node: Node, trg_node: Node) -> Optional[List[Node]]:
    """Run breadth-first search and return the shortest path as nodes."""
    assert (
        src_node in graph.nodes and trg_node in graph.nodes
    ), "target and source node need to be in the same graph"

    if src_node.id == trg_node.id:
        return [src_node]

    adjacency = graph.adjacency
    id_to_node = {node.id: node for node in graph.nodes}

    visited = {src_node.id}
    queue = deque([(src_node.id, [src_node.id])])

    while queue:
        current_id, path = queue.popleft()
        for neighbor_id in adjacency.get(current_id, []):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            next_path = path + [neighbor_id]
            if neighbor_id == trg_node.id:
                return [id_to_node[node_id] for node_id in next_path]
            queue.append((neighbor_id, next_path))

    return None
