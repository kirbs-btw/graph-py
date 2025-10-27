from __future__ import annotations

from heapq import heappop, heappush
from math import inf
from typing import Dict, List, Optional, Tuple, Union

from ..core import Graph, Node
from ._utils import DEFAULT_WEIGHT, build_weight_lookup, ensure_node, get_edge_weight


def dijkstra(
    graph: Graph,
    src_node: Union[Node, str],
    trg_node: Union[Node, str],
) -> Optional[List[Node]]:
    """Compute the shortest path between two nodes using Dijkstra's algorithm."""
    src = ensure_node(graph, src_node)
    trg = ensure_node(graph, trg_node)

    if src.id == trg.id:
        return [src]

    adjacency = graph.adjacency
    id_to_node = {node.id: node for node in graph.nodes}

    weight_lookup = build_weight_lookup(graph)

    distances: Dict[str, float] = {node.id: inf for node in graph.nodes}
    previous: Dict[str, Optional[str]] = {node.id: None for node in graph.nodes}
    distances[src.id] = 0.0

    heap: List[Tuple[float, str]] = [(0.0, src.id)]

    while heap:
        current_distance, current_id = heappop(heap)
        if current_distance > distances[current_id]:
            continue

        if current_id == trg.id:
            break

        for neighbor_id in adjacency.get(current_id, []):
            weight = get_edge_weight(
                weight_lookup,
                current_id,
                neighbor_id,
                default_weight=DEFAULT_WEIGHT,
            )
            if weight < 0:
                raise ValueError("Dijkstra's algorithm requires non-negative edge weights")

            candidate_distance = current_distance + weight
            if candidate_distance < distances[neighbor_id]:
                distances[neighbor_id] = candidate_distance
                previous[neighbor_id] = current_id
                heappush(heap, (candidate_distance, neighbor_id))

    if distances[trg.id] == inf:
        return None

    path_ids = []
    current_id: Optional[str] = trg.id
    while current_id is not None:
        path_ids.append(current_id)
        current_id = previous[current_id]
    path_ids.reverse()

    return [id_to_node[node_id] for node_id in path_ids]


__all__ = ["dijkstra"]
