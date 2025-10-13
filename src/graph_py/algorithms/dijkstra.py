from __future__ import annotations

from heapq import heappop, heappush
from math import inf
from typing import Dict, List, Optional, Tuple

from ..core import Graph, Node


def dijkstra(graph: Graph, src_node: Node, trg_node: Node) -> Optional[List[Node]]:
    """Compute the shortest path between two nodes using Dijkstra's algorithm."""
    assert (
        src_node in graph.nodes and trg_node in graph.nodes
    ), "target and source node need to be in the same graph"

    if src_node.id == trg_node.id:
        return [src_node]

    adjacency = graph.adjacency
    id_to_node = {node.id: node for node in graph.nodes}

    weight_lookup = _build_weight_lookup(graph)

    distances: Dict[str, float] = {node.id: inf for node in graph.nodes}
    previous: Dict[str, Optional[str]] = {node.id: None for node in graph.nodes}
    distances[src_node.id] = 0.0

    heap: List[Tuple[float, str]] = [(0.0, src_node.id)]

    while heap:
        current_distance, current_id = heappop(heap)
        if current_distance > distances[current_id]:
            continue

        if current_id == trg_node.id:
            break

        for neighbor_id in adjacency.get(current_id, []):
            weight = _resolve_weight(weight_lookup, current_id, neighbor_id)
            if weight < 0:
                raise ValueError("Dijkstra's algorithm requires non-negative edge weights")

            candidate_distance = current_distance + weight
            if candidate_distance < distances[neighbor_id]:
                distances[neighbor_id] = candidate_distance
                previous[neighbor_id] = current_id
                heappush(heap, (candidate_distance, neighbor_id))

    if distances[trg_node.id] == inf:
        return None

    path_ids = []
    current_id: Optional[str] = trg_node.id
    while current_id is not None:
        path_ids.append(current_id)
        current_id = previous[current_id]
    path_ids.reverse()

    return [id_to_node[node_id] for node_id in path_ids]


def _build_weight_lookup(graph: Graph) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    for edge in graph.edges:
        weight = getattr(edge, "weight", 1.0)
        try:
            lookup[(edge.source, edge.target)] = float(weight)
        except (TypeError, ValueError):
            lookup[(edge.source, edge.target)] = 1.0
    return lookup


def _resolve_weight(
    lookup: Dict[Tuple[str, str], float], source: str, target: str
) -> float:
    if (source, target) in lookup:
        return lookup[(source, target)]
    if (target, source) in lookup:
        return lookup[(target, source)]
    return 1.0


__all__ = ["dijkstra"]
