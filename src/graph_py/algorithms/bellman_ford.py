from __future__ import annotations

from math import inf
from typing import Dict, List, Optional, Tuple

from ..core import Graph, Node


def bellman_ford(graph: Graph, src_node: Node, trg_node: Node) -> Optional[List[Node]]:
    """Compute the shortest path allowing negative edge weights."""
    assert (
        src_node in graph.nodes and trg_node in graph.nodes
    ), "target and source node need to be in the same graph"

    if src_node.id == trg_node.id:
        return [src_node]

    adjacency = graph.adjacency
    id_to_node = {node.id: node for node in graph.nodes}

    distances: Dict[str, float] = {node.id: inf for node in graph.nodes}
    previous: Dict[str, Optional[str]] = {node.id: None for node in graph.nodes}
    distances[src_node.id] = 0.0

    weight_lookup = _build_weight_lookup(graph)
    edges = _expand_edges(adjacency, weight_lookup)

    for _ in range(len(graph.nodes) - 1):
        updated = False
        for source_id, target_id, weight in edges:
            if distances[source_id] == inf:
                continue
            candidate = distances[source_id] + weight
            if candidate < distances[target_id]:
                distances[target_id] = candidate
                previous[target_id] = source_id
                updated = True
        if not updated:
            break

    for source_id, target_id, weight in edges:
        if distances[source_id] + weight < distances[target_id]:
            raise ValueError("Bellman-Ford algorithm detected a negative-weight cycle")

    if distances[trg_node.id] == inf:
        return None

    path_ids: List[str] = []
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


def _expand_edges(
    adjacency: Dict[str, List[str]],
    lookup: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str, float]]:
    edges: List[Tuple[str, str, float]] = []
    for source_id, targets in adjacency.items():
        for target_id in targets:
            edges.append((source_id, target_id, _resolve_weight(lookup, source_id, target_id)))
    return edges


__all__ = ["bellman_ford"]
