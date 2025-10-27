from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Tuple, Union

from ..core import Graph, Node
from ..graphs import DirectedGraph
from ._utils import DEFAULT_WEIGHT, build_weight_lookup, ensure_node, get_edge_weight


@dataclass(frozen=True)
class FloydWarshallResult:
    """Distances and next-hop pointers produced by the Floyd–Warshall algorithm."""

    distances: Dict[str, Dict[str, float]]
    next_hop: Dict[str, Dict[str, Optional[str]]]

    def reconstruct_path(self, graph: Graph, source: Union[Node, str], target: Union[Node, str]) -> Optional[List[Node]]:
        """Rebuild the minimum path between the given nodes."""
        start = ensure_node(graph, source)
        end = ensure_node(graph, target)
        if self.next_hop[start.id][end.id] is None:
            if start.id == end.id:
                return [start]
            return None

        path_ids = [start.id]
        current = start.id
        while current != end.id:
            current = self.next_hop[current][end.id]  # type: ignore[assignment]
            if current is None:
                return None
            path_ids.append(current)

        id_to_node = {node.id: node for node in graph.nodes}
        return [id_to_node[node_id] for node_id in path_ids]


def floyd_warshall(
    graph: Graph,
    *,
    weight_attr: str = "weight",
    default_weight: float = DEFAULT_WEIGHT,
) -> FloydWarshallResult:
    """
    Compute all-pairs shortest paths using the Floyd–Warshall algorithm.

    Args:
        graph: Graph to analyse.
        weight_attr: Edge attribute providing weights.
        default_weight: Fallback weight when an edge does not expose the attribute.

    Returns:
        FloydWarshallResult containing distance and next-hop matrices.
    """
    node_ids = [node.id for node in graph.nodes]
    distances: Dict[str, Dict[str, float]] = {
        src: {dst: (0.0 if src == dst else inf) for dst in node_ids}
        for src in node_ids
    }
    next_hop: Dict[str, Dict[str, Optional[str]]] = {
        src: {dst: None for dst in node_ids}
        for src in node_ids
    }

    weight_lookup = build_weight_lookup(
        graph,
        weight_attr=weight_attr,
        default_weight=default_weight,
    )

    directed = isinstance(graph, DirectedGraph)

    for edge in graph.edges:
        weight = get_edge_weight(
            weight_lookup,
            edge.source,
            edge.target,
            default_weight=default_weight,
        )
        if weight < distances[edge.source][edge.target]:
            distances[edge.source][edge.target] = weight
            next_hop[edge.source][edge.target] = edge.target
        if not directed:
            if weight < distances[edge.target][edge.source]:
                distances[edge.target][edge.source] = weight
                next_hop[edge.target][edge.source] = edge.source

    for k in node_ids:
        for i in node_ids:
            dik = distances[i][k]
            if dik == inf:
                continue
            for j in node_ids:
                new_dist = dik + distances[k][j]
                if new_dist < distances[i][j]:
                    distances[i][j] = new_dist
                    next_hop[i][j] = next_hop[i][k]

    return FloydWarshallResult(distances=distances, next_hop=next_hop)


__all__ = ["floyd_warshall", "FloydWarshallResult"]
