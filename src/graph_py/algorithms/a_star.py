from __future__ import annotations

from heapq import heappop, heappush
from math import inf
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..core import Graph, Node
from ._utils import DEFAULT_WEIGHT, build_weight_lookup, ensure_node, get_edge_weight

Heuristic = Callable[[Node, Node], float]


def a_star(
    graph: Graph,
    start: Union[Node, str],
    goal: Union[Node, str],
    *,
    heuristic: Optional[Heuristic] = None,
    weight_attr: str = "weight",
    default_weight: float = DEFAULT_WEIGHT,
) -> Optional[List[Node]]:
    """
    Find the shortest path between two nodes using the A* algorithm.

    Args:
        graph: Graph to traverse.
        start: Starting node or node identifier.
        goal: Target node or node identifier.
        heuristic: Optional heuristic function returning the estimated remaining
            cost between a node and the goal. Defaults to zero (Dijkstra).
        weight_attr: Edge attribute used for weighting edges.
        default_weight: Fallback edge weight when attribute is not present or invalid.

    Returns:
        The optimal path as a list of nodes, or None when no path exists.
    """
    start_node = ensure_node(graph, start)
    goal_node = ensure_node(graph, goal)

    if start_node.id == goal_node.id:
        return [start_node]

    adjacency = graph.adjacency
    id_to_node = {node.id: node for node in graph.nodes}

    weight_lookup = build_weight_lookup(
        graph,
        weight_attr=weight_attr,
        default_weight=default_weight,
    )

    g_score: Dict[str, float] = {node.id: inf for node in graph.nodes}
    g_score[start_node.id] = 0.0

    f_score: Dict[str, float] = {node.id: inf for node in graph.nodes}
    f_score[start_node.id] = _heuristic(heuristic, start_node, goal_node)

    came_from: Dict[str, Optional[str]] = {node.id: None for node in graph.nodes}

    open_heap: List[Tuple[float, str]] = [(f_score[start_node.id], start_node.id)]
    in_open: Dict[str, bool] = {start_node.id: True}

    while open_heap:
        _, current_id = heappop(open_heap)
        in_open[current_id] = False

        if current_id == goal_node.id:
            return _reconstruct_path(current_id, came_from, id_to_node)

        current_g = g_score[current_id]
        for neighbor_id in adjacency.get(current_id, []):
            tentative_g = current_g + get_edge_weight(
                weight_lookup,
                current_id,
                neighbor_id,
                default_weight=default_weight,
            )
            if tentative_g >= g_score[neighbor_id]:
                continue

            g_score[neighbor_id] = tentative_g
            neighbor_node = id_to_node[neighbor_id]
            f_score[neighbor_id] = tentative_g + _heuristic(heuristic, neighbor_node, goal_node)
            came_from[neighbor_id] = current_id

            if not in_open.get(neighbor_id, False):
                heappush(open_heap, (f_score[neighbor_id], neighbor_id))
                in_open[neighbor_id] = True

    return None


def _heuristic(
    heuristic: Optional[Heuristic],
    node: Node,
    goal: Node,
) -> float:
    if not heuristic:
        return 0.0
    try:
        value = heuristic(node, goal)
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _reconstruct_path(
    current_id: str,
    came_from: Dict[str, Optional[str]],
    id_to_node: Dict[str, Node],
) -> List[Node]:
    path_ids = [current_id]
    while came_from[current_id] is not None:
        current_id = came_from[current_id]  # type: ignore[assignment]
        path_ids.append(current_id)
    path_ids.reverse()
    return [id_to_node[node_id] for node_id in path_ids]


__all__ = ["a_star"]

