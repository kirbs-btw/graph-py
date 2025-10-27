from __future__ import annotations

from typing import Dict, List, Tuple

from ..core import Graph
from ..graphs import DirectedGraph
from ._utils import DEFAULT_WEIGHT, clone_edge, clone_node


def minimum_spanning_tree(
    graph: Graph,
    *,
    weight_attr: str = "weight",
    default_weight: float = DEFAULT_WEIGHT,
) -> Graph:
    """
    Compute a minimum spanning tree for the given undirected graph.

    Args:
        graph: Source graph (must be undirected).
        weight_attr: Edge attribute providing weights.
        default_weight: Fallback weight when no attribute exists.

    Returns:
        A new graph instance representing the MST.

    Raises:
        ValueError: If the graph is directed.
    """
    if isinstance(graph, DirectedGraph):
        raise ValueError("Minimum spanning tree requires an undirected graph.")

    mst_graph = type(graph)(  # type: ignore[call-arg]
        id=f"{graph.id}_mst",
        name=f"MST({graph.name or graph.id})",
        nodes=[],
        edges=[],
    )

    for node in graph.nodes:
        mst_graph.add_node(clone_node(node))

    edges = sorted(
        graph.edges,
        key=lambda edge: _edge_weight(edge, weight_attr=weight_attr, default_weight=default_weight),
    )

    parent = {node.id: node.id for node in graph.nodes}
    rank = {node.id: 0 for node in graph.nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    added_edges = 0
    for edge in edges:
        source_root = find(edge.source)
        target_root = find(edge.target)
        if source_root == target_root:
            continue
        mst_graph.add_edge(clone_edge(edge))
        union(source_root, target_root)
        added_edges += 1
        if added_edges == len(graph.nodes) - 1:
            break

    return mst_graph


def _edge_weight(edge, *, weight_attr: str, default_weight: float) -> float:
    value = getattr(edge, weight_attr, None)
    if value is None and weight_attr != "weight":
        value = getattr(edge, "weight", None)
    if value is None:
        value = default_weight
    try:
        return float(value)
    except (TypeError, ValueError):
        return default_weight


__all__ = ["minimum_spanning_tree"]

