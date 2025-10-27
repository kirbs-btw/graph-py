from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

from ..core import Graph, Node

DEFAULT_WEIGHT = 1.0


def ensure_node(graph: Graph, node: Union[Node, str]) -> Node:
    """Return a node instance for either an object or identifier."""
    if isinstance(node, Node):
        if node not in graph.nodes:
            raise ValueError(f"Node '{node.id}' is not part of graph '{graph.id}'.")
        return node
    resolved = graph.get_node(node)
    if not resolved:
        raise ValueError(f"Node '{node}' is not part of graph '{graph.id}'.")
    return resolved


def build_weight_lookup(
    graph: Graph,
    *,
    weight_attr: str = "weight",
    default_weight: float = DEFAULT_WEIGHT,
) -> Dict[Tuple[str, str], float]:
    """Precompute edge weights for fast lookup."""
    lookup: Dict[Tuple[str, str], float] = {}
    for edge in graph.edges:
        value = getattr(edge, weight_attr, None)
        if value is None and weight_attr != "weight":
            value = getattr(edge, "weight", None)
        if value is None:
            value = default_weight
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = default_weight
        lookup[(edge.source, edge.target)] = weight
    return lookup


def get_edge_weight(
    lookup: Dict[Tuple[str, str], float],
    source: str,
    target: str,
    *,
    default_weight: float = DEFAULT_WEIGHT,
) -> float:
    """Resolve the weight between two node identifiers."""
    if (source, target) in lookup:
        return lookup[(source, target)]
    if (target, source) in lookup:
        return lookup[(target, source)]
    return default_weight


def clone_node(node: Node) -> Node:
    """Create a detached copy of a node."""
    clone = node.copy(deep=True)
    clone.graph = None
    return clone

