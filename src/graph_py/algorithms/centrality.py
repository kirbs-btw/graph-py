from __future__ import annotations

from typing import Dict, Optional

import networkx as nx

from ..core import Graph
from ..visualization import graph_to_networkx


def degree_centrality(graph: Graph, *, normalized: bool = True) -> Dict[str, float]:
    """Compute degree centrality for each node."""
    nx_graph = graph_to_networkx(graph)
    if normalized:
        centrality = nx.degree_centrality(nx_graph)
        return {node: float(value) for node, value in centrality.items()}
    return {node: float(degree) for node, degree in nx_graph.degree()}


def closeness_centrality(graph: Graph, *, weight: Optional[str] = None) -> Dict[str, float]:
    """Compute closeness centrality for each node."""
    nx_graph = graph_to_networkx(graph)
    centrality = nx.closeness_centrality(nx_graph, distance=weight)
    return {node: float(value) for node, value in centrality.items()}


def betweenness_centrality(
    graph: Graph,
    *,
    weight: Optional[str] = None,
    normalized: bool = True,
) -> Dict[str, float]:
    """Compute betweenness centrality for each node."""
    nx_graph = graph_to_networkx(graph)
    centrality = nx.betweenness_centrality(nx_graph, weight=weight, normalized=normalized)
    return {node: float(value) for node, value in centrality.items()}


def eigenvector_centrality(
    graph: Graph,
    *,
    weight: Optional[str] = "weight",
    max_iter: int = 100,
    tol: float = 1e-06,
) -> Dict[str, float]:
    """Compute eigenvector centrality for each node."""
    nx_graph = graph_to_networkx(graph)
    centrality = nx.eigenvector_centrality(nx_graph, max_iter=max_iter, tol=tol, weight=weight)
    return {node: float(value) for node, value in centrality.items()}


__all__ = [
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
]

