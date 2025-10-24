from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Union

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from .core import Graph
from .graphs.directed import DirectedGraph


class DegreeSummary(BaseModel):
    """Aggregated degree information for an undirected graph."""

    mean: float
    minimum: int
    maximum: int
    distribution: Dict[int, int] = Field(default_factory=dict)


class DirectedDegreeSummary(BaseModel):
    """Separate degree information for the in- and out-degree of a digraph."""

    out_degree: DegreeSummary
    in_degree: DegreeSummary


class DistanceSummary(BaseModel):
    """Distance-related metrics, computed on the largest connected component."""

    average_shortest_path_length: Optional[float] = None
    diameter: Optional[int] = None
    radius: Optional[int] = None
    eccentricity: Dict[str, float] = Field(default_factory=dict)
    pairwise_shortest_path_lengths: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class GraphMetrics(BaseModel):
    """High-level graph metrics combining size, degree and spectral information."""

    node_count: int
    edge_count: int
    is_directed: bool
    component_count: int
    component_sizes: List[int] = Field(default_factory=list)
    density: Optional[float] = None
    degree: Optional[Union[DegreeSummary, DirectedDegreeSummary]] = None
    distance: DistanceSummary = Field(default_factory=DistanceSummary)
    spectral_radius: Optional[float] = None


def build_networkx_graph(graph: Graph) -> nx.Graph:
    """Convert the in-memory Graph representation into a networkx graph."""
    is_directed = isinstance(graph, DirectedGraph)
    nx_graph: nx.Graph = nx.DiGraph() if is_directed else nx.Graph()

    for node in graph.nodes:
        nx_graph.add_node(node.id, name=node.name, raw=node)
    for edge in graph.edges:
        nx_graph.add_edge(edge.source, edge.target, id=edge.id, name=edge.name, raw=edge)

    return nx_graph


def compute_metrics(graph: Graph, *, include_pairwise: bool = False) -> GraphMetrics:
    """Generate a comprehensive metrics summary for the provided graph."""
    nx_graph = build_networkx_graph(graph)
    node_count = nx_graph.number_of_nodes()
    edge_count = nx_graph.number_of_edges()
    density = _compute_density(nx_graph)
    component_sizes = _component_sizes(nx_graph)
    degree_summary = _compute_degree_summary(nx_graph)
    distance_summary = _compute_distance_summary(nx_graph, include_pairwise=include_pairwise)
    spectral_radius = _compute_spectral_radius(nx_graph)

    return GraphMetrics(
        node_count=node_count,
        edge_count=edge_count,
        is_directed=nx_graph.is_directed(),
        component_count=len(component_sizes),
        component_sizes=component_sizes,
        density=density,
        degree=degree_summary,
        distance=distance_summary,
        spectral_radius=spectral_radius,
    )


def _compute_density(graph: nx.Graph) -> Optional[float]:
    if graph.number_of_nodes() == 0:
        return None
    return float(nx.density(graph))


def _component_sizes(graph: nx.Graph) -> List[int]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    return sorted((len(component) for component in components), reverse=True)


def _compute_degree_summary(graph: nx.Graph) -> Optional[Union[DegreeSummary, DirectedDegreeSummary]]:
    if graph.number_of_nodes() == 0:
        return None
    if graph.is_directed():
        out_summary = _summarize_degrees(degree for _, degree in graph.out_degree())
        in_summary = _summarize_degrees(degree for _, degree in graph.in_degree())
        return DirectedDegreeSummary(out_degree=out_summary, in_degree=in_summary)
    return _summarize_degrees(degree for _, degree in graph.degree())


def _summarize_degrees(degrees: Iterable[int]) -> DegreeSummary:
    degree_list = list(degrees)
    if not degree_list:
        return DegreeSummary(mean=0.0, minimum=0, maximum=0, distribution={})
    distribution = Counter(int(value) for value in degree_list)
    mean_value = float(sum(degree_list) / len(degree_list))
    return DegreeSummary(
        mean=mean_value,
        minimum=int(min(degree_list)),
        maximum=int(max(degree_list)),
        distribution=dict(distribution),
    )


def _compute_distance_summary(graph: nx.Graph, *, include_pairwise: bool) -> DistanceSummary:
    summary = DistanceSummary()
    if graph.number_of_nodes() == 0:
        return summary

    undirected = graph.to_undirected() if graph.is_directed() else graph
    if undirected.number_of_nodes() == 0:
        return summary

    components = list(nx.connected_components(undirected))
    if not components:
        return summary

    largest_nodes = max(components, key=len)
    component_graph = undirected.subgraph(largest_nodes).copy()

    if component_graph.number_of_nodes() == 1:
        node = next(iter(component_graph.nodes))
        summary.average_shortest_path_length = 0.0
        summary.diameter = 0
        summary.radius = 0
        summary.eccentricity = {node: 0.0}
        if include_pairwise:
            summary.pairwise_shortest_path_lengths = {node: {node: 0.0}}
        return summary

    summary.average_shortest_path_length = float(nx.average_shortest_path_length(component_graph))
    summary.diameter = int(nx.diameter(component_graph))
    summary.radius = int(nx.radius(component_graph))
    summary.eccentricity = {
        node: float(distance) for node, distance in nx.eccentricity(component_graph).items()
    }

    if include_pairwise:
        pairwise: Dict[str, Dict[str, float]] = {}
        for source, lengths in nx.shortest_path_length(component_graph):
            pairwise[source] = {target: float(dist) for target, dist in lengths.items()}
        summary.pairwise_shortest_path_lengths = pairwise

    return summary


def _compute_spectral_radius(graph: nx.Graph) -> Optional[float]:
    if graph.number_of_nodes() == 0:
        return None
    matrix = nx.to_numpy_array(graph, dtype=float)
    if matrix.size == 0:
        return None
    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None
    if eigenvalues.size == 0:
        return None
    return float(np.max(np.abs(eigenvalues)))


__all__ = [
    "GraphMetrics",
    "DegreeSummary",
    "DirectedDegreeSummary",
    "DistanceSummary",
    "build_networkx_graph",
    "compute_metrics",
]
