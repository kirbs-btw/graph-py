import math

from graph_py.core import Edge, Graph, Node
from graph_py.graphs import DirectedGraph
from graph_py.metrics import (
    DegreeSummary,
    DirectedDegreeSummary,
    GraphMetrics,
)


def _add_nodes(graph: Graph, *node_ids: str) -> None:
    for node_id in node_ids:
        graph.add_node(Node(id=node_id))


def _add_edges(graph: Graph, *pairs: tuple[str, str]) -> None:
    for index, (source, target) in enumerate(pairs, start=1):
        graph.add_edge(Edge(id=f"e{index}", source=source, target=target))


def test_metrics_for_path_graph():
    graph = Graph(id="path")
    _add_nodes(graph, "A", "B", "C")
    _add_edges(graph, ("A", "B"), ("B", "C"))

    metrics = graph.compute_metrics(include_pairwise=True)

    assert isinstance(metrics, GraphMetrics)
    assert metrics.node_count == 3
    assert metrics.edge_count == 2
    assert metrics.component_count == 1
    assert metrics.component_sizes == [3]
    assert metrics.density is not None and math.isclose(metrics.density, 4 / 6, rel_tol=1e-6)
    assert isinstance(metrics.degree, DegreeSummary)
    assert math.isclose(metrics.degree.mean, 4 / 3, rel_tol=1e-6)
    assert metrics.degree.minimum == 1
    assert metrics.degree.maximum == 2
    assert metrics.degree.distribution == {1: 2, 2: 1}
    assert metrics.distance.diameter == 2
    assert metrics.distance.radius == 1
    assert math.isclose(metrics.distance.average_shortest_path_length, 4 / 3, rel_tol=1e-6)
    assert metrics.distance.eccentricity == {"A": 2.0, "B": 1.0, "C": 2.0}
    assert metrics.distance.pairwise_shortest_path_lengths["A"]["C"] == 2.0
    assert metrics.distance.pairwise_shortest_path_lengths["B"]["B"] == 0.0
    assert metrics.spectral_radius is not None
    assert math.isclose(metrics.spectral_radius, math.sqrt(2), rel_tol=1e-6)


def test_directed_graph_metrics():
    graph = DirectedGraph(id="cycle")
    _add_nodes(graph, "A", "B", "C")
    _add_edges(graph, ("A", "B"), ("B", "C"), ("C", "A"))

    metrics = graph.compute_metrics()

    assert metrics.is_directed is True
    assert metrics.component_sizes == [3]
    assert metrics.density is not None and math.isclose(metrics.density, 0.5, rel_tol=1e-6)
    assert isinstance(metrics.degree, DirectedDegreeSummary)
    assert math.isclose(metrics.degree.out_degree.mean, 1.0, rel_tol=1e-6)
    assert math.isclose(metrics.degree.in_degree.mean, 1.0, rel_tol=1e-6)
    assert metrics.distance.diameter == 1
    assert metrics.distance.radius == 1
    assert math.isclose(metrics.distance.average_shortest_path_length, 1.0, rel_tol=1e-6)
    assert metrics.spectral_radius is not None
    assert math.isclose(metrics.spectral_radius, 1.0, rel_tol=1e-6)


def test_empty_graph_metrics():
    graph = Graph(id="empty")

    metrics = graph.compute_metrics()

    assert metrics.node_count == 0
    assert metrics.edge_count == 0
    assert metrics.component_count == 0
    assert metrics.component_sizes == []
    assert metrics.density is None
    assert metrics.degree is None
    assert metrics.distance.average_shortest_path_length is None
    assert metrics.distance.diameter is None
    assert metrics.distance.eccentricity == {}
    assert metrics.spectral_radius is None
