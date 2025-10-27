import networkx as nx
import pytest

from graph_py import Graph, DirectedGraph, Node, Edge
from graph_py.algorithms import (
    a_star,
    floyd_warshall,
    minimum_spanning_tree,
    max_flow,
    min_cut,
    degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
)
from graph_py.visualization import graph_to_networkx


class WeightedEdge(Edge):
    weight: float


class CapacityEdge(Edge):
    capacity: float


def test_a_star_finds_weighted_shortest_path():
    graph = Graph(id="weighted")
    positions = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (2, 0),
        "D": (0, 1),
        "E": (1, 1),
    }
    for node_id in positions:
        graph.add_node(Node(id=node_id))

    graph.add_edge(WeightedEdge(id="ab", source="A", target="B", weight=1))
    graph.add_edge(WeightedEdge(id="bc", source="B", target="C", weight=2))
    graph.add_edge(WeightedEdge(id="be", source="B", target="E", weight=5))
    graph.add_edge(WeightedEdge(id="ad", source="A", target="D", weight=2))
    graph.add_edge(WeightedEdge(id="dc", source="D", target="C", weight=1))
    graph.add_edge(WeightedEdge(id="ce", source="C", target="E", weight=1))

    def heuristic(node: Node, goal: Node) -> float:
        x1, y1 = positions[node.id]
        x2, y2 = positions[goal.id]
        return abs(x1 - x2) + abs(y1 - y2)

    path = a_star(graph, "A", "E", heuristic=heuristic)
    assert path is not None
    path_ids = [node.id for node in path]
    assert path_ids[0] == "A"
    assert path_ids[-1] == "E"
    assert len(path_ids) == 4

    total_weight = 0.0
    for left, right in zip(path, path[1:]):
        for edge in graph.edges:
            if {edge.source, edge.target} == {left.id, right.id}:
                total_weight += getattr(edge, "weight", 1.0)
                break
    assert total_weight == pytest.approx(4.0)


def test_floyd_warshall_all_pairs_paths():
    graph = Graph(id="all_pairs")
    for node_id in ["A", "B", "C", "D"]:
        graph.add_node(Node(id=node_id))

    graph.add_edge(WeightedEdge(id="ab", source="A", target="B", weight=1))
    graph.add_edge(WeightedEdge(id="bc", source="B", target="C", weight=2))
    graph.add_edge(WeightedEdge(id="ac", source="A", target="C", weight=5))
    graph.add_edge(WeightedEdge(id="cd", source="C", target="D", weight=1))
    graph.add_edge(WeightedEdge(id="bd", source="B", target="D", weight=4))

    result = floyd_warshall(graph)
    assert result.distances["A"]["D"] == pytest.approx(4.0)

    path = result.reconstruct_path(graph, "A", "D")
    assert path is not None
    assert [node.id for node in path] == ["A", "B", "C", "D"]


def test_minimum_spanning_tree_weight_total():
    graph = Graph(id="mst")
    for node_id in ["A", "B", "C", "D"]:
        graph.add_node(Node(id=node_id))

    graph.add_edge(WeightedEdge(id="ab", source="A", target="B", weight=1))
    graph.add_edge(WeightedEdge(id="ac", source="A", target="C", weight=4))
    graph.add_edge(WeightedEdge(id="ad", source="A", target="D", weight=3))
    graph.add_edge(WeightedEdge(id="bc", source="B", target="C", weight=2))
    graph.add_edge(WeightedEdge(id="bd", source="B", target="D", weight=5))
    graph.add_edge(WeightedEdge(id="cd", source="C", target="D", weight=1))

    mst = minimum_spanning_tree(graph)
    assert mst.number_of_nodes() == graph.number_of_nodes()
    assert mst.number_of_edges() == graph.number_of_nodes() - 1

    total_weight = sum(getattr(edge, "weight", 1.0) for edge in mst.edges)
    assert total_weight == pytest.approx(4.0)


def test_max_flow_and_min_cut():
    graph = DirectedGraph(id="flow")
    for node_id in ["S", "A", "B", "T"]:
        graph.add_node(Node(id=node_id))

    graph.add_edge(CapacityEdge(id="sa", source="S", target="A", capacity=3))
    graph.add_edge(CapacityEdge(id="sb", source="S", target="B", capacity=2))
    graph.add_edge(CapacityEdge(id="ab", source="A", target="B", capacity=1))
    graph.add_edge(CapacityEdge(id="at", source="A", target="T", capacity=2))
    graph.add_edge(CapacityEdge(id="bt", source="B", target="T", capacity=3))

    max_flow_result = max_flow(graph, "S", "T")
    assert max_flow_result.value == pytest.approx(5.0)
    assert max_flow_result.flow[("S", "A")] == pytest.approx(3.0)
    assert max_flow_result.flow[("S", "B")] == pytest.approx(2.0)

    min_cut_result = min_cut(graph, "S", "T")
    assert min_cut_result.value == pytest.approx(max_flow_result.value)
    assert min_cut_result.reachable == {"S"}
    assert min_cut_result.non_reachable == {"A", "B", "T"}


def test_centrality_wrappers_match_networkx():
    graph = Graph(id="centrality")
    for node_id in ["A", "B", "C", "D"]:
        graph.add_node(Node(id=node_id))

    graph.add_edge(WeightedEdge(id="ab", source="A", target="B", weight=1))
    graph.add_edge(WeightedEdge(id="bc", source="B", target="C", weight=2))
    graph.add_edge(WeightedEdge(id="cd", source="C", target="D", weight=3))
    graph.add_edge(WeightedEdge(id="ad", source="A", target="D", weight=4))

    nx_graph = graph_to_networkx(graph, include_edge_attrs=True)

    expected_degree = nx.degree_centrality(nx_graph)
    expected_closeness = nx.closeness_centrality(nx_graph)
    expected_betweenness = nx.betweenness_centrality(nx_graph)
    expected_eigenvector = nx.eigenvector_centrality(nx_graph, max_iter=1000, weight=None)

    _assert_dicts_close(degree_centrality(graph), expected_degree)
    _assert_dicts_close(closeness_centrality(graph), expected_closeness)
    _assert_dicts_close(betweenness_centrality(graph), expected_betweenness)
    _assert_dicts_close(
        eigenvector_centrality(graph, weight=None, max_iter=1000),
        expected_eigenvector,
        rel=1e-5,
    )


def _assert_dicts_close(actual, expected, *, rel=1e-6, abs_tol=1e-9):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], rel=rel, abs=abs_tol)
