from __future__ import annotations

from graph_py.algorithms import graph_intersection, graph_union
from graph_py.core import Edge, Node
from graph_py.graphs import DirectedGraph, UndirectedGraph


def _build_directed(graph_id: str, node_ids: list[str], edges: list[tuple[str, str, str]]) -> DirectedGraph:
    graph = DirectedGraph(id=graph_id)
    for node_id in node_ids:
        graph.add_node(Node(id=node_id))
    for edge_id, source, target in edges:
        graph.add_edge(Edge(id=edge_id, source=source, target=target))
    return graph


def test_graph_union_collects_all_nodes_edges():
    g1 = _build_directed("g1", ["A", "B"], [("e1", "A", "B")])
    g2 = _build_directed("g2", ["B", "C"], [("e2", "B", "C")])

    result = graph_union(g1, g2)

    assert isinstance(result, DirectedGraph)
    assert {node.id for node in result.nodes} == {"A", "B", "C"}
    assert {edge.id for edge in result.edges} == {"e1", "e2"}
    assert result.get_node("A").graph is result
    assert result.get_node("B").graph is result


def test_graph_intersection_returns_shared_structure():
    g1 = _build_directed("g1", ["A", "B", "C"], [("e1", "A", "B"), ("shared", "B", "C")])
    g2 = _build_directed("g2", ["B", "C", "D"], [("shared", "B", "C"), ("e3", "C", "D")])

    result = graph_intersection(g1, g2)

    assert isinstance(result, DirectedGraph)
    assert {node.id for node in result.nodes} == {"B", "C"}
    assert {edge.id for edge in result.edges} == {"shared"}
    assert result.get_node("B").graph is result
    assert result.get_node("C").graph is result


def test_union_with_undirected_graph_deduplicates_edges():
    g1 = UndirectedGraph(id="g1")
    g1.add_node(Node(id="A"))
    g1.add_node(Node(id="B"))
    g1.add_edge(Edge(id="e1", source="A", target="B"))

    g2 = UndirectedGraph(id="g2")
    g2.add_node(Node(id="B"))
    g2.add_node(Node(id="A"))
    g2.add_edge(Edge(id="e2", source="B", target="A"))

    result = graph_union(g1, g2)

    assert isinstance(result, UndirectedGraph)
    assert {node.id for node in result.nodes} == {"A", "B"}
    assert len(result.edges) == 1
