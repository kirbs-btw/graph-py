import pytest

from graph_py import Graph, Node, Edge


def test_add_node_rejects_duplicate_ids():
    graph = Graph(id="g")
    graph.add_node(Node(id="A"))

    with pytest.raises(ValueError, match="already exists"):
        graph.add_node(Node(id="A"))


def test_add_node_rejects_foreign_node():
    primary = Graph(id="primary")
    secondary = Graph(id="secondary")
    node = Node(id="N")
    primary.add_node(node)

    with pytest.raises(ValueError, match="belongs to a different graph"):
        secondary.add_node(node)


def test_add_edge_requires_existing_nodes():
    graph = Graph(id="g")
    graph.add_node(Node(id="A"))

    with pytest.raises(ValueError, match="unknown nodes"):
        graph.add_edge(Edge(id="AB", source="A", target="B"))


def test_add_edge_rejects_duplicate_ids():
    graph = Graph(id="g")
    graph.add_node(Node(id="A"))
    graph.add_node(Node(id="B"))
    graph.add_edge(Edge(id="AB", source="A", target="B"))

    with pytest.raises(ValueError, match="already exists"):
        graph.add_edge(Edge(id="AB", source="A", target="B"))


def test_remove_node_drops_incident_edges():
    graph = Graph(id="g")
    graph.add_node(Node(id="A"))
    graph.add_node(Node(id="B"))
    graph.add_node(Node(id="C"))
    graph.add_edge(Edge(id="AB", source="A", target="B"))
    graph.add_edge(Edge(id="BC", source="B", target="C"))

    removed = graph.remove_node("A")

    assert removed.id == "A"
    assert graph.get_node("A") is None
    assert all(edge.source != "A" and edge.target != "A" for edge in graph.edges)
    assert removed.graph is None


def test_remove_edge_removes_edge():
    graph = Graph(id="g")
    graph.add_node(Node(id="A"))
    graph.add_node(Node(id="B"))
    graph.add_edge(Edge(id="AB", source="A", target="B"))

    removed = graph.remove_edge("AB")

    assert removed.id == "AB"
    assert graph.get_edge("AB") is None


def test_remove_nonexistent_items_error():
    graph = Graph(id="g")
    graph.add_node(Node(id="A"))

    with pytest.raises(ValueError):
        graph.remove_node("missing")

    with pytest.raises(ValueError):
        graph.remove_edge("missing")

