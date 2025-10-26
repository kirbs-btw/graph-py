import pytest

matplotlib = pytest.importorskip("matplotlib")

from graph_py import DirectedGraph, Edge, Graph, Node
from graph_py.visualization import draw_graph, graph_to_networkx


matplotlib.use("Agg")


def _build_sample_graph(graph_cls=Graph):
    graph = graph_cls(id="g", name="Sample")
    graph.add_node(Node(id="n1", name="Alpha"))
    graph.add_node(Node(id="n2", name="Beta"))
    graph.add_edge(Edge(id="e1", source="n1", target="n2", name="connects"))
    return graph


def test_graph_to_networkx_preserves_attributes():
    graph = _build_sample_graph()
    nx_graph = graph_to_networkx(graph)

    assert set(nx_graph.nodes) == {"n1", "n2"}
    assert nx_graph.nodes["n1"]["name"] == "Alpha"
    assert tuple(nx_graph.edges) in ({("n1", "n2")}, {("n2", "n1")})
    assert nx_graph.nodes["n2"]["name"] == "Beta"


def test_graph_to_networkx_respects_directed_flag():
    directed = _build_sample_graph(DirectedGraph)
    nx_graph = directed.to_networkx()

    assert nx_graph.is_directed()
    assert ("n1", "n2") in nx_graph.edges
    assert ("n2", "n1") not in nx_graph.edges


def test_draw_graph_creates_image(tmp_path):
    graph = _build_sample_graph()
    output = tmp_path / "graph.png"

    fig = draw_graph(graph, save_path=output, edge_label_field="name", layout="circular")

    assert output.exists()
    assert output.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(fig)
