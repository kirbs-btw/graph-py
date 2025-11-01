from pathlib import Path

import pytest

from graph_py import DirectedGraph, Edge, Graph
from graph_py.core import PropertyNode
from graph_py.serialization import graph_from_graphml, graph_from_json, graph_to_graphml, graph_to_json


class WeightedEdge(Edge):
    weight: float


def build_sample_graph() -> Graph:
    graph = Graph(id="warehouse", name="Warehouse Network")
    graph.add_node(PropertyNode(id="A", name="Loading Bay", properties={"floor": "A"}))
    graph.add_node(PropertyNode(id="B", name="Cold Storage", properties={"floor": "B"}))
    graph.add_node(PropertyNode(id="C", name="Dispatch", properties={"floor": "C"}))

    graph.add_edge(WeightedEdge(id="AB", source="A", target="B", weight=3.5))
    graph.add_edge(WeightedEdge(id="BC", source="B", target="C", weight=4.2))
    return graph


def test_json_roundtrip_preserves_custom_types():
    graph = build_sample_graph()

    json_payload = graph_to_json(graph)
    restored = graph_from_json(json_payload)

    assert restored.id == "warehouse"
    assert restored.name == "Warehouse Network"

    node = restored.get_node("A")
    assert isinstance(node, PropertyNode)
    assert node.properties["floor"] == "A"

    edge = restored.get_edge("AB")
    assert isinstance(edge, WeightedEdge)
    assert edge.weight == pytest.approx(3.5)


def test_graphml_roundtrip(tmp_path: Path):
    graph = DirectedGraph(id="water", name="Water Supply")
    graph.add_node(PropertyNode(id="SRC", name="Reservoir", properties={"capacity": 5000}))
    graph.add_node(PropertyNode(id="CITY", name="City Intake", properties={"demand": 3200}))
    graph.add_edge(WeightedEdge(id="pipe", source="SRC", target="CITY", weight=2.75))

    output = tmp_path / "network.graphml"
    graph_to_graphml(graph, output)

    assert output.exists()
    restored = graph_from_graphml(output)

    assert isinstance(restored, DirectedGraph)
    assert restored.id == "water"
    assert restored.name == "Water Supply"

    restored_edge = restored.get_edge("pipe")
    assert isinstance(restored_edge, WeightedEdge)
    assert restored_edge.weight == pytest.approx(2.75)

    restored_src = restored.get_node("SRC")
    assert isinstance(restored_src, PropertyNode)
    assert restored_src.properties["capacity"] == 5000
