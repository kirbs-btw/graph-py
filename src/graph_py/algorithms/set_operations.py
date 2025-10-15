from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Type, TypeVar

from ..core import Edge, Graph, Node

GraphT = TypeVar("GraphT", bound=Graph)


def _dump_model(model, exclude: Optional[Iterable[str]] = None) -> dict:
    """Support both Pydantic v1 and v2 model dumping."""
    exclude_set = set(exclude or [])
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude=exclude_set)  # Pydantic v2
    return model.dict(exclude=exclude_set)  # type: ignore[attr-defined]


def _clone_node(node: Node) -> Node:
    data = _dump_model(node, exclude={"graph"})
    return node.__class__(**data)


def _clone_edge(edge: Edge) -> Edge:
    data = _dump_model(edge)
    return edge.__class__(**data)


def _resolve_graph_type(graph_a: Graph, graph_b: Graph) -> Type[Graph]:
    type_a = type(graph_a)
    type_b = type(graph_b)
    if type_a is type_b:
        return type_a
    if issubclass(type_a, type_b):
        return type_a
    if issubclass(type_b, type_a):
        return type_b
    return Graph


def _build_graph(
    graph_cls: Type[GraphT],
    graph_id: str,
    graph_name: Optional[str],
    nodes: List[Node],
    edges: List[Edge],
) -> GraphT:
    graph = graph_cls(id=graph_id, name=graph_name, nodes=[], edges=[])  # type: ignore[call-arg]
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        if graph.get_node(edge.source) and graph.get_node(edge.target):
            graph.add_edge(edge)
    return graph


def graph_union(graph_a: GraphT, graph_b: GraphT) -> GraphT:
    """Create a new graph containing every node and edge from both graphs."""
    graph_cls = _resolve_graph_type(graph_a, graph_b)
    nodes_seen: Dict[str, Node] = {}
    for node in (*graph_a.nodes, *graph_b.nodes):
        if node.id not in nodes_seen:
            nodes_seen[node.id] = _clone_node(node)

    edges_seen: Dict[str, Edge] = {}
    for edge in (*graph_a.edges, *graph_b.edges):
        if edge.id not in edges_seen:
            edges_seen[edge.id] = _clone_edge(edge)

    graph_id = f"{graph_a.id}_union_{graph_b.id}"
    graph_name = f"Union({graph_a.name or graph_a.id}, {graph_b.name or graph_b.id})"
    return _build_graph(
        graph_cls,
        graph_id,
        graph_name,
        list(nodes_seen.values()),
        list(edges_seen.values()),
    )


def graph_intersection(graph_a: GraphT, graph_b: GraphT) -> GraphT:
    """Create a new graph with nodes/edges shared by both graphs."""
    graph_cls = _resolve_graph_type(graph_a, graph_b)

    ids_a = {node.id: node for node in graph_a.nodes}
    ids_b = {node.id: node for node in graph_b.nodes}
    shared_node_ids = ids_a.keys() & ids_b.keys()
    shared_nodes: List[Node] = []
    for node_id in shared_node_ids:
        node = ids_a.get(node_id) or ids_b[node_id]
        shared_nodes.append(_clone_node(node))

    edges_a: Dict[str, Edge] = {edge.id: edge for edge in graph_a.edges}
    edges_b: Dict[str, Edge] = {edge.id: edge for edge in graph_b.edges}
    shared_edge_ids = edges_a.keys() & edges_b.keys()
    shared_edges: List[Edge] = []
    for edge_id in shared_edge_ids:
        edge = edges_a.get(edge_id) or edges_b[edge_id]
        if edge.source in shared_node_ids and edge.target in shared_node_ids:
            shared_edges.append(_clone_edge(edge))

    graph_id = f"{graph_a.id}_intersection_{graph_b.id}"
    graph_name = f"Intersection({graph_a.name or graph_a.id}, {graph_b.name or graph_b.id})"
    return _build_graph(graph_cls, graph_id, graph_name, shared_nodes, shared_edges)


__all__ = ["graph_union", "graph_intersection"]
