from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, IO, Optional, Type, TypeVar, Union

import networkx as nx

from .core import Edge, Graph, Node
from .graphs import DirectedGraph

GraphT = TypeVar("GraphT", bound=Graph)


def _model_dump(model: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility wrapper returning a dict for Pydantic v1/v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[call-arg]
    return model.dict(**kwargs)  # type: ignore[attr-defined]


def _qualified_name(cls: Type[Any]) -> str:
    return f"{cls.__module__}:{cls.__name__}"


def _resolve_class(identifier: Optional[str], base: Type[Any]) -> Type[Any]:
    if not identifier:
        return base
    try:
        module_name, class_name = identifier.split(":", 1)
        module = importlib.import_module(module_name)
        candidate = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        return base
    if isinstance(candidate, type) and issubclass(candidate, base):
        return candidate
    return base


def _graphml_friendly_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def _normalise_graphml_mapping(mapping: Dict[str, Any]) -> None:
    for key, value in list(mapping.items()):
        if key == "__type__":
            continue
        mapping[key] = _graphml_friendly_value(value)


def _restore_graphml_value(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"{", "["}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _restore_graphml_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    restored: Dict[str, Any] = {}
    for key, value in mapping.items():
        if key == "__type__":
            restored[key] = value
        else:
            restored[key] = _restore_graphml_value(value)
    return restored


def graph_to_dict(graph: Graph) -> Dict[str, Any]:
    """Serialise a graph to a JSON-compatible dictionary."""
    graph_attributes = _model_dump(graph, exclude={"nodes", "edges"}, exclude_none=True)
    if "id" not in graph_attributes:
        graph_attributes["id"] = graph.id

    payload: Dict[str, Any] = {
        "graph": {
            "type": _qualified_name(type(graph)),
            "attributes": graph_attributes,
        },
        "nodes": [],
        "edges": [],
    }

    for node in graph.nodes:
        node_payload = {
            "type": _qualified_name(type(node)),
            "attributes": _model_dump(node, exclude={"graph"}, exclude_none=True),
        }
        payload["nodes"].append(node_payload)

    for edge in graph.edges:
        edge_payload = {
            "type": _qualified_name(type(edge)),
            "attributes": _model_dump(edge, exclude_none=True),
        }
        payload["edges"].append(edge_payload)

    return payload


def graph_from_dict(data: Dict[str, Any]) -> Graph:
    """Reconstruct a graph instance from a dictionary produced by graph_to_dict."""
    if "graph" not in data:
        raise ValueError("Missing 'graph' section in serialised payload.")

    graph_section = data["graph"] or {}
    graph_attrs = dict(graph_section.get("attributes") or {})
    graph_type = _resolve_class(graph_section.get("type"), Graph)

    if "id" not in graph_attrs:
        raise ValueError("Graph attributes must include an 'id' field.")

    graph: Graph = graph_type(**graph_attrs)

    for node_entry in data.get("nodes", []):
        node_type = _resolve_class((node_entry or {}).get("type"), Node)
        node_attrs = dict((node_entry or {}).get("attributes") or {})
        if "id" not in node_attrs:
            raise ValueError("Serialised node is missing required 'id'.")
        node = node_type(**node_attrs)
        graph.add_node(node)

    for edge_entry in data.get("edges", []):
        edge_type = _resolve_class((edge_entry or {}).get("type"), Edge)
        edge_attrs = dict((edge_entry or {}).get("attributes") or {})
        if "id" not in edge_attrs or "source" not in edge_attrs or "target" not in edge_attrs:
            raise ValueError("Serialised edge must include 'id', 'source', and 'target'.")
        edge = edge_type(**edge_attrs)
        graph.add_edge(edge)

    return graph


def graph_to_json(graph: Graph, *, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
    """Serialise a graph to JSON, optionally writing to disk."""
    payload = graph_to_dict(graph)
    text = json.dumps(payload, indent=indent, sort_keys=True)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return text


def graph_from_json(source: Union[str, Path, IO[str], Dict[str, Any]]) -> Graph:
    """Load a graph from JSON text, a file path, or a dictionary."""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        data = json.loads(text)
    elif hasattr(source, "read"):
        data = json.load(source)  # type: ignore[arg-type]
    elif isinstance(source, str):
        data = json.loads(source)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError("Unsupported source type for graph_from_json.")

    if not isinstance(data, dict):
        raise ValueError("JSON payload must decode to a dictionary.")
    return graph_from_dict(data)


def graph_to_graphml(
    graph: Graph,
    path: Union[str, Path],
    *,
    directed: Optional[bool] = None,
) -> Path:
    """Export a graph to GraphML, preserving custom node and edge attributes."""
    nx_graph = graph.to_networkx(directed=directed)

    nx_graph.graph["__type__"] = _qualified_name(type(graph))
    nx_graph.graph["graph_id"] = graph.id
    if graph.name is not None:
        nx_graph.graph["name"] = graph.name
    _normalise_graphml_mapping(nx_graph.graph)

    node_lookup = {node.id: node for node in graph.nodes}
    for node_id, attributes in nx_graph.nodes(data=True):
        node_obj = node_lookup.get(node_id)
        if node_obj is None:
            continue
        attributes["__type__"] = _qualified_name(type(node_obj))
        _normalise_graphml_mapping(attributes)

    edge_lookup = {edge.id: edge for edge in graph.edges}
    for _, _, attributes in nx_graph.edges(data=True):
        edge_id = attributes.get("id")
        if edge_id and edge_id in edge_lookup:
            attributes["__type__"] = _qualified_name(type(edge_lookup[edge_id]))
        _normalise_graphml_mapping(attributes)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(nx_graph, destination)
    return destination


def graph_from_graphml(
    path: Union[str, Path],
    *,
    default_graph_cls: Type[GraphT] = Graph,  # type: ignore[assignment]
    default_directed_cls: Type[GraphT] = DirectedGraph,  # type: ignore[assignment]
) -> GraphT:
    """Load a graph from a GraphML file."""
    source = Path(path)
    nx_graph = nx.read_graphml(source)

    graph_attrs = _restore_graphml_mapping(dict(nx_graph.graph))
    graph_type = _resolve_class(graph_attrs.pop("__type__", None), DirectedGraph if nx_graph.is_directed() else Graph)
    if nx_graph.is_directed() and not issubclass(graph_type, Graph):
        # _resolve_class may return non-Graph subclass; fall back appropriately.
        graph_type = DirectedGraph
    if not issubclass(graph_type, Graph):
        graph_type = DirectedGraph if nx_graph.is_directed() else Graph

    if graph_type is Graph:
        graph_type = default_graph_cls
    elif graph_type is DirectedGraph:
        graph_type = default_directed_cls

    if not issubclass(graph_type, Graph):
        graph_type = default_directed_cls if nx_graph.is_directed() else default_graph_cls

    graph_identifier = graph_attrs.pop("graph_id", None) or graph_attrs.get("id")
    if graph_identifier:
        graph_attrs["id"] = graph_identifier
    if not graph_attrs.get("id"):
        graph_attrs["id"] = source.stem

    graph: Graph = graph_type(**graph_attrs)

    for node_id, attributes in nx_graph.nodes(data=True):
        node_attrs = _restore_graphml_mapping(dict(attributes))
        node_type = _resolve_class(node_attrs.pop("__type__", None), Node)
        node_attrs.setdefault("id", node_id)
        node = node_type(**node_attrs)
        graph.add_node(node)

    for source_id, target_id, attributes in nx_graph.edges(data=True):
        edge_attrs = _restore_graphml_mapping(dict(attributes))
        edge_type = _resolve_class(edge_attrs.pop("__type__", None), Edge)
        edge_attrs.setdefault("id", edge_attrs.get("id") or f"{source_id}->{target_id}")
        edge_attrs.setdefault("source", source_id)
        edge_attrs.setdefault("target", target_id)
        edge = edge_type(**edge_attrs)
        graph.add_edge(edge)

    return graph  # type: ignore[return-value]


__all__ = [
    "graph_from_dict",
    "graph_from_graphml",
    "graph_from_json",
    "graph_to_dict",
    "graph_to_graphml",
    "graph_to_json",
]
