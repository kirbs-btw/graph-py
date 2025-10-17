from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Any, Sequence, Dict, Union
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from uuid import UUID


class SearchError(RuntimeError):
    """Raised when a search execution fails."""


class UnknownStrategyError(LookupError):
    """Raised when a requested search strategy is not registered."""

class Edge(BaseModel):
    """Base class for all edge types."""
    id: str
    name: Optional[str] = None
    source: str
    target: str

class Node(BaseModel):
    """Base class for all nodes."""
    id: str
    name: Optional[str] = None
    graph: Optional[Graph] = Field(default=None, exclude=True, repr=False)

    @property
    def edges(self) -> List[Edge]:
        """View: edges connected to this node."""
        if not self.graph:
            return []
        return [e for e in self.graph.edges if e.source == self.id or e.target == self.id]

    @property
    def neighbors(self) -> List[Node]:
        """View: connected nodes."""
        if not self.graph:
            return []
        neighbor_ids = {e.target if e.source == self.id else e.source for e in self.edges}
        return [n for n in self.graph.nodes if n.id in neighbor_ids]

class PropertyNode(Node):
    """Node subclass that stores arbitrary key/value properties."""
    properties: dict[str, Any] = Field(default_factory=dict)

    def set_property(self, key: str, value: Any) -> None:
        """Store a property value under the given key."""
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Retrieve a property value, returning default when missing."""
        return self.properties.get(key, default)

class NodeSearchQuery(BaseModel):
    """Container describing a node search request."""
    pattern: str
    fields: Optional[List[str]] = None
    limit: Optional[int] = None
    case_sensitive: bool = False
    parameters: Dict[str, Any] = Field(default_factory=dict)

class NodeSearchResult(BaseModel):
    """Represents a single search match."""
    node_id: str
    score: Optional[float] = None
    highlights: Dict[str, Any] = Field(default_factory=dict)
    node: Optional[Node] = Field(default=None, exclude=True, repr=False)

    def resolve(self, graph: Graph) -> Optional[Node]:
        """Lookup the Node object for this result within the provided graph."""
        if self.node and self.node.id == self.node_id:
            return self.node
        return graph.get_node(self.node_id)

class NodeSearchStrategy(ABC):
    """Abstract base for interchangeable node search implementations."""
    name: str

    def __init__(self, *, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def search(self, nodes: Sequence[Node], query: NodeSearchQuery) -> List[NodeSearchResult]:
        """Execute the search over the provided nodes."""

class RegexNodeSearch(NodeSearchStrategy):
    """Simple regex-based node search strategy."""

    def __init__(self, *, target_fields: Optional[List[str]] = None, name: Optional[str] = None) -> None:
        super().__init__(name=name or "regex")
        self._default_fields = target_fields

    def search(self, nodes: Sequence[Node], query: NodeSearchQuery) -> List[NodeSearchResult]:
        flags = 0 if query.case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query.pattern, flags=flags)
        except re.error as exc:
            raise SearchError(f"Invalid regex pattern '{query.pattern}': {exc}") from exc

        results: List[NodeSearchResult] = []
        fields = query.fields or self._default_fields
        for node in nodes:
            matches = self._match_node(node, fields, pattern)
            if not matches:
                continue
            results.append(
                NodeSearchResult(
                    node_id=node.id,
                    score=float(matches["score"]),
                    highlights=matches["highlights"],
                    node=node,
                )
            )
            if query.limit and len(results) >= query.limit:
                break
        return results

    def _match_node(self, node: Node, fields: Optional[List[str]], pattern: re.Pattern[str]) -> Dict[str, Any]:
        """Return match metadata when pattern hits; otherwise an empty dict."""
        candidate_fields = fields or self._resolve_fields(node)
        highlights: Dict[str, str] = {}
        for field_name in candidate_fields:
            value = self._extract_field(node, field_name)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                candidate_text = str(value)
            elif isinstance(value, str):
                candidate_text = value
            else:
                candidate_text = repr(value)

            if pattern.search(candidate_text):
                highlights[field_name] = candidate_text

        if not highlights:
            return {}
        return {"score": len(highlights), "highlights": highlights}

    def _resolve_fields(self, node: Node) -> List[str]:
        if isinstance(node, PropertyNode):
            return ["id", "name", *node.properties.keys()]
        return ["id", "name"]

    def _extract_field(self, node: Node, field_name: str) -> Any:
        if hasattr(node, field_name):
            return getattr(node, field_name)
        if isinstance(node, PropertyNode):
            return node.properties.get(field_name)
        return None

class TFIDFNodeSearch(NodeSearchStrategy):
    """Placeholder for TF-IDF based node search."""

    def __init__(self) -> None:
        super().__init__(name="tf_idf")

    def search(self, nodes: Sequence[Node], query: NodeSearchQuery) -> List[NodeSearchResult]:
        raise NotImplementedError("TF-IDF search strategy not implemented.")

class BM25NodeSearch(NodeSearchStrategy):
    """Placeholder for BM25 based node search."""

    def __init__(self) -> None:
        super().__init__(name="bm25")

    def search(self, nodes: Sequence[Node], query: NodeSearchQuery) -> List[NodeSearchResult]:
        raise NotImplementedError("BM25 search strategy not implemented.")

class Graph(BaseModel):
    """Graph-level structure holding nodes and edges."""
    id: str
    name: Optional[str] = None
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    _search_strategies: Dict[str, NodeSearchStrategy] = PrivateAttr(default_factory=dict)
    _default_strategy_key: Optional[str] = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.register_search_strategy(RegexNodeSearch(), default=True)

    def add_node(self, node: Node):
        """Add a node and link it to this graph."""
        node.graph = self
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        """Add an edge and ensure referenced nodes exist."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return next((e for e in self.edges if e.id == edge_id), None)

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """View: adjacency list representation."""
        adj = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
            adj[e.target].append(e.source)
        return adj

    def register_search_strategy(self, strategy: NodeSearchStrategy, *, alias: Optional[str] = None, default: bool = False) -> None:
        """Register a search strategy for later use."""
        key = alias or strategy.name
        self._search_strategies[key] = strategy
        if default or self._default_strategy_key is None:
            self._default_strategy_key = key

    def unregister_search_strategy(self, key: str) -> None:
        """Remove a previously registered search strategy."""
        if key in self._search_strategies:
            del self._search_strategies[key]
            if self._default_strategy_key == key:
                self._default_strategy_key = next(iter(self._search_strategies), None)

    def list_search_strategies(self) -> List[str]:
        """Return the identifiers for all registered strategies."""
        return list(self._search_strategies.keys())

    def set_default_search_strategy(self, key: str) -> None:
        """Set the default strategy used when none is specified."""
        if key not in self._search_strategies:
            raise UnknownStrategyError(f"Search strategy '{key}' is not registered.")
        self._default_strategy_key = key

    def search_nodes(
        self,
        query: Union[NodeSearchQuery, str],
        *,
        strategy: Union[None, str, NodeSearchStrategy] = None,
    ) -> List[NodeSearchResult]:
        """Execute a node search using the given or default strategy."""
        normalized_query = query if isinstance(query, NodeSearchQuery) else NodeSearchQuery(pattern=query)
        strategy_impl = self._resolve_search_strategy(strategy)
        results = strategy_impl.search(self.nodes, normalized_query)
        return results

    def _resolve_search_strategy(
        self, strategy: Union[None, str, NodeSearchStrategy]
    ) -> NodeSearchStrategy:
        if isinstance(strategy, NodeSearchStrategy):
            return strategy
        if isinstance(strategy, str):
            if strategy not in self._search_strategies:
                raise UnknownStrategyError(f"Search strategy '{strategy}' is not registered.")
            return self._search_strategies[strategy]
        if self._default_strategy_key and self._default_strategy_key in self._search_strategies:
            return self._search_strategies[self._default_strategy_key]
        if self._search_strategies:
            return next(iter(self._search_strategies.values()))
        raise UnknownStrategyError("No search strategies have been registered for this graph.")


__all__ = [
    "Graph",
    "Edge",
    "Node",
    "PropertyNode",
    "NodeSearchQuery",
    "NodeSearchResult",
    "NodeSearchStrategy",
    "RegexNodeSearch",
    "TFIDFNodeSearch",
    "BM25NodeSearch",
    "SearchError",
    "UnknownStrategyError",
]

