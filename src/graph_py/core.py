from __future__ import annotations

import math
import re
from collections import Counter
from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Any, Sequence, Dict, Union, Iterable, Set, Tuple

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from uuid import UUID
    import networkx as nx
    from matplotlib.figure import Figure


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
    properties: Dict[str, Any] = Field(default_factory=dict)

    def set_property(self, key: str, value: Any) -> None:
        """Store a property value under the given key."""
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Retrieve a property value, returning default when missing."""
        return self.properties.get(key, default)


_DEFAULT_TOKEN_PATTERN = re.compile(r"\w+")


def _resolve_node_fields(node: Node, fields: Optional[Sequence[str]]) -> List[str]:
    """Determine which fields should be examined for a node search."""
    if fields is not None:
        seen: Set[str] = set()
        resolved: List[str] = []
        for field in fields:
            if field not in seen:
                resolved.append(field)
                seen.add(field)
        return resolved
    if isinstance(node, PropertyNode):
        return ["id", "name", *node.properties.keys()]
    return ["id", "name"]


def _extract_node_field(node: Node, field_name: str) -> Any:
    """Return a field value from a node or its properties."""
    if hasattr(node, field_name):
        return getattr(node, field_name)
    if isinstance(node, PropertyNode):
        return node.properties.get(field_name)
    return None


def _coerce_to_text(value: Any) -> Optional[str]:
    """Convert arbitrary field values into textual form for search."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return repr(value)
    except Exception:  # pragma: no cover - defensive
        return None


def _collect_field_texts(node: Node, fields: Optional[Sequence[str]]) -> Dict[str, str]:
    """Create a mapping of candidate field names to their textual representation."""
    texts: Dict[str, str] = {}
    for field_name in _resolve_node_fields(node, fields):
        value = _extract_node_field(node, field_name)
        text = _coerce_to_text(value)
        if text:
            texts[field_name] = text
    return texts


def _determine_token_pattern(parameters: Dict[str, Any]) -> re.Pattern[str]:
    """Compile a token pattern from search parameters or use the default."""
    pattern_value = parameters.get("token_pattern") if parameters else None
    if isinstance(pattern_value, re.Pattern):
        return pattern_value
    if isinstance(pattern_value, str):
        try:
            return re.compile(pattern_value)
        except re.error as exc:
            raise SearchError(f"Invalid token pattern '{pattern_value}': {exc}") from exc
    return _DEFAULT_TOKEN_PATTERN


def _prepare_stop_words(parameters: Dict[str, Any], *, case_sensitive: bool) -> Set[str]:
    """Extract stop words from search parameters, normalising case."""
    stop_words_value = parameters.get("stop_words") if parameters else None
    if not stop_words_value:
        return set()
    if isinstance(stop_words_value, str):
        words = stop_words_value.split()
    elif isinstance(stop_words_value, Iterable):
        words = list(stop_words_value)
    else:
        return set()
    if not case_sensitive:
        return {word.lower() for word in words if isinstance(word, str)}
    return {word for word in words if isinstance(word, str)}


def _tokenize_for_search(
    text: str,
    *,
    pattern: re.Pattern[str],
    case_sensitive: bool,
    stop_words: Set[str],
) -> List[str]:
    """Tokenize text using the provided pattern and normalise case/stop words."""
    tokens = pattern.findall(text)
    if not tokens:
        return []
    if not case_sensitive:
        tokens = [token.lower() for token in tokens]
    if stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    return tokens


def _tokenize_documents(
    nodes: Sequence[Node],
    *,
    fields: Optional[Sequence[str]],
    pattern: re.Pattern[str],
    case_sensitive: bool,
    stop_words: Set[str],
) -> List[Tuple[Node, int, Counter[str], Dict[str, str], Dict[str, Counter[str]]]]:
    """Create tokenised representations for each node."""
    documents: List[Tuple[Node, int, Counter[str], Dict[str, str], Dict[str, Counter[str]]]] = []
    for node in nodes:
        field_texts = _collect_field_texts(node, fields)
        if not field_texts:
            continue
        doc_tokens: List[str] = []
        field_token_counts: Dict[str, Counter[str]] = {}
        for field_name, text in field_texts.items():
            field_tokens = _tokenize_for_search(
                text, pattern=pattern, case_sensitive=case_sensitive, stop_words=stop_words
            )
            if not field_tokens:
                continue
            field_token_counts[field_name] = Counter(field_tokens)
            doc_tokens.extend(field_tokens)
        if not doc_tokens:
            continue
        token_counts = Counter(doc_tokens)
        documents.append((node, sum(token_counts.values()), token_counts, field_texts, field_token_counts))
    return documents


def _build_highlights(
    field_texts: Dict[str, str],
    field_token_counts: Dict[str, Counter[str]],
    relevant_terms: Iterable[str],
) -> Dict[str, str]:
    """Select field texts that contain at least one relevant term."""
    highlights: Dict[str, str] = {}
    relevant_set = set(relevant_terms)
    for field_name, token_counts in field_token_counts.items():
        if any(term in token_counts for term in relevant_set):
            highlights[field_name] = field_texts[field_name]
    return highlights

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
        candidate_fields = _resolve_node_fields(node, fields)
        highlights: Dict[str, str] = {}
        for field_name in candidate_fields:
            value = _extract_node_field(node, field_name)
            text = _coerce_to_text(value)
            if not text:
                continue
            if pattern.search(text):
                highlights[field_name] = text

        if not highlights:
            return {}
        return {"score": len(highlights), "highlights": highlights}

class TFIDFNodeSearch(NodeSearchStrategy):
    """Term-frequency inverse-document-frequency node search."""

    def __init__(self, *, target_fields: Optional[List[str]] = None, name: Optional[str] = None) -> None:
        super().__init__(name=name or "tf_idf")
        self._default_fields = target_fields

    def search(self, nodes: Sequence[Node], query: NodeSearchQuery) -> List[NodeSearchResult]:
        pattern = _determine_token_pattern(query.parameters)
        stop_words = _prepare_stop_words(query.parameters, case_sensitive=query.case_sensitive)
        query_tokens = _tokenize_for_search(
            query.pattern,
            pattern=pattern,
            case_sensitive=query.case_sensitive,
            stop_words=stop_words,
        )
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        documents = _tokenize_documents(
            nodes,
            fields=query.fields or self._default_fields,
            pattern=pattern,
            case_sensitive=query.case_sensitive,
            stop_words=stop_words,
        )
        if not documents:
            return []

        num_docs = len(documents)
        query_terms = list(query_counter.keys())
        df_counts: Dict[str, int] = {term: 0 for term in query_terms}
        for _, _, token_counts, _, _ in documents:
            for term in query_terms:
                if token_counts.get(term, 0):
                    df_counts[term] += 1

        idf: Dict[str, float] = {}
        for term, df in df_counts.items():
            if df == 0:
                continue
            idf[term] = math.log((1 + num_docs) / (1 + df)) + 1.0
        if not idf:
            return []

        query_length = sum(query_counter.values()) or 1
        query_tf = {term: query_counter[term] / query_length for term in idf.keys()}
        query_norm = math.sqrt(sum((weight * idf[term]) ** 2 for term, weight in query_tf.items())) or 1.0

        results: List[NodeSearchResult] = []
        for node, doc_length, token_counts, field_texts, field_token_counts in documents:
            doc_length = doc_length or 1
            dot_product = 0.0
            doc_norm = 0.0
            for term in idf.keys():
                term_frequency = token_counts.get(term, 0)
                if not term_frequency:
                    continue
                doc_tf = term_frequency / doc_length
                doc_weight = doc_tf * idf[term]
                query_weight = query_tf[term] * idf[term]
                dot_product += doc_weight * query_weight
                doc_norm += doc_weight * doc_weight
            if dot_product <= 0 or doc_norm <= 0:
                continue
            score = dot_product / (math.sqrt(doc_norm) * query_norm)
            highlights = _build_highlights(field_texts, field_token_counts, idf.keys())
            results.append(
                NodeSearchResult(
                    node_id=node.id,
                    score=score,
                    highlights=highlights,
                    node=node,
                )
            )

        results.sort(key=lambda item: (-item.score if item.score is not None else 0.0, item.node_id))
        if query.limit:
            results = results[: query.limit]
        return results

class BM25NodeSearch(NodeSearchStrategy):
    """Okapi BM25 ranking search strategy."""

    def __init__(
        self,
        *,
        target_fields: Optional[List[str]] = None,
        name: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        super().__init__(name=name or "bm25")
        self._default_fields = target_fields
        self.k1 = k1
        self.b = b

    def search(self, nodes: Sequence[Node], query: NodeSearchQuery) -> List[NodeSearchResult]:
        pattern = _determine_token_pattern(query.parameters)
        stop_words = _prepare_stop_words(query.parameters, case_sensitive=query.case_sensitive)
        query_tokens = _tokenize_for_search(
            query.pattern,
            pattern=pattern,
            case_sensitive=query.case_sensitive,
            stop_words=stop_words,
        )
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        documents = _tokenize_documents(
            nodes,
            fields=query.fields or self._default_fields,
            pattern=pattern,
            case_sensitive=query.case_sensitive,
            stop_words=stop_words,
        )
        if not documents:
            return []

        num_docs = len(documents)
        avg_doc_len = sum(doc_length for _, doc_length, _, _, _ in documents) / num_docs
        query_terms = list(query_counter.keys())
        df_counts: Dict[str, int] = {term: 0 for term in query_terms}
        for _, _, token_counts, _, _ in documents:
            for term in query_terms:
                if token_counts.get(term, 0):
                    df_counts[term] += 1

        idf: Dict[str, float] = {}
        for term, df in df_counts.items():
            if df == 0:
                continue
            idf[term] = math.log(1 + ((num_docs - df + 0.5) / (df + 0.5)))
        if not idf:
            return []

        results: List[NodeSearchResult] = []
        for node, doc_length, token_counts, field_texts, field_token_counts in documents:
            score = 0.0
            for term in idf.keys():
                freq = token_counts.get(term, 0)
                if not freq:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_length / (avg_doc_len or 1)))
                score += idf[term] * (numerator / denominator)
            if score <= 0:
                continue
            highlights = _build_highlights(field_texts, field_token_counts, idf.keys())
            results.append(
                NodeSearchResult(
                    node_id=node.id,
                    score=score,
                    highlights=highlights,
                    node=node,
                )
            )

        results.sort(key=lambda item: (-item.score if item.score is not None else 0.0, item.node_id))
        if query.limit:
            results = results[: query.limit]
        return results

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

    def number_of_nodes(self) -> int:
        """Return the total number of nodes stored in the graph."""
        return len(self.nodes)

    def number_of_edges(self) -> int:
        """Return the total number of edges stored in the graph."""
        return len(self.edges)

    def size(self) -> int:
        """Alias for the number of edges, matching common graph terminology."""
        return self.number_of_edges()

    def compute_metrics(self, *, include_pairwise: bool = False) -> "GraphMetrics":
        """Compute graph-level metrics such as degree distribution and distances."""
        from .metrics import compute_metrics

        return compute_metrics(self, include_pairwise=include_pairwise)

    def get_node(self, node_id: str) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return next((e for e in self.edges if e.id == edge_id), None)

    @property
    def adjacency(self) -> Dict[str, List[str]]:
        """View: adjacency list representation."""
        adj = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
            adj[e.target].append(e.source)
        return adj

    def to_networkx(
        self,
        *,
        directed: Optional[bool] = None,
        include_node_attrs: bool = True,
        include_edge_attrs: bool = True,
    ) -> "nx.Graph":
        """Create a networkx graph representation of this graph."""
        from .visualization import graph_to_networkx

        return graph_to_networkx(
            self,
            directed=directed,
            include_node_attrs=include_node_attrs,
            include_edge_attrs=include_edge_attrs,
        )

    def visualize(self, **kwargs: Any) -> "Figure":
        """Render the graph using the visualization helpers."""
        from .visualization import draw_graph

        return draw_graph(self, **kwargs)

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
