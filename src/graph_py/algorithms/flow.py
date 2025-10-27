from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple, Union

from ..core import Graph, Node
from ..graphs import DirectedGraph
from ._utils import DEFAULT_WEIGHT, ensure_node

FlowDict = Dict[Tuple[str, str], float]
ResidualGraph = Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class MaxFlowResult:
    """Container for maximum flow output."""

    value: float
    flow: FlowDict
    residual: ResidualGraph
    source: str
    sink: str

    def min_cut(self, graph: Graph) -> "MinCutResult":
        """Derive the minimum s-t cut from this max-flow result."""
        reachable = _reachable_in_residual(self.source, self.residual)
        non_reachable = {node.id for node in graph.nodes if node.id not in reachable}
        return MinCutResult(
            value=self.value,
            reachable=reachable,
            non_reachable=non_reachable,
            flow=self.flow,
        )


@dataclass(frozen=True)
class MinCutResult:
    """Minimum cut description."""

    value: float
    reachable: Set[str]
    non_reachable: Set[str]
    flow: FlowDict


def max_flow(
    graph: Graph,
    source: Union[Node, str],
    sink: Union[Node, str],
    *,
    capacity_attr: str = "capacity",
    default_capacity: float = DEFAULT_WEIGHT,
) -> MaxFlowResult:
    """
    Compute the maximum flow between source and sink using Edmonds–Karp.

    Args:
        graph: Graph containing capacity labelled edges.
        source: Source node or identifier.
        sink: Sink node or identifier.
        capacity_attr: Edge attribute providing capacity values.
        default_capacity: Fallback capacity when attribute is missing.

    Returns:
        MaxFlowResult with total flow value, flows per edge and residual graph.
    """
    source_node = ensure_node(graph, source)
    sink_node = ensure_node(graph, sink)
    if source_node.id == sink_node.id:
        return MaxFlowResult(
            value=0.0,
            flow={},
            residual=defaultdict(dict),
            source=source_node.id,
            sink=sink_node.id,
        )

    capacity_map: FlowDict = defaultdict(float)
    residual: ResidualGraph = defaultdict(dict)
    adjacency: Dict[str, Set[str]] = defaultdict(set)

    directed = isinstance(graph, DirectedGraph)
    for edge in graph.edges:
        capacity = _edge_capacity(edge, capacity_attr=capacity_attr, default_capacity=default_capacity)
        if capacity <= 0:
            continue
        orientations = [(edge.source, edge.target)]
        if not directed:
            orientations.append((edge.target, edge.source))
        for u, v in orientations:
            capacity_map[(u, v)] += capacity
            residual[u][v] = residual[u].get(v, 0.0) + capacity
            adjacency[u].add(v)
            adjacency[v].add(u)
            residual[v].setdefault(u, 0.0)

    max_flow_value = 0.0

    while True:
        path, path_capacity = _bfs_augmenting_path(source_node.id, sink_node.id, residual, adjacency)
        if path_capacity == 0.0:
            break
        max_flow_value += path_capacity
        for u, v in path:
            residual[u][v] -= path_capacity
            residual[v][u] = residual[v].get(u, 0.0) + path_capacity

    flow: FlowDict = {}
    for (u, v), capacity in capacity_map.items():
        remaining = residual[u].get(v, 0.0)
        sent = capacity - remaining
        if sent > 1e-12:
            flow[(u, v)] = sent

    return MaxFlowResult(
        value=max_flow_value,
        flow=flow,
        residual=residual,
        source=source_node.id,
        sink=sink_node.id,
    )


def min_cut(
    graph: Graph,
    source: Union[Node, str],
    sink: Union[Node, str],
    *,
    capacity_attr: str = "capacity",
    default_capacity: float = DEFAULT_WEIGHT,
) -> MinCutResult:
    """
    Compute an s-t minimum cut by running a maximum flow computation.

    Args:
        graph: Graph to analyse.
        source: Source node or identifier.
        sink: Sink node or identifier.
        capacity_attr: Edge attribute providing capacity values.
        default_capacity: Fallback capacity when attribute is missing.
    """
    max_flow_result = max_flow(
        graph,
        source,
        sink,
        capacity_attr=capacity_attr,
        default_capacity=default_capacity,
    )
    return max_flow_result.min_cut(graph)


def _edge_capacity(edge, *, capacity_attr: str, default_capacity: float) -> float:
    value = getattr(edge, capacity_attr, None)
    if value is None and capacity_attr != "capacity":
        value = getattr(edge, "capacity", None)
    if value is None:
        value = getattr(edge, "weight", default_capacity)
    try:
        capacity = float(value)
    except (TypeError, ValueError):
        capacity = default_capacity
    return max(capacity, 0.0)


def _bfs_augmenting_path(
    source: str,
    sink: str,
    residual: ResidualGraph,
    adjacency: Dict[str, Set[str]],
) -> Tuple[Iterable[Tuple[str, str]], float]:
    parent: Dict[str, Optional[str]] = {source: None}
    queue: deque[str] = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacency.get(u, set()):
            if v in parent:
                continue
            capacity = residual[u].get(v, 0.0)
            if capacity <= 1e-12:
                continue
            parent[v] = u
            if v == sink:
                path: List[Tuple[str, str]] = []
                bottleneck = float("inf")
                current = sink
                while parent[current] is not None:
                    prev = parent[current]
                    path.append((prev, current))
                    bottleneck = min(bottleneck, residual[prev].get(current, 0.0))
                    current = prev
                path.reverse()
                return path, bottleneck if bottleneck != float("inf") else 0.0
            queue.append(v)
    return [], 0.0


def _reachable_in_residual(source: str, residual: ResidualGraph) -> Set[str]:
    visited: Set[str] = set()
    queue: deque[str] = deque([source])
    while queue:
        u = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v, capacity in residual.get(u, {}).items():
            if capacity > 1e-12 and v not in visited:
                queue.append(v)
    return visited


__all__ = ["max_flow", "min_cut", "MaxFlowResult", "MinCutResult"]
