from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union, TYPE_CHECKING, Callable

import networkx as nx

from .core import Edge, Graph, Node, PropertyNode
from .graphs import DirectedGraph
from ._compat import model_dump

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure


class VisualizationError(RuntimeError):
    """Raised when graph visualization or rendering fails."""


LayoutFunc = Callable[[nx.Graph], Dict[str, Tuple[float, float]]]

_LAYOUT_FUNCTIONS: Dict[str, LayoutFunc] = {
    "spring": nx.spring_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout,
    "shell": nx.shell_layout,
    "spectral": nx.spectral_layout,
}


def graph_to_networkx(
    graph: Graph,
    *,
    directed: Optional[bool] = None,
    include_node_attrs: bool = True,
    include_edge_attrs: bool = True,
) -> nx.Graph:
    """Convert a Graph instance to a networkx graph."""
    is_directed = isinstance(graph, DirectedGraph) if directed is None else directed
    nx_graph: nx.Graph = nx.DiGraph() if is_directed else nx.Graph()

    for node in graph.nodes:
        attributes = model_dump(node, exclude={"graph"}, exclude_none=True) if include_node_attrs else {}
        nx_graph.add_node(node.id, **attributes)

    for edge in graph.edges:
        attributes = model_dump(edge, exclude={"source", "target"}, exclude_none=True) if include_edge_attrs else {}
        nx_graph.add_edge(edge.source, edge.target, **attributes)

    return nx_graph


def draw_graph(
    graph: Graph,
    *,
    directed: Optional[bool] = None,
    layout: str = "spring",
    node_color: Union[str, Iterable[str]] = "#4C78A8",
    node_size: Union[int, Iterable[int]] = 700,
    edge_color: Union[str, Iterable[str]] = "#9CA3AF",
    edge_width: float = 1.3,
    edge_alpha: float = 0.9,
    with_labels: bool = True,
    label_field: str = "name",
    font_size: int = 10,
    font_color: str = "#0F172A",
    edge_label_field: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    backend: Optional[str] = None,
) -> "Figure":
    """Render the graph using matplotlib and return the created figure."""
    nx_graph = graph_to_networkx(graph, directed=directed)
    layout_key = layout.lower()
    if layout_key not in _LAYOUT_FUNCTIONS:
        available = ", ".join(sorted(_LAYOUT_FUNCTIONS))
        raise VisualizationError(f"Unknown layout '{layout}'. Available layouts: {available}.")
    positions = _LAYOUT_FUNCTIONS[layout_key](nx_graph)

    matplotlib_backend = backend or ("Agg" if not show else None)
    plt = _load_matplotlib(matplotlib_backend)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    is_directed = nx_graph.is_directed()
    nx.draw_networkx_nodes(nx_graph, positions, ax=ax, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(
        nx_graph,
        positions,
        ax=ax,
        edge_color=edge_color,
        width=edge_width,
        alpha=edge_alpha,
        arrows=is_directed,
        arrowstyle="-|>" if is_directed else "-",
    )

    if with_labels:
        labels = _build_node_labels(graph, label_field)
        nx.draw_networkx_labels(nx_graph, positions, labels=labels, ax=ax, font_size=font_size, font_color=font_color)

    if edge_label_field:
        edge_labels = _build_edge_labels(graph, edge_label_field)
        if edge_labels:
            nx.draw_networkx_edge_labels(nx_graph, positions, edge_labels=edge_labels, ax=ax, font_size=font_size - 1)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _load_matplotlib(preferred_backend: Optional[str]):
    try:
        import matplotlib

        if preferred_backend and "matplotlib.pyplot" not in sys.modules:
            matplotlib.use(preferred_backend)

        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise VisualizationError(
            "matplotlib is required for graph rendering. Install it with 'pip install matplotlib'."
        ) from exc

    return plt


def _build_node_labels(graph: Graph, label_field: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for node in graph.nodes:
        label_value = _extract_node_label(node, label_field)
        labels[node.id] = label_value if label_value is not None else node.id
    return labels


def _build_edge_labels(graph: Graph, label_field: str) -> Dict[Tuple[str, str], str]:
    labels: Dict[Tuple[str, str], str] = {}
    for edge in graph.edges:
        value = getattr(edge, label_field, None)
        if value is None:
            continue
        labels[(edge.source, edge.target)] = str(value)
    return labels


def _extract_node_label(node: Node, label_field: str) -> Optional[str]:
    if hasattr(node, label_field):
        return getattr(node, label_field)
    if isinstance(node, PropertyNode):
        value = node.properties.get(label_field)
        if value is not None:
            return str(value)
    return None


__all__ = ["VisualizationError", "graph_to_networkx", "draw_graph"]
