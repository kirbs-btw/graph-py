"""Visualization example showcasing graph-py rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from graph_py import DirectedGraph, Edge, Node
from graph_py.visualization import VisualizationError, draw_graph


def build_sample_graph() -> DirectedGraph:
    """Create a small real-world inspired flight network."""
    graph = DirectedGraph(id="lufthansa_short_haul", name="Lufthansa Short-Haul Routes")

    airports: Iterable[Node] = (
        Node(id="FRA", name="Frankfurt"),
        Node(id="BER", name="Berlin Brandenburg"),
        Node(id="MUC", name="Munich"),
        Node(id="HAM", name="Hamburg"),
        Node(id="DUS", name="Duesseldorf"),
    )
    for airport in airports:
        graph.add_node(airport)

    routes: Iterable[Edge] = (
        Edge(id="LH098", source="FRA", target="BER", name="LH98"),
        Edge(id="LH112", source="FRA", target="MUC", name="LH112"),
        Edge(id="LH206", source="FRA", target="HAM", name="LH206"),
        Edge(id="LH074", source="FRA", target="DUS", name="LH74"),
        Edge(id="LH201", source="MUC", target="HAM", name="LH201"),
        Edge(id="LH202", source="HAM", target="BER", name="LH202"),
        Edge(id="LH173", source="MUC", target="BER", name="LH173"),
        Edge(id="LH181", source="DUS", target="BER", name="LH181"),
    )
    for route in routes:
        graph.add_edge(route)

    return graph


def main(output_path: Path | None = None) -> None:
    graph = build_sample_graph()
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "output" / "lufthansa_routes.png"

    try:
        figure = draw_graph(
            graph,
            layout="spring",
            edge_label_field="name",
            node_color="#1d4ed8",
            edge_color="#6b7280",
            node_size=900,
            font_size=9,
            save_path=output_path,
            show=False,
        )
    except VisualizationError as exc:
        print(f"Visualization failed: {exc}")
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        plt = None
    if plt:
        plt.close(figure)

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
