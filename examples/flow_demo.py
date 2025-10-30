"""Maximum flow example using graph-py's Edmonds-Karp implementation."""

from __future__ import annotations

from typing import Iterable

from graph_py import DirectedGraph, Edge, Node
from graph_py.algorithms.flow import max_flow


class CapacityEdge(Edge):
    """Edge carrying capacity metadata required for flow analysis."""

    capacity: float


def build_colorado_river_network() -> DirectedGraph:
    """
    Model a simplified section of the Colorado River water distribution.

    Capacities are approximate cubic metres per second derived from public
    reports published by water authorities. The structure highlights how
    reservoirs feed regional delivery systems.
    """
    graph = DirectedGraph(id="colorado_water", name="Colorado River Deliveries")

    junctions: Iterable[Node] = (
        Node(id="GCN", name="Glen Canyon Dam"),
        Node(id="HVR", name="Hoover Dam"),
        Node(id="LVS", name="Southern Nevada Supply"),
        Node(id="PHX", name="Central Arizona Project"),
        Node(id="LAX", name="Metropolitan Water District"),
    )
    for junction in junctions:
        graph.add_node(junction)

    canals: Iterable[CapacityEdge] = (
        CapacityEdge(id="gcn_hvr", source="GCN", target="HVR", name="Lake Mead Release", capacity=2550.0),
        CapacityEdge(id="hvr_lvs", source="HVR", target="LVS", name="Las Vegas Intake", capacity=400.0),
        CapacityEdge(id="hvr_phx", source="HVR", target="PHX", name="Arizona Allocation", capacity=1200.0),
        CapacityEdge(id="hvr_lax", source="HVR", target="LAX", name="California Aqueduct", capacity=1700.0),
        CapacityEdge(id="gcn_phx", source="GCN", target="PHX", name="CAP Pumping", capacity=900.0),
        CapacityEdge(id="phx_lax", source="PHX", target="LAX", name="Interstate Canal", capacity=600.0),
    )
    for canal in canals:
        graph.add_edge(canal)

    return graph


def main(source: str = "GCN", sink: str = "LAX") -> None:
    graph = build_colorado_river_network()
    result = max_flow(graph, source, sink)
    cut = result.min_cut(graph)

    print(f"Maximum deliverable flow from {source} to {sink}: {result.value:.1f} m^3/s")
    print("Utilised canal flows:")
    for (u, v), value in sorted(result.flow.items()):
        print(f"  {u} -> {v}: {value:.1f} m^3/s")

    print("\nMinimum cut separating supply and demand:")
    print(f"  Reachable side: {', '.join(sorted(cut.reachable))}")
    print(f"  Demand side: {', '.join(sorted(cut.non_reachable))}")


if __name__ == "__main__":
    main()
