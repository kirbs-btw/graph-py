from ..core import Graph, Edge, Node

class UndirectedGraph(Graph):
    """Graph representation with undirected edges."""

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Adjacency list treating edges as symmetric."""
        adj = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
            adj[e.target].append(e.source)
        return adj

    def add_edge(self, edge: Edge):
        """Add edge only if not already represented (A–B same as B–A)."""
        if not any(
            (e.source == edge.source and e.target == edge.target) or
            (e.source == edge.target and e.target == edge.source)
            for e in self.edges
        ):
            self.edges.append(edge)

    def neighbors(self, node_id: str) -> list[Node]:
        """Return all nodes connected to node_id."""
        ids = set()
        for e in self.edges:
            if e.source == node_id:
                ids.add(e.target)
            elif e.target == node_id:
                ids.add(e.source)
        return [n for n in self.nodes if n.id in ids]
