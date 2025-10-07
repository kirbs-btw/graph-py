from ..core import Graph, Edge, Node

class DirectedGraph(Graph):
    """Graph representation with directed edges."""

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Adjacency list respecting edge direction."""
        adj = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
        return adj

    def successors(self, node_id: str) -> list[Node]:
        """Nodes reachable by outgoing edges."""
        ids = [e.target for e in self.edges if e.source == node_id]
        return [n for n in self.nodes if n.id in ids]

    def predecessors(self, node_id: str) -> list[Node]:
        """Nodes with edges incoming to node_id."""
        ids = [e.source for e in self.edges if e.target == node_id]
        return [n for n in self.nodes if n.id in ids]