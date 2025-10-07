from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from uuid import UUID

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

class Graph(BaseModel):
    """Graph-level structure holding nodes and edges."""
    id: str
    name: Optional[str] = None
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

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


__all__ = ["Graph", "Edge", "Node"]

