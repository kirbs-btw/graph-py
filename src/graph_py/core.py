"""Core abstractions for graph_py."""

from pydantic import BaseModel
from typing import Optional

class Edge(BaseModel):
    """Base class for all edge types."""
    name: str
    id: str

class Node(BaseModel):
    """Base class for all nodes in the library"""
    id: str
    edges: Optional[list[Edge]] = []

class Graph(BaseModel):
    """
    Base class for all graphs in the library.

    The question is going: 
    graph centric, node centric or edge centric? 
    Or somehow doing a pointer to the other ones to have all the representations
    

    """
    name: Optional[str] = None
    id: str
    nodes: Optional[list[Node]] = []

__all__ = ["Graph", "Edge", "Node"]

