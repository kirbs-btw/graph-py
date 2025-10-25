from __future__ import annotations

from collections import defaultdict, deque
from typing import List, Optional, Set, Tuple, Union

from ..core import Graph, Node
from ..graphs import DirectedGraph


def all_paths(
    graph: Graph, 
    start: Union[Node, str], 
    end: Union[Node, str], 
    max_length: Optional[int] = None
) -> List[List[Node]]:
    """
    Find all simple paths between two nodes.
    
    Args:
        graph: The graph to search in
        start: Starting node or node ID
        end: Target node or node ID  
        max_length: Maximum path length (None for no limit)
        
    Returns:
        List of all simple paths as lists of nodes
    """
    # Convert string IDs to Node objects if needed
    if isinstance(start, str):
        start_node = graph.get_node(start)
        if start_node is None:
            raise ValueError(f"Start node '{start}' not found in graph")
    else:
        start_node = start
        
    if isinstance(end, str):
        end_node = graph.get_node(end)
        if end_node is None:
            raise ValueError(f"End node '{end}' not found in graph")
    else:
        end_node = end
    
    if start_node.id == end_node.id:
        return [[start_node]]
    
    paths = []
    visited = set()
    
    def dfs_paths(current: Node, target: Node, path: List[Node]):
        if current.id == target.id:
            paths.append(path[:])
            return
            
        if max_length and len(path) > max_length:
            return
            
        visited.add(current.id)
        for neighbor in current.neighbors:
            if neighbor.id not in visited:
                path.append(neighbor)
                dfs_paths(neighbor, target, path)
                path.pop()
        visited.remove(current.id)
    
    dfs_paths(start_node, end_node, [start_node])
    return paths


def cycles(graph: Graph) -> List[List[Node]]:
    """
    Find all simple cycles in the graph.
    
    Args:
        graph: The graph to search for cycles
        
    Returns:
        List of all simple cycles as lists of nodes
    """
    cycles_found = []
    visited = set()
    rec_stack = set()
    
    def dfs_cycle(node: Node, parent: Optional[Node], path: List[Node]):
        visited.add(node.id)
        rec_stack.add(node.id)
        path.append(node)
        
        for neighbor in node.neighbors:
            if neighbor.id not in visited:
                dfs_cycle(neighbor, node, path)
            elif neighbor.id in rec_stack and neighbor != parent:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles_found.append(cycle[:])
        
        path.pop()
        rec_stack.remove(node.id)
    
    for node in graph.nodes:
        if node.id not in visited:
            dfs_cycle(node, None, [])
    
    return cycles_found


def strongly_connected_components(graph: Graph) -> List[List[Node]]:
    """
    Find strongly connected components using Kosaraju's algorithm.
    
    Args:
        graph: The graph to analyze
        
    Returns:
        List of strongly connected components, each as a list of nodes
    """
    if not isinstance(graph, DirectedGraph):
        # For undirected graphs, SCCs are the same as connected components
        return weakly_connected_components(graph)
    
    visited = set()
    stack = []
    
    def dfs1(node: Node):
        visited.add(node.id)
        for neighbor in node.neighbors:
            if neighbor.id not in visited:
                dfs1(neighbor)
        stack.append(node)
    
    # First DFS to fill the stack
    for node in graph.nodes:
        if node.id not in visited:
            dfs1(node)
    
    # Create transpose graph
    transpose_adj = defaultdict(list)
    for edge in graph.edges:
        transpose_adj[edge.target].append(edge.source)
    
    # Second DFS on transpose graph
    visited.clear()
    sccs = []
    
    def dfs2(node: Node, component: List[Node]):
        visited.add(node.id)
        component.append(node)
        for neighbor_id in transpose_adj.get(node.id, []):
            neighbor = graph.get_node(neighbor_id)
            if neighbor and neighbor.id not in visited:
                dfs2(neighbor, component)
    
    while stack:
        node = stack.pop()
        if node.id not in visited:
            component = []
            dfs2(node, component)
            if component:
                sccs.append(component)
    
    return sccs


def weakly_connected_components(graph: Graph) -> List[List[Node]]:
    """
    Find weakly connected components (connected components for undirected graphs).
    
    Args:
        graph: The graph to analyze
        
    Returns:
        List of connected components, each as a list of nodes
    """
    visited = set()
    components = []
    
    def dfs_component(node: Node, component: List[Node]):
        visited.add(node.id)
        component.append(node)
        for neighbor in node.neighbors:
            if neighbor.id not in visited:
                dfs_component(neighbor, component)
    
    for node in graph.nodes:
        if node.id not in visited:
            component = []
            dfs_component(node, component)
            if component:
                components.append(component)
    
    return components


def topological_sort(graph: Graph) -> Optional[List[Node]]:
    """
    Perform topological sort on a directed acyclic graph (DAG).
    
    Args:
        graph: The directed graph to sort
        
    Returns:
        Topologically sorted list of nodes, or None if graph contains cycles
        
    Raises:
        ValueError: If graph is not directed
    """
    if not isinstance(graph, DirectedGraph):
        raise ValueError("Topological sort requires a directed graph")
    
    # Calculate in-degrees
    in_degree = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        in_degree[edge.target] += 1
    
    # Initialize queue with nodes having no incoming edges
    queue = deque()
    for node in graph.nodes:
        if in_degree[node.id] == 0:
            queue.append(node)
    
    result = []
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # Reduce in-degree for all neighbors
        for neighbor in current.neighbors:
            in_degree[neighbor.id] -= 1
            if in_degree[neighbor.id] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(result) != len(graph.nodes):
        return None  # Cycle detected
    
    return result


def reachable_nodes(graph: Graph, start: Union[Node, str]) -> List[Node]:
    """
    Find all nodes reachable from a starting node.
    
    Args:
        graph: The graph to search in
        start: Starting node or node ID
        
    Returns:
        List of all reachable nodes
    """
    # Convert string ID to Node object if needed
    if isinstance(start, str):
        start_node = graph.get_node(start)
        if start_node is None:
            raise ValueError(f"Start node '{start}' not found in graph")
    else:
        start_node = start
    
    visited = set()
    reachable = []
    queue = deque([start_node])
    visited.add(start_node.id)
    
    while queue:
        current = queue.popleft()
        reachable.append(current)
        
        for neighbor in current.neighbors:
            if neighbor.id not in visited:
                visited.add(neighbor.id)
                queue.append(neighbor)
    
    return reachable


def is_dag(graph: Graph) -> bool:
    """
    Check if a directed graph is a Directed Acyclic Graph (DAG).
    
    Args:
        graph: The graph to check
        
    Returns:
        True if the graph is a DAG, False otherwise
    """
    if not isinstance(graph, DirectedGraph):
        return True  # Undirected graphs are considered DAGs
    
    return topological_sort(graph) is not None


def has_cycle(graph: Graph) -> bool:
    """
    Check if the graph contains any cycles.
    
    Args:
        graph: The graph to check
        
    Returns:
        True if the graph contains cycles, False otherwise
    """
    if isinstance(graph, DirectedGraph):
        return not is_dag(graph)
    else:
        # For undirected graphs, check if any component has more edges than nodes-1
        components = weakly_connected_components(graph)
        for component in components:
            component_nodes = {node.id for node in component}
            component_edges = [edge for edge in graph.edges 
                             if edge.source in component_nodes and edge.target in component_nodes]
            if len(component_edges) >= len(component):
                return True
        return False


__all__ = [
    "all_paths",
    "cycles", 
    "strongly_connected_components",
    "weakly_connected_components",
    "topological_sort",
    "reachable_nodes",
    "is_dag",
    "has_cycle",
]
