import pytest
from graph_py import Graph, DirectedGraph, Node, Edge
from graph_py.algorithms.traversal import (
    all_paths,
    cycles,
    strongly_connected_components,
    weakly_connected_components,
    topological_sort,
    reachable_nodes,
    is_dag,
    has_cycle,
)


def test_all_paths():
    """Test finding all paths between nodes."""
    graph = Graph(id="test")
    
    # Create a simple graph: A -> B -> C
    #                   A -> C
    a = Node(id="A")
    b = Node(id="B") 
    c = Node(id="C")
    
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    graph.add_edge(Edge(id="e3", source="A", target="C"))
    
    paths = all_paths(graph, "A", "C")
    assert len(paths) == 2
    assert [node.id for node in paths[0]] == ["A", "B", "C"]
    assert [node.id for node in paths[1]] == ["A", "C"]
    
    # Test with max_length
    paths_limited = all_paths(graph, "A", "C", max_length=1)
    assert len(paths_limited) == 1
    assert [node.id for node in paths_limited[0]] == ["A", "C"]


def test_cycles():
    """Test cycle detection."""
    graph = Graph(id="test")
    
    # Create a graph with cycles: A -> B -> C -> A
    #                            B -> D -> B
    a = Node(id="A")
    b = Node(id="B")
    c = Node(id="C")
    d = Node(id="D")
    
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    graph.add_node(d)
    
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    graph.add_edge(Edge(id="e3", source="C", target="A"))
    graph.add_edge(Edge(id="e4", source="B", target="D"))
    graph.add_edge(Edge(id="e5", source="D", target="B"))
    
    found_cycles = cycles(graph)
    assert len(found_cycles) >= 1  # Should find at least one cycle
    
    # Check that cycles are valid
    for cycle in found_cycles:
        assert len(cycle) >= 3  # Minimum cycle length
        assert cycle[0].id == cycle[-1].id  # Cycle should start and end at same node


def test_strongly_connected_components():
    """Test strongly connected components."""
    graph = DirectedGraph(id="test")
    
    # Create graph: A -> B -> C -> A (SCC)
    #               D -> E (SCC)
    #               F (isolated)
    nodes = [Node(id=id) for id in ["A", "B", "C", "D", "E", "F"]]
    for node in nodes:
        graph.add_node(node)
    
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    graph.add_edge(Edge(id="e3", source="C", target="A"))
    graph.add_edge(Edge(id="e4", source="D", target="E"))
    graph.add_edge(Edge(id="e5", source="E", target="D"))  # Make D->E->D a cycle
    
    sccs = strongly_connected_components(graph)
    assert len(sccs) == 3  # Three components
    
    # Check that A, B, C are in the same SCC
    abc_scc = next((scc for scc in sccs if len(scc) == 3), None)
    assert abc_scc is not None
    scc_ids = {node.id for node in abc_scc}
    assert scc_ids == {"A", "B", "C"}


def test_weakly_connected_components():
    """Test weakly connected components."""
    graph = Graph(id="test")
    
    # Create two disconnected components
    nodes = [Node(id=id) for id in ["A", "B", "C", "D", "E"]]
    for node in nodes:
        graph.add_node(node)
    
    # Component 1: A - B - C
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    
    # Component 2: D - E
    graph.add_edge(Edge(id="e3", source="D", target="E"))
    
    wccs = weakly_connected_components(graph)
    assert len(wccs) == 2
    
    # Check component sizes
    component_sizes = [len(comp) for comp in wccs]
    assert 3 in component_sizes
    assert 2 in component_sizes


def test_topological_sort():
    """Test topological sorting."""
    graph = DirectedGraph(id="test")
    
    # Create DAG: A -> B -> D
    #             A -> C -> D
    nodes = [Node(id=id) for id in ["A", "B", "C", "D"]]
    for node in nodes:
        graph.add_node(node)
    
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="D"))
    graph.add_edge(Edge(id="e3", source="A", target="C"))
    graph.add_edge(Edge(id="e4", source="C", target="D"))
    
    topo_order = topological_sort(graph)
    assert topo_order is not None
    assert len(topo_order) == 4
    
    # Check that A comes before B and C, and B, C come before D
    node_positions = {node.id: i for i, node in enumerate(topo_order)}
    assert node_positions["A"] < node_positions["B"]
    assert node_positions["A"] < node_positions["C"]
    assert node_positions["B"] < node_positions["D"]
    assert node_positions["C"] < node_positions["D"]
    
    # Test with cycle (should return None)
    graph.add_edge(Edge(id="e5", source="D", target="A"))
    topo_order_cycle = topological_sort(graph)
    assert topo_order_cycle is None


def test_reachable_nodes():
    """Test finding reachable nodes."""
    graph = Graph(id="test")
    
    # Create graph: A - B - C
    #               D - E
    nodes = [Node(id=id) for id in ["A", "B", "C", "D", "E"]]
    for node in nodes:
        graph.add_node(node)
    
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    graph.add_edge(Edge(id="e3", source="D", target="E"))
    
    reachable_from_a = reachable_nodes(graph, "A")
    reachable_ids = {node.id for node in reachable_from_a}
    assert reachable_ids == {"A", "B", "C"}
    
    reachable_from_d = reachable_nodes(graph, "D")
    reachable_ids_d = {node.id for node in reachable_from_d}
    assert reachable_ids_d == {"D", "E"}


def test_is_dag():
    """Test DAG detection."""
    # Test DAG
    dag = DirectedGraph(id="dag")
    nodes = [Node(id=id) for id in ["A", "B", "C"]]
    for node in nodes:
        dag.add_node(node)
    
    dag.add_edge(Edge(id="e1", source="A", target="B"))
    dag.add_edge(Edge(id="e2", source="B", target="C"))
    
    assert is_dag(dag) is True
    
    # Test with cycle
    dag.add_edge(Edge(id="e3", source="C", target="A"))
    assert is_dag(dag) is False
    
    # Test undirected graph (should be True)
    undirected = Graph(id="undirected")
    for node in nodes:
        undirected.add_node(node)
    undirected.add_edge(Edge(id="e1", source="A", target="B"))
    assert is_dag(undirected) is True


def test_has_cycle():
    """Test cycle detection."""
    # Test graph without cycles
    graph = Graph(id="no_cycle")
    nodes = [Node(id=id) for id in ["A", "B", "C"]]
    for node in nodes:
        graph.add_node(node)
    
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    
    assert has_cycle(graph) is False
    
    # Test graph with cycles
    graph.add_edge(Edge(id="e3", source="C", target="A"))
    assert has_cycle(graph) is True


def test_error_handling():
    """Test error handling for invalid inputs."""
    graph = Graph(id="test")
    a = Node(id="A")
    graph.add_node(a)
    
    # Test with non-existent node
    with pytest.raises(ValueError):
        all_paths(graph, "A", "B")
    
    with pytest.raises(ValueError):
        reachable_nodes(graph, "B")
    
    # Test topological sort on undirected graph
    with pytest.raises(ValueError):
        topological_sort(graph)


if __name__ == "__main__":
    # Run a simple demonstration
    print("Testing graph traversal algorithms...")
    
    # Create a test graph
    graph = Graph(id="demo")
    
    # Add nodes
    nodes = [Node(id=id) for id in ["A", "B", "C", "D"]]
    for node in nodes:
        graph.add_node(node)
    
    # Add edges: A -> B -> C -> D
    #            A -> C
    graph.add_edge(Edge(id="e1", source="A", target="B"))
    graph.add_edge(Edge(id="e2", source="B", target="C"))
    graph.add_edge(Edge(id="e3", source="C", target="D"))
    graph.add_edge(Edge(id="e4", source="A", target="C"))
    
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Test all paths
    paths = all_paths(graph, "A", "D")
    print(f"All paths from A to D: {len(paths)} paths found")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {' -> '.join(node.id for node in path)}")
    
    # Test reachable nodes
    reachable = reachable_nodes(graph, "A")
    print(f"Nodes reachable from A: {[node.id for node in reachable]}")
    
    # Test connected components
    components = weakly_connected_components(graph)
    print(f"Connected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {[node.id for node in comp]}")
    
    print("All tests passed!")
