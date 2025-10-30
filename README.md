# graph-py
This is a project in memory python graph handling


# Algorithms included
## Pathfinding
- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- Dijkstra's algorithm
- Bellman-Ford algorithm
- A* search with custom heuristics

## All-Pairs Shortest Paths
- Floyd-Warshall algorithm

## Graph Traversal & Exploration
- all_paths() - Find all paths between two nodes
- cycles() - Detect all cycles in the graph
- strongly_connected_components() - Find SCCs using Kosaraju's algorithm
- weakly_connected_components() - Find connected components
- topological_sort() - Sort DAG nodes topologically
- reachable_nodes() - Find all reachable nodes from a start node
- is_dag() - Check if graph is a DAG
- has_cycle() - Check if graph contains cycles

## Graph Operations
- union - Combine two graphs
- intersection - Find common elements between graphs
- minimum_spanning_tree - Compute MSTs for weighted undirected graphs

## Graph Metrics
- Node/edge counts, component analysis
- Degree statistics for directed and undirected graphs
- Distance measures (diameter, radius, average shortest paths)
- Spectral radius of the adjacency matrix
- Centrality measures (degree, closeness, betweenness, eigenvector)

## Advanced Search
- TF-IDF search with configurable parameters
- BM25 ranking algorithm
- Regex-based search
- Vector search capabilities

## Network Flow
- max_flow - Edmonds-Karp implementation with residual graph output
- min_cut - Minimum s-t cut derived from max flow results

## Visualization
You can quickly convert any `Graph` into a `networkx` object or render it with matplotlib:

```python
from graph_py import Graph, Node, Edge

graph = Graph(id="demo")
graph.add_node(Node(id="A", name="Alpha"))
graph.add_node(Node(id="B", name="Beta"))
graph.add_edge(Edge(id="A-B", source="A", target="B", name="connects"))

# Convert to a networkx.Graph for further processing
nx_graph = graph.to_networkx()

# Create and save a visualization
graph.visualize(layout="kamada_kawai", edge_label_field="name", save_path="demo.png")
```

# Examples
Hands-on examples live in the `examples/` package and can be executed without installing the library globally:

```bash
python -m examples.flow_demo          # Edmonds-Karp max-flow and min-cut
python -m examples.ontology_demo      # Patient graph validated against an ICD snippet
python -m examples.visualization_demo # Network rendering (requires matplotlib)
```

Each script prints a short walkthrough and writes outputs (such as rendered images) into the `examples/output/` directory.
