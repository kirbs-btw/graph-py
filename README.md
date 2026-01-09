# graph-py
An in-memory Python toolkit for building graphs, running classical algorithms, and analysing connectivity without relying on heavyweight databases.

## Installation
### From PyPI (recommended)
```bash
pip install graph-py
```

### From source
```bash
git clone https://github.com/kirbs-btw/graph-py.git
cd graph-py
pip install .
```

For local development, install editable dependencies and tooling:
```bash
pip install -e ".[dev]"  # assumes dev extras are defined
```

## Quick Start
```python
from graph_py import Graph, Node, Edge
from graph_py.algorithms.bfs import bfs
from graph_py.algorithms.dijkstra import dijkstra
from graph_py.metrics import compute_metrics

# Build a graph
graph = Graph(id="demo", name="Sample Network")
graph.add_node(Node(id="A", name="Alpha"))
graph.add_node(Node(id="B", name="Beta"))
graph.add_node(Node(id="C", name="Gamma"))
graph.add_edge(Edge(id="AB", source="A", target="B", name="fiber"))
graph.add_edge(Edge(id="BC", source="B", target="C", name="fiber"))

# Traverse (returns Node objects)
start = graph.get_node("A")
target = graph.get_node("C")
path = bfs(graph, start, target) if start and target else None
print("BFS path:", [node.id for node in path] if path else None)

# Weighted shortest path (defaults to unit weights when none supplied)
path = dijkstra(graph, "A", "C")
print("Dijkstra path:", [node.id for node in path] if path else None)

# Metrics snapshot
metrics = compute_metrics(graph)
print("Degree distribution:", metrics.degree_distribution)
```

## Persistence
Export graphs for storage or interchange:

```python
from graph_py.serialization import graph_to_json, graph_from_json, graph_to_graphml, graph_from_graphml

payload = graph_to_json(graph, path="graph.json")
restored = graph_from_json(payload)

graph_to_graphml(graph, "graph.graphml")
round_tripped = graph_from_graphml("graph.graphml")
```

More examples live in `tests/test_serialization.py:1`.

## Algorithm Overview
| Category | Primary routines | Best when | Notes |
| --- | --- | --- | --- |
| Traversal & reachability | `bfs`, `dfs`, `all_paths`, `reachable_nodes` | Exploring unweighted graphs or enumerating neighbours | BFS gives shortest hop paths; DFS suits structural discovery and cycle detection. |
| Shortest paths | `dijkstra`, `bellman_ford`, `a_star`, `floyd_warshall` | Routing with weights | Use Dijkstra for non-negative weights, Bellman-Ford when negatives exist, A* when a heuristic is available, and Floyd-Warshall for dense all-pairs analysis. |
| Connectivity & structure | `strongly_connected_components`, `weakly_connected_components`, `topological_sort`, `cycles`, `is_dag`, `has_cycle` | Understanding graph shape | Ideal for dependency graphs and to validate acyclicity before scheduling. |
| Spanning & flow | `minimum_spanning_tree`, `max_flow`, `min_cut` | Optimising cost or capacity | MST provides cheapest undirected backbones; max_flow/min_cut quantify throughput and bottlenecks. |
| Similarity & search | `RegexNodeSearch`, `TFIDFNodeSearch`, `BM25NodeSearch`, vector search helpers | Finding nodes by text or semantic relevance | Register strategies on `Graph` for flexible node lookup experiences. |
| Metrics & centrality | `compute_metrics`, `degree_centrality`, `closeness_centrality`, `betweenness_centrality`, `eigenvector_centrality` | Ranking influence and summarising graphs | Integrates with NetworkX to leverage robust numerical implementations. |
| Visualization | `graph_to_networkx`, `draw_graph`, `Graph.visualize` | Communicating structure | Requires `matplotlib`; exports as images or feeds networkx for further styling. |

## Choosing the Right Routine
- Need the fewest hops in an unweighted network? Use BFS; for weighted edges stick with Dijkstra (non-negative) or Bellman-Ford (negative edges allowed).
- Evaluating alternate routes between many pairs? `floyd_warshall` pre-computes an all-pairs matrix suitable for dense graphs with up to a few hundred nodes.
- Scheduling tasks or resolving dependencies? Validate DAG status with `is_dag`, then run `topological_sort`.
- Diagnosing possible bottlenecks? Combine `max_flow` with `.min_cut()` to surface the critical edges whose removal partitions supply and demand.
- Mapping real-world taxonomies? Use `Ontology` plus `OntologyValidator` to match graph nodes to canonical concepts and report inconsistencies.
- Prioritising important actors? Degree and betweenness centrality highlight bridge nodes, while eigenvector centrality spotlights globally influential vertices.

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

Each script prints a short walkthrough and writes outputs into the `examples/output/` directory.
