from ..core import Graph, Node


def bfs(graph: Graph, src_node: Node, trg_node: Node):
    """
    Breadth-first search
    """


    # checking that both nodes are in the same graph 
    assert (src_node in graph.nodes and trg_node in graph.nodes), "target and source node need to be in the same graph"

    # getting the neighborhood of a node
    # checking if the neighborhood contains the trg_node
    # if not repeat for every node in the neighbor hood and do an expainsion throught the graph
    # if yes backtrack the path
    
    # so what is missing 
    # I would like to use a get neighborhood function for a node 
    