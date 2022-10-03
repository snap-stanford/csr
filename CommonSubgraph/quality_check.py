import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def line_graph(G):
    L = nx.empty_graph(0, default=G.__class__)
    
    for from_node in G.edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in G.edges(from_node[1]):
            L.add_edge(from_node, to_node, direction = "same")
        for to_node in G.edges(from_node[0]):
            if from_node != to_node :
                L.add_edge(from_node, to_node, direction = "out")
        for to_node in G.in_edges(from_node[1]):
            if from_node != to_node :
                L.add_edge(from_node, to_node, direction = "in")        
    return L


def node_match(n1, n2): 
    return n1["category"] == n2["category"]

def edge_match(e1, e2): 
    return e1["direction"] == e2["direction"]

def solution_for_task(task):
    r, g_pos, g_neg = task 
    find_common_subgraph(g_pos)

def contains_graph(graph, small_graph):
    small_graph_line = line_graph(small_graph)
    for i in small_graph.edges:
        small_graph_line.nodes()[i]["category"] = small_graph.edges[i]["category"]

    graph_line = line_graph(graph)
    for i in graph.edges:
        graph_line.nodes()[i]["category"] = graph.edges[i]["category"]

    from networkx.algorithms import isomorphism
    DiGM = isomorphism.DiGraphMatcher(graph_line, small_graph_line, node_match = node_match, edge_match = edge_match)
    return DiGM.subgraph_is_isomorphic(), DiGM
    
def find_common_subgraph(g_pos, figsize = (10,10), vis = False):
    
    graph1, pos1 = g_pos[0][0], g_pos[0][1]
    graph2, pos2 = g_pos[1][0], g_pos[1][1]
    graph3, pos3 = g_pos[2][0], g_pos[2][1]
    
    ## remove reverse edge
    for e in list(graph1.edges()):
        if graph1.get_edge_data(*e)["category"] < 0:
            graph1.remove_edge(*e)
    for e in list(graph2.edges()):
        if graph2.get_edge_data(*e)["category"] < 0:
            graph2.remove_edge(*e)
    for e in list(graph3.edges()):
        if graph3.get_edge_data(*e)["category"] < 0:
            graph3.remove_edge(*e)
         
    
    
    graph1_line = line_graph(graph1)
    for i in graph1.edges:
        graph1_line.nodes()[i]["category"] = graph1.edges[i]["category"]


    graph2_line = line_graph(graph2)
    for i in graph2.edges:
        graph2_line.nodes()[i]["category"] = graph2.edges[i]["category"]  

    graph3_line = line_graph(graph3)
    for i in graph3.edges:
        graph3_line.nodes()[i]["category"] = graph3.edges[i]["category"]     
        
      
#     print("Compute first")
    ismags2 = nx.isomorphism.ISMAGS(graph1_line, graph2_line, node_match = node_match, edge_match = edge_match)
    largest_common_subgraph = list(ismags2.largest_common_subgraph())
    
    if vis:
        plt.figure(figsize=figsize)
        vis_graph(graph1, pos1, largest_common_subgraph[0].keys())
        plt.show()

        plt.figure(figsize=figsize)
        vis_graph(graph2, pos2, largest_common_subgraph[0].values())
        plt.show()
    
#     print("Compute second")
    ismags2_2 = nx.isomorphism.ISMAGS(graph1_line.subgraph(largest_common_subgraph[0].keys()), graph3_line, node_match = node_match, edge_match = edge_match)
    largest_common_subgraph_2 = list(ismags2_2.largest_common_subgraph())
    


#     print(largest_common_subgraph, len(largest_common_subgraph), len(largest_common_subgraph[0]))
#     print(largest_common_subgraph_2, len(largest_common_subgraph_2), len(largest_common_subgraph_2[0]))
    return largest_common_subgraph, largest_common_subgraph_2

# if __name__ == "__main__":
#     correct = 0
#     for _ in tqdm(range(100)):
#         task = get_a_random_task(num_paths = range(4,6), categories = range(2, 51))
#         r, _, _ = task
#         rule_length = len(np.array(r).nonzero()[0])
#         s1, s2 = solution_for_task(task, vis = False)
#         if len(s2) == 1 and len(s2[0]) == rule_length + 1:
#             correct += 1
#             print("correct")
#         print("=============================")
#     print(correct)