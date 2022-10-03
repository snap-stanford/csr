import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor
try:
    from .quality_check import find_common_subgraph, contains_graph
except ImportError:
    from quality_check import find_common_subgraph, contains_graph

from tqdm import tqdm

def get_task(index, prefix = "", base_path = ".", hop = 2, num_neg = 100, rule_type = "single_line", use_cache = True):
    
    ## if exsits
    path = os.path.join(base_path, f"{prefix}{index}_hop={hop}_neg={num_neg}.npy")
    if use_cache and os.path.exists(path):
        task = np.load(path, allow_pickle = True)
    else:
        print("generating")
        task = get_a_random_task(rule_type, hop, num_neg = num_neg)
        np.save(path, np.array(task, dtype=object))
    return index, task
        
def save_torch_geometric(index, task, hop = 2, ignore_neg=False):
    rule_type = task[0]
    rule = task[1]
    pos_examples = task[2]
    neg_examples = task[3]
    
    
#     rule_type = "single_line"
#     rule = task[0]
#     pos_examples = task[1]
#     neg_examples = task[2]
    pos_edge_index, pos_x, pos_edge_attr, pos_edge_mask, pos_x_pos, pos_n_size, pos_e_size = [], [], [], [], [], [], []
    neg_edge_index, neg_x, neg_edge_attr, neg_edge_mask, neg_x_pos, neg_n_size, neg_e_size = [], [], [], [], [], [], []
    for g, pos, mapping in pos_examples:
        d = G_to_torch_geoemtric(True, rule_type, rule, g, pos, mapping, hop=hop)
        pos_edge_index.append(d.edge_index)
        pos_x.append(d.x)
        pos_edge_attr.append(d.edge_attr)
        pos_edge_mask.append(d.rule_mask)
        pos_x_pos.append(d.x_pos)
        pos_n_size.append(g.number_of_nodes())
        pos_e_size.append(g.number_of_edges())
    for g, pos, mapping in neg_examples:
        d = G_to_torch_geoemtric(False, rule_type, rule, g, pos, mapping, hop=hop)
        neg_edge_index.append(d.edge_index)
        neg_x.append(d.x)
        neg_edge_attr.append(d.edge_attr)
        neg_edge_mask.append(d.rule_mask)
        neg_x_pos.append(d.x_pos)
        neg_n_size.append(g.number_of_nodes())
        neg_e_size.append(g.number_of_edges())

    assert len(pos_edge_index) == 100 and len(neg_edge_index) == 100
    return torch.cat(pos_edge_index, 1), torch.cat(pos_x, 0), torch.cat(pos_edge_attr, 0), torch.cat(pos_edge_mask, 0), torch.cat(pos_x_pos, 0), pos_n_size, pos_e_size, torch.cat(neg_edge_index, 1), torch.cat(neg_x, 0), torch.cat(neg_edge_attr, 0), torch.cat(neg_edge_mask, 0), torch.cat(neg_x_pos, 0), neg_n_size, neg_e_size
    # return None, None, None, None, None, pos_n_size, pos_e_size, None, None, None, None, None, neg_n_size, neg_e_size
    
def task_to_torch_geometric(index, task, hop = 2, ignore_neg=False):
    rule_type = task[0]
    rule = task[1]
    pos_examples = task[2]
    neg_examples = task[3]
    
    
#     rule_type = "single_line"
#     rule = task[0]
#     pos_examples = task[1]
#     neg_examples = task[2]

    return index, [G_to_torch_geoemtric(True, rule_type, rule, g, pos, mapping, hop=hop) for (g, pos, mapping) in pos_examples], [G_to_torch_geoemtric(False, rule_type, rule, g, pos, mapping, hop=hop) for (g, pos, mapping) in neg_examples]    
    

def cumsum(data):
    shape = data.shape
    return torch.cumsum(data.view(-1), 0).view(*shape)

    
def dict_to_torch_geometric(index, data_dict):
    if index == 0:
        start_e = 0
        start_n = 0
    else:
        start_e = data_dict['e_size'][index-1][-1]
        start_n = data_dict['n_size'][index-1][-1]
    
    graphs = []
    for i in range(data_dict['e_size'].shape[1]):
        end_e = data_dict['e_size'][index][i]
        end_n = data_dict['n_size'][index][i]
        edge_index = data_dict['edge_index'][:, start_e:end_e]
        x = data_dict['x'][start_n:end_n]
        edge_attr = data_dict['edge_attr'][start_e:end_e]
        edge_mask = data_dict['edge_mask'][start_e:end_e]
        x_pos = data_dict['x_pos'][start_n:end_n]
        graphs.append(Data(edge_index = edge_index, x = x, edge_attr = edge_attr, rule_mask = edge_mask, x_pos = x_pos))
        start_e = end_e
        start_n = end_n

    return graphs


def G_to_torch_geoemtric(positive, rule_type, rule, G, pos, mapping, hop = 2):
    ## G is assumed to be after relabeling, so the nodes numbers are continuous from 0 to len(G.nodes())
    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()[:2].reshape(2, -1)



    x = []
    x_pos = []
    for i in range(len(G.nodes())):
        f = torch.zeros(hop*2 + 2)
        f[G.nodes()[i]["labels"][0]] = 1
        f[G.nodes()[i]["labels"][1]+hop+1] = 1
        x.append(f)
        x_pos.append(list(pos[i]))

    x = torch.stack(x)
    x_pos = torch.tensor(x_pos)
    

    edge_attr = []

    for e in edges:
        edge_attr.append(G.get_edge_data(*e)["category"] + 50)

    edge_attr = torch.tensor(edge_attr).long()
    

   
    
    # sort it
#     adj_m = SparseTensor.from_edge_index(edge_index, sparse_sizes = [x.shape[0],x.shape[0]])
#     row, col, _  = adj_m.coo()
#     edge_index = torch.stack([row,col], 0) 
    row = edge_index[0]
    col = edge_index[1]
    idx = col.new_zeros(col.numel() + 1)
    idx[1:] = row
    idx[1:] *= x.shape[0]
    idx[1:] += col
    perm = idx[1:].argsort()
    row = row[perm]
    col = col[perm]
    edge_attr = edge_attr[perm]
    edge_index = torch.stack([row,col], 0) 

    edge_index_t = edge_index.transpose(0,1)
    edge_mask = torch.zeros(edge_index_t.shape[0])
    if positive:
        if rule_type == "single_line":
            n0 = 0
            n1 = mapping[2]

            n2 = mapping[3]
            n3 = 1
            edge_mask[(edge_index_t == torch.tensor([n0, n1])).all(1)] = 1
            edge_mask[(edge_index_t == torch.tensor([n1, n0])).all(1)] = 1
            edge_mask[(edge_index_t == torch.tensor([[n1, n2]])).all(1)] = 1
            edge_mask[(edge_index_t == torch.tensor([[n2, n1]])).all(1)] = 1
            edge_mask[(edge_index_t == torch.tensor([[n2, n3]])).all(1)] = 1
            edge_mask[(edge_index_t == torch.tensor([[n3, n2]])).all(1)] = 1
            if rule[3] != 0:
                edge_mask[(edge_index_t == torch.tensor([[n0, n2]])).all(1)] = 1
                edge_mask[(edge_index_t == torch.tensor([[n2, n0]])).all(1)] = 1
            if rule[4] != 0:
                edge_mask[(edge_index_t == torch.tensor([[n1, n3]])).all(1)] = 1
                edge_mask[(edge_index_t == torch.tensor([[n3, n1]])).all(1)] = 1

        elif rule_type.startswith("multi_line"):
            num_lines = int(rule_type.split("_")[-1])
            n = 2
            for i in range(num_lines):
                n0 = 0
                n1 = -1
                n2 = -1
                
                if n in mapping: 
                    n1 = mapping[n]
                n = n+1

                if n in mapping: 
                    n2 = mapping[n]
                n = n+1
                
                n3 = 1
                if rule[0 + 6*i] != 0:
                    edge_mask[(edge_index_t == torch.tensor([n0, n1])).all(1) & (edge_attr == rule[0 + 6*i]+ 50)] = 1
                    edge_mask[(edge_index_t == torch.tensor([n1, n0])).all(1) & (edge_attr == - rule[0 + 6*i]+ 50)] = 1
                if rule[1 + 6*i] != 0:
                    edge_mask[(edge_index_t == torch.tensor([n1, n2])).all(1) & (edge_attr == rule[1 + 6*i]+ 50)] = 1
                    edge_mask[(edge_index_t == torch.tensor([n2, n1])).all(1) & (edge_attr == -rule[1 + 6*i]+ 50)] = 1
                if rule[2 + 6*i] != 0:   
                    edge_mask[(edge_index_t == torch.tensor([[n2, n3]])).all(1) & (edge_attr == rule[2 + 6*i]+ 50)] = 1
                    edge_mask[(edge_index_t == torch.tensor([[n3, n2]])).all(1) & (edge_attr == -rule[2 + 6*i]+ 50)] = 1
                if rule[3 + 6*i] != 0:    
                    edge_mask[(edge_index_t == torch.tensor([[n0, n2]])).all(1) & (edge_attr == rule[3 + 6*i]+ 50)] = 1
                    edge_mask[(edge_index_t == torch.tensor([[n2, n0]])).all(1) & (edge_attr == -rule[3 + 6*i]+ 50)] = 1
                if rule[4 + 6*i] != 0:
                    edge_mask[(edge_index_t == torch.tensor([[n1, n3]])).all(1) & (edge_attr == rule[4 + 6*i]+ 50)] = 1
                    edge_mask[(edge_index_t == torch.tensor([[n3, n1]])).all(1) & (edge_attr == -rule[4 + 6*i]+ 50)] = 1
                if rule[5 + 6*i] != 0:
                    edge_mask[(edge_index_t == torch.tensor([[n0, n3]])).all(1) & (edge_attr == rule[5 + 6*i]+ 50)] = 1
                    edge_mask[(edge_index_t == torch.tensor([[n3, n0]])).all(1) & (edge_attr == -rule[5 + 6*i]+ 50)] = 1
    # print(edge_index.shape, x.shape, edge_attr.shape, edge_mask.shape, x_pos.shape)
    data = Data(edge_index = edge_index, x = x, edge_attr = edge_attr, rule_mask = edge_mask, x_pos = x_pos)
    # print(data.edge_index.shape, data.x.shape, data.edge_attr.shape, data.rule_mask.shape, data.x_pos.shape)
    return data


def gen_graph(add_rule, pos = True, shuffle = True, rule_only = False, hop = 2, num_paths = None, expected_num_nodes = 75, categories = range(1, 51)):
    
    p =0.5
    p_d = 0.5
    
    
    # #hop neighborhoom of 0
    

    n = 2
    positions = {0:(0,0),1:(4,0)}
    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)  

    # rules
    G, n, positions= add_rule(G, n, positions, pos = pos)
    if not rule_only:
        
        if np.random.random() < 0.25 and not G.has_edge(0,1):
            category = np.random.choice(categories)
            G.add_edge(0,1, category = category)   
            G.add_edge(1,0, category = -category) 
            
        if num_paths is not None:    
            ## previous sampling
            for i in range(np.random.choice(num_paths)):
                start_node = 0
                for j in range(hop):
                    if np.random.random() < p and n != 2:
                        node = np.random.choice(range(2, n))
                    else:
                        node = n
                        G.add_node(node)
                        positions[node] = (j+2, i + 1)
                        n = n + 1

                    category = np.random.choice(categories)
                    if (start_node, node) not in G.edges and (node, start_node) not in G.edges:
                        if np.random.random() < p_d:
                            G.add_edge(start_node, node, category=category)
                            G.add_edge(node, start_node, category=-category)
                        else:
                            G.add_edge(node, start_node, category=category)
                            G.add_edge(start_node, node, category=-category)
                    start_node = node


            # #hop neighborhoom of 1
            for i in range(np.random.choice(num_paths)):
                start_node = 1
                for j in range(hop):
                    # no self loop
    #                 node = np.random.choice(range(2, n))
                    node =np.random.choice([x for x in range(2, n) if x != start_node])
                    category = np.random.choice(categories)
                    if (start_node, node) not in G.edges and (node, start_node) not in G.edges:
                        if np.random.random() < p_d:
                            G.add_edge(start_node, node, category=category)
                            G.add_edge(node, start_node, category=-category)
                        else:
                            G.add_edge(node, start_node, category=category)
                            G.add_edge(start_node, node, category=-category)
                    start_node = node
        else:
            num_nodes_middle = max(expected_num_nodes + int(np.random.normal(0, 10)), 0)
            G.add_nodes_from(range(2, num_nodes_middle + 2))
            middle_graph = nx.fast_gnp_random_graph(num_nodes_middle, p = 0.1)
            for e in middle_graph.edges():
                if e[0] == e[1] or (e[0]+2, e[1]+2) in G.edges():
                    continue
                category = np.random.choice(categories)
                if np.random.random() < 0.5:
                    G.add_edge(e[0]+2, e[1]+2, category=category)
                    G.add_edge(e[1]+2, e[0]+2, category=-category)
                else:
                    G.add_edge(e[0]+2, e[1]+2, category=-category)
                    G.add_edge(e[1]+2, e[0]+2, category=category)
            
            side1 = 0
            side2 = 0
            side3 = 0
            for end_node in [0,1]:
                for i in range(n, num_nodes_middle+2):
                    if np.random.random() < 0.1:
                        category = np.random.choice(categories)
                        if np.random.random() < 0.5:
                            G.add_edge(end_node, i, category=category)
                            G.add_edge(i, end_node, category=-category)
                        else:
                            G.add_edge(end_node, i, category=-category)
                            G.add_edge(i, end_node, category=category)
                        
                        if i not in positions:
                            if end_node == 0:
                                positions[i] = (1, side1*5 + 1)
                                side1 += 1
                            else:
                                positions[i] = (3, side3*5 + 1)
                                side3 += 1
                    else:
                        if end_node == 1 and i not in positions:
                            positions[i] = (2, side2 + 1)
                            side2 += 1
                
#         print("=========")
    bad_nodes = [-1]
    while len(bad_nodes) > 0:
#         print(G.nodes)
#         print(G.edges)
        length_0 = nx.single_source_shortest_path_length(G.to_undirected(), 0)
        length_1 = nx.single_source_shortest_path_length(G.to_undirected(), 1)
#         print(length_0)
#         print(length_1)
        bad_nodes = []
        for i in G.nodes:
            if i!= 0 and i != 1 and (i not in length_0 or i not in length_1  or length_0[i]> hop or length_1[i] > hop):
                bad_nodes.append(i)
                
#         print(bad_nodes)
        for i in bad_nodes: 
#             print("remove node", i)
            G.remove_node(i)
#         print("===,======")

    length_0 = nx.single_source_shortest_path_length(G.to_undirected(), 0)
    length_1 = nx.single_source_shortest_path_length(G.to_undirected(), 1) 
#     print(G.nodes)
#     print(length_0)
#     print(length_1)
    for i in G.nodes:
        if i == 0:
            G.nodes[i]["labels"] = [0, 1] 
        elif i == 1:
            G.nodes[i]["labels"] = [1, 0] 
        else:
            G.nodes[i]["labels"] = [length_0[i], length_1[i]] 
        
        
    # permute nodes
    if shuffle:
        arr = np.arange(2, len(G.nodes))
        np.random.shuffle(arr)

        mapping = {list(G.nodes)[i]:arr[i-2] for i in range(2, len(G.nodes))}
        H = nx.relabel_nodes(G, mapping)
        new_positions = {0: (0,0), 1: (4,0)}
        for k, v in mapping.items():
            new_positions[v] = positions[k]    
        return H, new_positions, mapping
    return G, positions, None


def get_a_random_task(rule_type = "single_line", hop = 2, num_neg = 100, num_nodes = range(10, 150), categories = range(1, 51) ):
    rule, add_rule = get_a_random_rule(rule_type, categories = categories)
    expected_num_nodes = np.random.choice(num_nodes)
    # rule graph
    G, pos, _ = gen_graph(add_rule, pos = True, shuffle = False, rule_only = True, hop = 2, num_paths = None, categories = categories)
    # vis_graph(G, pos)
    rule_graph = G

    
    # generate postive examples
    n = num_neg
    while True:
        pos_examples = []
        for _ in range(3):
            G, pos, mapping = gen_graph(add_rule, pos = True, shuffle = True, rule_only = False, hop = 2, num_paths = None, expected_num_nodes = expected_num_nodes, categories = categories)
            # vis_graph(G, pos)
            assert contains_rule(G, rule_graph)[0]
            pos_examples.append((G, pos, mapping))
#         print("check common subgraph")
        if is_valid_support_2(pos_examples, rule_graph):
#             print("success")
            break
#         else:
#             print("fail")
#     print("--------------")
        
    for _ in range(n-3):
        G, pos, mapping = gen_graph(add_rule, pos = True, shuffle = True, rule_only = False, hop = 2, num_paths = None, expected_num_nodes = expected_num_nodes,categories = categories)
        # vis_graph(G, pos)
        assert contains_rule(G, rule_graph)[0]
        pos_examples.append((G, pos, mapping))
    

    # generate negative examples
    incorrect_sample = 0
    n = num_neg
    neg_examples = []
    while incorrect_sample < n:
        G, pos, mapping = gen_graph(add_rule, pos = False, shuffle = True, rule_only = False, hop = 2, num_paths = None, expected_num_nodes = expected_num_nodes, categories = categories)
        # vis_graph(G, pos)
        if not contains_rule(G, rule_graph)[0]:
            incorrect_sample += 1
            neg_examples.append((G, pos, mapping))

    return rule_type, rule, pos_examples, neg_examples
    
def construct_diverse_rules(num_lines, categories):
    corrects = []
    has_single = False
    types = 6
    for i in range(num_lines):
        while True:
            p = np.random.random()
            sample = np.random.choice(list( categories) + list([-n for n in categories]) , 6)
#             if p < 1./types :
#                 ### one edge
#                 assert not has_single
#                 correct[:5] = 0
            if p < 1./types:    
                ### two path
                patterns = np.array([
                    [1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                ])
            elif p < 2./types:        
                ### three path
                patterns = np.array([
                    [1, 1, 1, 0,0, 0],
                    [0, 1, 0, 1, 1, 0],
                ])
            elif p < 3./types:            
                ### three edges 
                patterns = np.array([
                    # triangle
                    [1, 0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 1],
                    # and single out?
                    [1, 1, 0, 0, 1, 0],
                    [0, 1, 1, 1, 0, 0],
                ])
            elif p < 4./types:        
                ### 4 edges
                patterns = np.array([
                    # triangle + 1
                    [1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 0, 1],
                    # one middle
                    [1, 1, 1, 0, 1, 0],
                    [1, 1, 1, 1, 0, 0],
                    # double path
                    [1, 1, 1, 0, 0, 1],
                    [1, 0, 1, 1, 1, 0],
                ])

            elif p < 5./types:             
                ### 4 clique - 1
                patterns = np.array([[1, 1, 1, 1, 1, 1]])
                r = np.random.randint(6)
                patterns[0][r] = 0
           
            else:        
                ### 4 clique
                patterns = np.array([[1, 1, 1, 1, 1, 1]])
            
            pattern = patterns[np.random.choice(range(patterns.shape[0]))]
            correct = pattern * sample
            curr_has_single = (correct[5] != 0)
            if curr_has_single:
                if  has_single == False:
                    has_single = True
                break
            else:
                break
            
        corrects.append(correct)
    correct = np.concatenate(corrects)
    return correct
    
def get_a_random_rule(rule_type = "single_line", categories = range(1, 51)):
    if rule_type == "single_line":
        correct_3 = np.random.choice(list( categories) + list([-n for n in categories]) , 3)
        correct_2 = np.random.choice([0]+ list( categories) + list([-n for n in categories]) , 2)
        correct = np.concatenate([correct_3, correct_2])

        def add_rule(G, n, positions, pos):
            return single_line_rule_3edges(G=G, n = n, correct = correct, positions = positions, pos = pos, categories = categories)
        return correct, add_rule
    elif rule_type.startswith("multi_line"):
        num_lines = int(rule_type.split("_")[-1])
        correct = construct_diverse_rules(num_lines, categories)
#         print(correct)
        def add_rule(G, n, positions, pos):
            return multiple_line_rule(G=G, n = n, correct = correct, num_lines  =num_lines,  positions = positions, pos = pos, categories = categories)
        return correct, add_rule        
    else:
        raise "Not implemented"

def single_line_rule_3edges(G, n, positions, correct = [2, 3, 4, 5, 6], pos = True, categories = range(1, 51)):
    G.add_node(n)
    
    
    if not pos:
        incorrect = correct
        while (correct == incorrect).all():
            incorrect_3 = np.random.choice(list( categories) + list([-n for n in categories]) , 3)
            incorrect_2 = np.random.choice([0]+ list( categories) + list([-n for n in categories]) , 2)
            incorrect = np.concatenate([incorrect_3, incorrect_2])
            p = np.random.random()
            if p < 0.3:
                incorrect[0] = correct[0]
                incorrect[3] = correct[3]
            elif p > 0.7:
                incorrect[2] = correct[2]
                incorrect[4] = correct[4]
            if np.random.random() > 0.5:
                incorrect[1] = correct[1]
            


        rule_relations = incorrect
    else:
        rule_relations = correct
        
    
    positions[n] = (2, -1)
    n = n + 1

    G.add_edge(0, n-1, category = rule_relations[0])
    G.add_edge( n-1, 0,  category = - rule_relations[0])   

    G.add_node(n)
    positions[n] = (3, -1)
    n = n + 1

    G.add_edge(n - 2, n-1, category = rule_relations[1])   
    G.add_edge(n - 1, n-2, category = -rule_relations[1])    

    G.add_edge(n-1, 1, category= rule_relations[2])   
    G.add_edge(1, n-1, category = - rule_relations[2])   

    if rule_relations[3] != 0:
        G.add_edge(0, n-1, category = rule_relations[3])   
        G.add_edge(n - 1, 0, category = -rule_relations[3] )      
    
    if rule_relations[4] != 0:
        G.add_edge(n - 2, 1, category = rule_relations[4])   
        G.add_edge(1, n - 2, category = -rule_relations[4])       
    return G, n, positions

def multiple_line_rule(G, n, positions, correct, num_lines = 2, pos = True, categories = range(1, 51)):

    if not pos:
        incorrect = correct
        while (correct == incorrect).all():
            incorrect = construct_diverse_rules(num_lines, categories)        
            # make incorrect harder for this correct
            for i in range(num_lines):
                p = np.random.random()

                if p < 0.3:
                    incorrect[0 + i*6 ] = correct[0+ i*6 ]
                    incorrect[3 + i*6 ] = correct[3+ i*6 ]
                    
                elif p > 0.7:
                    incorrect[2 + i*6 ] = correct[2+ i*6 ]
                    incorrect[4 + i*6 ] = correct[4+ i*6 ]            


        rule_relations = incorrect
    else:
        rule_relations = correct
        
#     print(rule_relations )
    for i in range(num_lines):
        G.add_node(n)
        positions[n] = (2, -1 - i)
        n = n + 1
        
        G.add_node(n)
        positions[n] = (3, -1 - i)
        n = n + 1

        if rule_relations[0+ i*5 ] != 0:
            G.add_edge(0, n-2, category = rule_relations[0+ i*5 ])
            G.add_edge( n-2, 0,  category = - rule_relations[0+ i*5 ])   
            
        if rule_relations[1+ i*5 ] != 0:   
            G.add_edge( n-2, n-1, category = rule_relations[1+ i*5 ])
            G.add_edge( n-1, n-2,  category = - rule_relations[1+ i*5 ])   

        if rule_relations[2+ i*5 ] != 0:
            G.add_edge(n-1, 1, category= rule_relations[2+ i*5 ])   
            G.add_edge(1, n-1, category = - rule_relations[2+ i*5 ])   
            
        if rule_relations[3+ i*5 ] != 0:
            G.add_edge(0, n-1, category= rule_relations[3+ i*5 ])   
            G.add_edge(n-1, 0, category = - rule_relations[3+ i*5 ])   
            
        if rule_relations[4+ i*5 ] != 0:   
            G.add_edge(n - 2, 1, category = rule_relations[4+ i*5 ])   
            G.add_edge(1, n - 2, category = -rule_relations[4+ i*5 ]) 
            
        if rule_relations[5+ i*5 ] != 0:   
            G.add_edge(0, 1, category = rule_relations[5+ i*5 ])   
            G.add_edge(1, 0, category = -rule_relations[5+ i*5 ]) 
                        
        
    return G, n, positions

def vis_graph(graphs, batch_n = 0, edge_mask = None, hd_off_set = 30):
    g1 = graphs.to_data_list()[batch_n]
    edge_mask_1 = None
    if edge_mask is not None:
        row, col = graphs.edge_index
        edge_mask_1 = edge_mask[graphs.batch[row] == batch_n]
    vis_single_graph(g1, edge_mask_1, hd_off_set = hd_off_set)

def vis_single_graph(g1, edge_mask_1 = None, hd_off_set = 30):
    G = torch_geometric.utils.to_networkx(g1)
    edges_idx = [ i for i in range(g1.edge_index.shape[1]) if g1.edge_attr[i] > 50]
    edges = [ tuple(g1.edge_index[:,i].tolist()) for i in edges_idx]
    edges_types = g1.edge_attr[edges_idx] 
    colors = ((edges_types - 50 ) / 50).tolist()
    pos = dict(zip(range(g1.x.shape[0]), g1.x_pos.tolist()))
    pos[0] = (0, hd_off_set)
    pos[1] = (4, hd_off_set)
    if edge_mask_1 is not None:
        colors = edge_mask_1[edges_idx].tolist()
        edge_labels = {tuple(g1.edge_index[:,i].tolist()): str(g1.edge_attr[i].item())  for i in edges_idx if edge_mask_1[i] > 0.8}

        nx.draw_networkx(G,pos =pos ,  node_size = 10, with_labels = False, edgelist = edges, 
                         edge_color=colors, edge_vmin = 0,edge_vmax = 1, edge_cmap=plt.get_cmap("OrRd"))
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels = edge_labels, label_pos=0.75)
    else:
        nx.draw_networkx(G,pos = pos,  node_size = 10, with_labels = False, edgelist = edges, 
                         edge_color=colors, edge_vmin = 0,edge_vmax = 1, edge_cmap=plt.get_cmap("rainbow"))

def contains_rule(graph, rule_graph):
    return contains_graph(graph, rule_graph)


def is_valid_support(pos_examples, rule):
    rule_length = len(np.array(rule).nonzero()[0])
    s1, s2 = find_common_subgraph(pos_examples)
    return len(s2) == 1 and len(s2[0]) == rule_length


def not_same_edges_in_subgraphs(graph1, graph2, nodes1, nodes2,  min_not_same_edges = 100000000):
    not_same_edges = 0
    correspondense = {}
    for i in range(len(nodes1)):
        for j in range(i+1, len(nodes1)):
            if not_same_edges > min_not_same_edges:
                return {}, not_same_edges
            try:
                if not (graph1.get_edge_data(nodes1[i], nodes1[j])["category"] == graph2.get_edge_data(nodes2[i], nodes2[j])["category"]):
                    not_same_edges += 1
                else:
                    correspondense[(nodes1[i], nodes1[j])] = (nodes2[i], nodes2[j])
            except Exception as e:
                not_same_edges += 1

    return correspondense, not_same_edges
    


def find_common_2_subgraph(graph1, graph2, min_not_same_edges = 100000000):

    found = {}
    found[min_not_same_edges] = []
    
    all_nodes1 = []
    all_nodes2 = []

    for i in range(2, len(graph1.nodes())):
        for j in range(i+1, len(graph1.nodes())):
#             for k in range(j+1, len(graph1.nodes())):
            all_nodes1.append([0,i,j,1])
            
    for i in range(2, len(graph2.nodes())):
        for j in range(2, len(graph2.nodes())):
#             for k in range(j+1, len(graph1.nodes())):
            if i == j:
                continue
            all_nodes2.append([0,i,j,1])
                        
    for nodes1 in all_nodes1:
        for nodes2 in all_nodes2:
            correspondense, not_same_edges = not_same_edges_in_subgraphs(graph1, graph2, nodes1, nodes2, min_not_same_edges)
            if min_not_same_edges > not_same_edges:
                min_not_same_edges = not_same_edges
                found[min_not_same_edges] = [] 
            if min_not_same_edges == not_same_edges: 
                if correspondense not in found[min_not_same_edges]:
                    found[min_not_same_edges].append(correspondense)
            
    return found, 6 - min_not_same_edges 


def is_valid_support_2(pos_examples, rule_graph):
    graph1 = pos_examples[0][0]
    graph2 = pos_examples[1][0]
    graph3 = pos_examples[2][0]

    csg1, max_num1 = find_common_2_subgraph(pos_examples[0][0], pos_examples[1][0], 6 - len(rule_graph.edges())//2 + 1)
    csg2, max_num2 =  find_common_2_subgraph(pos_examples[0][0], pos_examples[2][0], 6 - len(rule_graph.edges())//2 + 1)
    rule_size = len(rule_graph.edges())//2
    max_num = min(max_num1, max_num2)
    for s in range(rule_size+1, max_num + 1):
        if s in csg1 and s in csg2:
            for d in csg1[s]:
                if d in csg2[s]:
                    return False
    return True
    

if __name__ == "__main__":
    for _ in tqdm(range(10)):
        get_a_random_task(rule_type = "multi_lines_1", hop=2, num_neg = 0)
#     task_to_torch_geometric(*get_task(0))
#     task_to_torch_geometric(*get_task(1))