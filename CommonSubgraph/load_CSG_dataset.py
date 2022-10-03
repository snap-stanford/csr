import os
import re
import torch
import sys

try:
    from .common_subgraph import get_task, task_to_torch_geometric, save_torch_geometric, dict_to_torch_geometric, cumsum
except ImportError:
    from common_subgraph import get_task, task_to_torch_geometric, save_torch_geometric, dict_to_torch_geometric, cumsum
import numpy as np 
import os
from torch.utils.data import random_split, Subset, DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
import multiprocessing as mp
from tqdm import tqdm
import time

class CSGDataset(Dataset):
    def __init__(self, root, dataset = "CommonSubgraph/data51_diverse_multi_1_10k", add_traspose_rels=False, shot = 1, n_query = 3, hop = 2, mode='train', generation = False, rule_type = "multi_line_1", preprocess=False, num_examples=100):
        self.few = shot
        self.nq = n_query
        self.mode = mode
        self.graph = None
        self.num_nodes_bg = 1000 # fake
        if self.mode == "train":
            # self.num_rels = 10
            self.num_rels = 10000
        else:
            self.num_rels = 50     
            self.curr_tri_idx = 0
        self.raw_data_paths = os.path.join(root, dataset)
        self.use_cache = True    
        self.num_rels_bg = 101
        self.hop = hop
        self.max_n_label = np.array([hop , hop ])
        self.t_torch, self.t_others = 0, 0
        self.num_examples = num_examples
        # self.tasks = [get_task(index, prefix = self.mode, base_path = self.raw_data_paths, use_cache = self.use_cache, hop = self.hop) for index in tqdm(range(self.num_rels))]
        
#         ## pre-generation
        if generation:
            if mode == 'train':
                print(int(sys.argv[1]),int(sys.argv[2]))
                for index in tqdm(range(int(sys.argv[1]),int(sys.argv[2]))):
                    get_task(index, self.mode, base_path = self.raw_data_paths, hop = hop, rule_type = rule_type, use_cache = False)        
            else:
                for index in tqdm(range(self.num_rels)):
                    get_task(index, self.mode, base_path = self.raw_data_paths, hop = hop, rule_type = rule_type, use_cache = False)        
        save_path = os.path.join(root, f"{dataset}_preprocessed")  
        if preprocess:
            self._preprocess(save_path)
        else:
            self.pos_dict = torch.load(os.path.join(save_path, "pos-%s.pt" % self.mode))
            self.neg_dict = torch.load(os.path.join(save_path, "neg-%s.pt" % self.mode))
        
    def __len__(self):
        return self.num_rels
    
    def _preprocess(self, save_path):
        print("start preprocessing %s" % self.mode)
        all_pos_edge_index, all_pos_x, all_pos_edge_attr, all_pos_edge_mask, all_pos_x_pos, all_pos_n_size, all_pos_e_size = [], [], [], [], [], [], []
        all_neg_edge_index, all_neg_x, all_neg_edge_attr, all_neg_edge_mask, all_neg_x_pos, all_neg_n_size, all_neg_e_size = [], [], [], [], [], [], []
        for index in tqdm(range(self.num_rels)):
            pos_edge_index, pos_x, pos_edge_attr, pos_edge_mask, pos_x_pos, pos_n_size, pos_e_size, neg_edge_index, neg_x, neg_edge_attr, neg_edge_mask, neg_x_pos, neg_n_size, neg_e_size = save_torch_geometric(*get_task(index, prefix = self.mode, base_path = self.raw_data_paths, use_cache = self.use_cache, hop = self.hop))
            all_pos_edge_index.append(pos_edge_index)
            all_pos_x.append(pos_x)
            all_pos_edge_attr.append(pos_edge_attr)
            all_pos_edge_mask.append(pos_edge_mask)
            all_pos_x_pos.append(pos_x_pos)
            all_pos_n_size.append(pos_n_size)
            all_pos_e_size.append(pos_e_size)
            
            all_neg_edge_index.append(neg_edge_index)
            all_neg_x.append(neg_x)
            all_neg_edge_attr.append(neg_edge_attr)
            all_neg_edge_mask.append(neg_edge_mask)
            all_neg_x_pos.append(neg_x_pos)
            all_neg_n_size.append(neg_n_size)
            all_neg_e_size.append(neg_e_size)

        print("concat all")
        all_pos_edge_index = torch.cat(all_pos_edge_index, 1)
        all_pos_x = torch.cat(all_pos_x, 0)            
        all_pos_edge_attr = torch.cat(all_pos_edge_attr, 0)
        all_pos_edge_mask = torch.cat(all_pos_edge_mask, 0)
        all_pos_x_pos = torch.cat(all_pos_x_pos, 0)

        all_neg_edge_index = torch.cat(all_neg_edge_index, 1)
        all_neg_x = torch.cat(all_neg_x, 0)            
        all_neg_edge_attr = torch.cat(all_neg_edge_attr, 0)
        all_neg_edge_mask = torch.cat(all_neg_edge_mask, 0)
        all_neg_x_pos = torch.cat(all_neg_x_pos, 0)

        all_pos_n_size = torch.tensor(all_pos_n_size)
        all_pos_e_size = torch.tensor(all_pos_e_size)
        all_neg_n_size = torch.tensor(all_neg_n_size)
        all_neg_e_size = torch.tensor(all_neg_e_size)

        all_pos_n_size = cumsum(all_pos_n_size)
        all_pos_e_size = cumsum(all_pos_e_size)
        all_neg_n_size = cumsum(all_neg_n_size)
        all_neg_e_size = cumsum(all_neg_e_size)

        pos_save_dict = {
            'edge_index': all_pos_edge_index,
            'x': all_pos_x,
            'edge_attr': all_pos_edge_attr,
            'edge_mask': all_pos_edge_mask,
            'x_pos': all_pos_x_pos,
            'n_size': all_pos_n_size,
            'e_size': all_pos_e_size
        }

        neg_save_dict = {
            'edge_index': all_neg_edge_index,
            'x': all_neg_x,
            'edge_attr': all_neg_edge_attr,
            'edge_mask': all_neg_edge_mask,
            'x_pos': all_neg_x_pos,
            'n_size': all_neg_n_size,
            'e_size': all_neg_e_size
        }

        print("saving")
        torch.save(pos_save_dict, os.path.join(save_path, "pos-%s.pt" % self.mode))
        torch.save(neg_save_dict, os.path.join(save_path, "neg-%s.pt" % self.mode))
        self.pos_dict = pos_save_dict
        self.neg_dict = neg_save_dict

    def __getitem__(self, index): 
        t1 = time.time()
        pos_graphs = dict_to_torch_geometric(index, self.pos_dict)
        neg_graphs = dict_to_torch_geometric(index, self.neg_dict)
        t2 = time.time()
        n = len(pos_graphs)
#         curr_tasks_idx = list(range(0, self.few+self.nq))
        curr_tasks_idx = np.random.choice(range(n), self.few+self.nq)
        support_triples = curr_tasks_idx[:self.few]
        query_triples = curr_tasks_idx[self.few:]       

        support_subgraphs = [pos_graphs[i] for i in curr_tasks_idx[:self.few]]
        query_subgraphs = [pos_graphs[i] for i in curr_tasks_idx[self.few:]]       
        
#         import pdb;pdb.set_trace()
#         curr_tasks_idx = list(range(0, self.few+self.nq))
        curr_tasks_idx = np.random.choice(range(n), self.few+self.nq)
        support_negative_triples = curr_tasks_idx[:self.few]
        negative_triples = curr_tasks_idx[self.few:]       
            

        support_negative_subgraphs = [neg_graphs[i] for i in curr_tasks_idx[:self.few]]
        negative_subgraphs = [neg_graphs[i] for i in curr_tasks_idx[self.few:]]          
        
        
        curr_rel = index
        t3 = time.time()
        self.t_torch += t2 - t1
        self.t_others += t3 - t2
        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel
    
    
    def next_one_on_eval(self):
        assert False
        if self.curr_tri_idx == self.num_rels:
            return "EOT", "EOT"
        self.curr_tri_idx += 1
        
        index, pos_graphs, neg_graphs = task_to_torch_geometric(*get_task(self.curr_tri_idx, prefix = self.mode, base_path = self.raw_data_paths, use_cache = self.use_cache, num_neg = 1000,hop = self.hop))
        n = len(pos_graphs)
        curr_tasks_idx = np.random.choice(range(n), self.few+1)
        support_triples = curr_tasks_idx[:self.few]
        query_triples = curr_tasks_idx[self.few:]       

        support_subgraphs = [pos_graphs[i] for i in curr_tasks_idx[:self.few]]
        query_subgraphs = [pos_graphs[i] for i in curr_tasks_idx[self.few:]]       
        
        
        n = len(neg_graphs)
        curr_tasks_idx = np.random.choice(range(n), n)
        support_negative_triples = curr_tasks_idx[:self.few]
        negative_triples = curr_tasks_idx[self.few:]       
            

        support_negative_subgraphs = [neg_graphs[i] for i in curr_tasks_idx[:self.few]]
        negative_subgraphs = [neg_graphs[i] for i in curr_tasks_idx[self.few:]]          
        
        
        curr_rel = index

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triples = [query_triples]
        negative_triples = [negative_triples]        
        

        support_subgraphs = Batch.from_data_list(support_subgraphs) 
        support_negative_subgraphs  = Batch.from_data_list(support_negative_subgraphs)       
        query_subgraphs  = Batch.from_data_list(query_subgraphs)      
        
        return [support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs], [curr_rel]
        

        
if __name__ == "__main__":


    dataset = CSGDataset(".", dataset = "data51_diverse_multi_1_10k", mode = "test", preprocess = True, rule_type = "multi_line_1")
#     dataset = CSGDataset(".", dataset = "data51_diverse_multi_1_10k", mode = "dev", preprocess = True, rule_type = "multi_line_1")
    dataset = CSGDataset(".", dataset = "data51_diverse_multi_1_10k", mode = "train", preprocess = True, rule_type = "multi_line_1")


        
        
        
        