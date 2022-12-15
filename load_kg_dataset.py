import os
import glob
import json
import torch
import struct
import logging
import copy
import pickle
import numpy as np
import random
import os.path as osp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset, DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from torch import Tensor
import multiprocessing as mp


from tqdm import tqdm
import lmdb
from scipy.sparse import csc_matrix

class Collater:
    def __init__(self):
        pass

    def __call__(self, batch):
        support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel = list(map(list, zip(*batch)))
        if support_subgraphs[0] is None:
            return ((torch.tensor(support_triples), None,
                torch.tensor(support_negative_triples), None,
                torch.tensor(query_triples), None,
                torch.tensor(negative_triples), None),
                curr_rel)

        support_subgraphs = [item for sublist in support_subgraphs for item in sublist]
        support_negative_subgraphs = [item for sublist in support_negative_subgraphs for item in sublist]
        query_subgraphs = [item for sublist in query_subgraphs for item in sublist]
        negative_subgraphs = [item for sublist in negative_subgraphs for item in sublist]

        return ((support_triples, Batch.from_data_list(support_subgraphs),
                support_negative_triples, Batch.from_data_list(support_negative_subgraphs),
                query_triples, Batch.from_data_list(query_subgraphs),
                negative_triples, Batch.from_data_list(negative_subgraphs)),
                curr_rel)




class PairSubgraphsFewShotDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(),
            **kwargs,
       )

    def next_batch(self):
        return next(iter(self))



def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))



def ssp_multigraph_to_g(graph, cache = None):
    """
    Converting ssp multigraph (i.e. list of adjs) to torch geometric graph
    """
    if cache and os.path.exists(cache):
        print("Use cache from: ", cache)
        g = torch.load(cache)
        return g, g.edge_attr.max() + 1, g.num_nodes


    edge_list = [[],[]]
    edge_features = []
    for i in range(len(graph)):
        edge_list[0].append(graph[i].nonzero()[0])
        edge_list[1].append(graph[i].nonzero()[1])
        edge_features.append(torch.full((len(graph[i].nonzero()[0]),), i))

    edge_list[0] = np.concatenate(edge_list[0])
    edge_list[1] = np.concatenate(edge_list[1])
    edge_index = torch.tensor(np.array(edge_list))

    g = Data(x=None, edge_index=edge_index.long(), edge_attr= torch.cat(edge_features).long(), num_nodes=graph[0].shape[0])

    if cache:
        torch.save(g, cache)

    return g, len(graph), g.num_nodes

class SubgraphFewshotDataset(Dataset):
    def __init__(self, root, add_traspose_rels=False, shot = 1, n_query = 3, hop = 2, dataset='', mode='dev',kind = "union_prune_plus", preprocess = False, preprocess_50neg = False, skip= False, rev = False, use_fix2 = False, num_rank_negs = 50, inductive = False, orig_test = False):
        self.root = root
        if orig_test and mode == "test":
            mode = "orig_test"
        self.mode = mode
        self.dataset = dataset
        self.inductive = inductive
        self.rev = rev
        raw_data_paths = os.path.join(root, dataset)

        postfix = "" if not inductive else "_inductive"

        if mode == "pretrain":
            self.tasks = json.load(open(os.path.join(raw_data_paths, mode + f'_tasks{postfix}.json')))
            self.tasks_neg = json.load(open(os.path.join(raw_data_paths, mode + f'_tasks_neg{postfix}.json')))
            print(os.path.join(raw_data_paths, mode + f'_tasks{postfix}.json'))
        else:
            # dev and test
            self.tasks = json.load(open(os.path.join(raw_data_paths, mode + '_tasks.json')))
            self.tasks_neg = json.load(open(os.path.join(raw_data_paths, mode + '_tasks_neg.json')))
            print(os.path.join(raw_data_paths, mode + '_tasks.json'))

        if mode == "test" and inductive and not preprocess and not preprocess_50neg:
            print("subsample tasks!!!!!!!!!!!!!!!!!!!")
            self.test_tasks_idx = json.load(open(os.path.join(raw_data_paths,  'sample_test_tasks_idx.json')))
            for r in list(self.tasks.keys()):
                if r not in self.test_tasks_idx:
                    self.tasks[r] = []
                else:
                    self.tasks[r] = np.array(self.tasks[r])[self.test_tasks_idx[r]].tolist()

        self.e1rel_e2 = json.load(open(os.path.join(raw_data_paths,'e1rel_e2.json')))
        self.all_rels = sorted(list(self.tasks.keys()))
        self.all_rels2id = { self.all_rels[i]:i for i in range(len(self.all_rels))}

        if mode == "test" and inductive and not preprocess and not preprocess_50neg:
            for idx, r in enumerate(list(self.all_rels)):
                if len(self.tasks[r]) == 0:
                    del self.tasks[r]
                    print("remove empty tasks!!!!!!!!!!!!!!!!!!!")
            self.all_rels = sorted(list(self.tasks.keys()))

        self.num_rels = len(self.all_rels)



        self.few = shot
        self.nq = n_query
        try:
            if mode == "pretrain":
                self.tasks_neg_all = json.load(open(os.path.join(raw_data_paths, mode + f'_tasks_{num_rank_negs}neg{postfix}.json')))
            else:
                self.tasks_neg_all = json.load(open(os.path.join(raw_data_paths, mode + f'_tasks_{num_rank_negs}neg.json')))


            self.all_negs = sorted(list(self.tasks_neg_all.keys()))
            self.all_negs2id = { self.all_negs[i]:i for i in range(len(self.all_negs))}
            self.num_all_negs = len(self.all_negs)
        except:
            print(mode + f'_tasks_{num_rank_negs}neg.json', "not exists")

        if mode not in ['train', 'pretrain']:

            self.eval_triples = []
            self.eval_triples_ids = []
            for rel in self.all_rels:
                for i in np.arange(0, len(self.tasks[rel]), 1)[self.few:]:
                    self.eval_triples.append(self.tasks[rel][i])
                    self.eval_triples_ids.append(i)

            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0


        ###### backgroud KG #######
        if mode=='pretrain':
            cache = os.path.join(raw_data_paths, f'graph{postfix}.pt')
        else:
            cache = os.path.join(raw_data_paths, f'graph.pt')
        if os.path.exists(cache):
            print("Use cache from: ", cache)
            ssp_graph = None

            with open(os.path.join(raw_data_paths, f'relation2id{postfix}.json'), 'r') as f:
                relation2id = json.load(f)
            with open(os.path.join(raw_data_paths, f'entity2id{postfix}.json'), 'r') as f:
                entity2id = json.load(f)

            id2relation = {v: k for k, v in relation2id.items()}
            id2entity = {v: k for k, v in entity2id.items()}

        else:
            ssp_graph, __, entity2id, relation2id, id2entity, id2relation = process_files(raw_data_paths, inductive = inductive, inductive_graph=mode == 'pretrain')
#             self.num_rels_bg = len(ssp_graph)

            # Add transpose matrices to handle both directions of relations.
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

            # the effective number of relations after adding symmetric adjacency matrices and/or self connections
#             self.num_rels_bg = len(ssp_graph)

        self.graph, _, self.num_nodes_bg = ssp_multigraph_to_g(ssp_graph, cache)

        self.num_rels_bg = len(relation2id.keys())
        if rev:
            self.num_rels_bg = self.num_rels_bg * 2 # add rev edges
#         self.ssp_graph = ssp_graph
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = id2entity
        self.id2relation = id2relation

        ###### preprocess subgraphs #######

        if rev:
            self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_rev_fix_new_{kind}_hop={hop}" + postfix)
        else:
            self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_fix_new_{kind}_hop={hop}"+ postfix)
        if use_fix2:
            if rev:
                self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_rev_fix2_new_{kind}_hop={hop}"+ postfix)
            else:
                self.dict_save_path = os.path.join(raw_data_paths, f"preprocessed_fix2_new_{kind}_hop={hop}"+ postfix)
        print(self.dict_save_path)
        if not os.path.exists(self.dict_save_path):
            os.mkdir(self.dict_save_path)

        if preprocess:
            db_path = os.path.join(raw_data_paths, f"subgraphs_fix_new_{kind}_hop=" + str(hop)+ postfix)
            if use_fix2:
                db_path = os.path.join(raw_data_paths, f"subgraphs_fix2_new_{kind}_hop=" + str(hop)+ postfix)
            if mode == "pretrain":
                db_path = os.path.join(raw_data_paths, f"subgraphs_fix_new_{kind}_hop=" + str(hop)+ postfix)
            print(db_path)
            self.main_env = lmdb.open(db_path, readonly=True, max_dbs=4, lock=False)

            self.db_pos = self.main_env.open_db((mode + "_pos").encode())
            self.db_neg = self.main_env.open_db((mode + "_neg").encode())


            self.max_n_label = np.array([3, 3])

            self._preprocess()

        if preprocess_50neg:
            db_path_50negs = os.path.join(raw_data_paths, f"subgraphs_fix_new_{kind}_{num_rank_negs}negs_hop=" + str(hop)+ postfix)
            if use_fix2:
                db_path_50negs = os.path.join(raw_data_paths, f"subgraphs_fix2_new_{kind}_{num_rank_negs}negs_hop=" + str(hop)+ postfix)
            print(db_path_50negs)
            self.main_env = lmdb.open(db_path_50negs, readonly=True, max_dbs=3, lock=False)

            self.db_50negs = self.main_env.open_db((mode + "_neg").encode())

            self.max_n_label = np.array([0, 0])
            with self.main_env.begin() as txn:
                self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
                self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self._preprocess_50negs(num_rank_negs)


        if (not preprocess) and (not preprocess_50neg) and (not skip):
            try:
                self.pos_dict = torch.load(os.path.join(self.dict_save_path, "pos-%s.pt" % self.mode))
                self.neg_dict = torch.load(os.path.join(self.dict_save_path, "neg-%s.pt" % self.mode))
            except:
                print("pos-%s.pt" % self.mode,"neg-%s.pt" % self.mode, "not exists")

            try:
                self.all_neg_dict  = torch.load(os.path.join(self.dict_save_path, f"neg_{num_rank_negs}negs-%s.pt" % self.mode))
            except:
                print( f"neg_{num_rank_negs}negs-%s.pt" % self.mode, "not exists")



    def __len__(self):
        return self.num_rels if self.num_rels != 0 else 1 ## dummy train

    def _save_torch_geometric(self, index):
        curr_rel = self.all_rels[index]
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_neg = self.tasks_neg[curr_rel]
        curr_tasks_neg_idx = np.arange(0, len(curr_tasks_neg), 1)


        pos_edge_index, pos_x, pos_x_id, pos_edge_attr, pos_n_size, pos_e_size = [], [], [], [], [], []
        neg_edge_index, neg_x, neg_x_id, neg_edge_attr, neg_n_size, neg_e_size = [], [], [], [], [], []

        with self.main_env.begin(db=self.db_pos) as txn:
            for idx, i in enumerate(curr_tasks_idx):
                str_id = curr_rel.encode()+'{:08}'.format(i).encode('ascii')
                nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
                d = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
                if nodes_pos[0] == nodes_pos[1]:
                    print(curr_rel, index, i, curr_tasks[i])
                pos_edge_index.append(d.edge_index)
                pos_x.append(d.x)
                pos_x_id.append(d.x_id)
                pos_edge_attr.append(d.edge_attr)
                pos_n_size.append(d.x.shape[0])
                pos_e_size.append(d.edge_index.shape[1])

        with self.main_env.begin(db=self.db_neg) as txn:
            for idx, i in enumerate(curr_tasks_neg_idx):
                str_id = curr_rel.encode()+'{:08}'.format(i).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                d = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                if nodes_neg[0] == nodes_neg[1]:
                    print("neg", curr_rel, index, i, curr_tasks[i])
                neg_edge_index.append(d.edge_index)
                neg_x.append(d.x)
                neg_x_id.append(d.x_id)
                neg_edge_attr.append(d.edge_attr)
                neg_n_size.append(d.x.shape[0])
                neg_e_size.append(d.edge_index.shape[1])

        return torch.cat(pos_edge_index, 1), torch.cat(pos_x, 0), torch.cat(pos_x_id, 0), torch.cat(pos_edge_attr, 0), torch.LongTensor(pos_n_size), torch.LongTensor(pos_e_size), torch.cat(neg_edge_index, 1), torch.cat(neg_x, 0), torch.cat(neg_x_id, 0), torch.cat(neg_edge_attr, 0), torch.LongTensor(neg_n_size), torch.LongTensor(neg_e_size)

    def dict_to_torch_geometric(self, index, data_dict):

        if index == 0:
            task_index = 0
            start_e = 0
            start_n = 0
        else:
            task_index = data_dict["task_offsets"][index-1]
            start_e = data_dict['e_size'][task_index - 1]
            start_n = data_dict['n_size'][task_index - 1]

        task_index_end = data_dict["task_offsets"][index]

        graphs = []
        for i in range(task_index_end - task_index):
            end_e = data_dict['e_size'][task_index + i]
            end_n = data_dict['n_size'][task_index + i]
            edge_index = data_dict['edge_index'][:, start_e:end_e]
            x = data_dict['x'][start_n:end_n]
            x_id = data_dict['x_id'][start_n:end_n]
            edge_attr = data_dict['edge_attr'][start_e:end_e]
            graphs.append(Data(edge_index = edge_index, x = x, x_id = x_id, edge_attr = edge_attr))
            start_e = end_e
            start_n = end_n

        return graphs


    def _preprocess_50negs(self, num_rank_negs):
        print("start preprocessing 50negs for %s" % self.mode)

        all_neg_edge_index, all_neg_x, all_neg_x_id, all_neg_edge_attr, all_neg_n_size, all_neg_e_size = [], [], [], [], [], []
        task_offsets_neg = []
        for index in tqdm(range(self.num_all_negs)):
            curr_rel = self.all_negs[index]
            curr_tasks_neg = self.tasks_neg_all[curr_rel]
            curr_tasks_neg_idx = np.arange(0, len(curr_tasks_neg), 1)

            neg_edge_index, neg_x, neg_x_id, neg_edge_attr, neg_n_size, neg_e_size = [], [], [], [], [], []


            with self.main_env.begin(db=self.db_50negs) as txn:
                for idx, i in enumerate(curr_tasks_neg_idx):
                    str_id = curr_rel.encode()+'{:08}'.format(i).encode('ascii')
                    nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                    d = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                    neg_edge_index.append(d.edge_index)
                    neg_x.append(d.x)
                    neg_x_id.append(d.x_id)
                    neg_edge_attr.append(d.edge_attr)
                    neg_n_size.append(d.x.shape[0])
                    neg_e_size.append(d.edge_index.shape[1])


            all_neg_edge_index.append(torch.cat(neg_edge_index, 1))
            all_neg_x.append(torch.cat(neg_x, 0))
            all_neg_x_id.append(torch.cat(neg_x_id, 0))
            all_neg_edge_attr.append(torch.cat(neg_edge_attr, 0))
            all_neg_n_size.append(torch.LongTensor(neg_n_size))
            all_neg_e_size.append(torch.LongTensor(neg_e_size))
            task_offsets_neg.append(len(torch.LongTensor(neg_n_size)))

        print("concat all")

        all_neg_edge_index = torch.cat(all_neg_edge_index, 1)
        all_neg_x = torch.cat(all_neg_x, 0)
        all_neg_x_id = torch.cat(all_neg_x_id, 0)
        all_neg_edge_attr = torch.cat(all_neg_edge_attr, 0)

        all_neg_n_size = torch.cat(all_neg_n_size)
        all_neg_e_size = torch.cat(all_neg_e_size)

        all_neg_n_size = torch.cumsum(all_neg_n_size, 0)
        all_neg_e_size = torch.cumsum(all_neg_e_size, 0)


        task_offsets_neg = torch.tensor(task_offsets_neg)
        task_offsets_neg = torch.cumsum(task_offsets_neg, 0)

        save_path = self.dict_save_path

        neg_save_dict = {
            'edge_index': all_neg_edge_index,
            'x': all_neg_x,
            'x_id': all_neg_x_id,
            'edge_attr': all_neg_edge_attr,
            'task_offsets': task_offsets_neg,
            'n_size': all_neg_n_size,
            'e_size': all_neg_e_size
        }

        print("saving to", os.path.join(save_path, f"neg_{num_rank_negs}negs-%s.pt" % self.mode))
        torch.save(neg_save_dict, os.path.join(save_path, f"neg_{num_rank_negs}negs-%s.pt" % self.mode))
        self.all_neg_dict = neg_save_dict

    def _preprocess(self):
        print("start preprocessing %s" % self.mode)
        all_pos_edge_index, all_pos_x, all_pos_x_id, all_pos_edge_attr, all_pos_n_size, all_pos_e_size = [], [], [], [], [], []
        all_neg_edge_index, all_neg_x, all_neg_x_id, all_neg_edge_attr, all_neg_n_size, all_neg_e_size = [], [], [], [], [], []
        task_offsets_pos = []
        task_offsets_neg = []
        for index in tqdm(range(self.num_rels)):
            pos_edge_index, pos_x, pos_x_id, pos_edge_attr, pos_n_size, pos_e_size, neg_edge_index, neg_x, neg_x_id, neg_edge_attr, neg_n_size, neg_e_size = self._save_torch_geometric(index)
            all_pos_edge_index.append(pos_edge_index)
            all_pos_x.append(pos_x)
            all_pos_x_id.append(pos_x_id)
            all_pos_edge_attr.append(pos_edge_attr)
            all_pos_n_size.append(pos_n_size)
            all_pos_e_size.append(pos_e_size)
            task_offsets_pos.append(len(pos_n_size))

            all_neg_edge_index.append(neg_edge_index)
            all_neg_x.append(neg_x)
            all_neg_x_id.append(neg_x_id)
            all_neg_edge_attr.append(neg_edge_attr)
            all_neg_n_size.append(neg_n_size)
            all_neg_e_size.append(neg_e_size)
            task_offsets_neg.append(len(neg_n_size))

        print("concat all")
        all_pos_edge_index = torch.cat(all_pos_edge_index, 1)
        all_pos_x = torch.cat(all_pos_x, 0)
        all_pos_x_id = torch.cat(all_pos_x_id, 0)
        all_pos_edge_attr = torch.cat(all_pos_edge_attr, 0)


        all_neg_edge_index = torch.cat(all_neg_edge_index, 1)
        all_neg_x = torch.cat(all_neg_x, 0)
        all_neg_x_id = torch.cat(all_neg_x_id, 0)
        all_neg_edge_attr = torch.cat(all_neg_edge_attr, 0)


        all_pos_n_size = torch.cat(all_pos_n_size)
        all_pos_e_size = torch.cat(all_pos_e_size)
        all_neg_n_size = torch.cat(all_neg_n_size)
        all_neg_e_size = torch.cat(all_neg_e_size)

        all_pos_n_size = torch.cumsum(all_pos_n_size, 0)
        all_pos_e_size = torch.cumsum(all_pos_e_size, 0)
        all_neg_n_size = torch.cumsum(all_neg_n_size, 0)
        all_neg_e_size = torch.cumsum(all_neg_e_size, 0)


        task_offsets_pos = torch.tensor(task_offsets_pos)
        task_offsets_pos = torch.cumsum(task_offsets_pos, 0)
        task_offsets_neg = torch.tensor(task_offsets_neg)
        task_offsets_neg = torch.cumsum(task_offsets_neg, 0)

        save_path = self.dict_save_path
        pos_save_dict = {
            'edge_index': all_pos_edge_index,
            'x': all_pos_x,
            'x_id': all_pos_x_id,
            'edge_attr': all_pos_edge_attr,
            'task_offsets': task_offsets_pos,
            'n_size': all_pos_n_size,
            'e_size': all_pos_e_size
        }

        neg_save_dict = {
            'edge_index': all_neg_edge_index,
            'x': all_neg_x,
            'x_id': all_neg_x_id,
            'edge_attr': all_neg_edge_attr,
            'task_offsets': task_offsets_neg,
            'n_size': all_neg_n_size,
            'e_size': all_neg_e_size
        }

        print("saving")
        torch.save(pos_save_dict, os.path.join(save_path, "pos-%s.pt" % self.mode))
        torch.save(neg_save_dict, os.path.join(save_path, "neg-%s.pt" % self.mode))
        self.pos_dict = pos_save_dict
        self.neg_dict = neg_save_dict

    def __getitem__(self, index):
        # get current relation and current candidates
        curr_rel = self.all_rels[index]
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        if self.nq is not None:
            curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few+self.nq, replace = False)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        all_pos_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.pos_dict)
        all_neg_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.neg_dict)

        ### extract subgraphs
        support_subgraphs = []
        query_subgraphs = []
        for idx, i in enumerate(curr_tasks_idx):

            if self.mode == "test" and self.inductive:
                subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                subgraph_pos = all_pos_graphs[i]
            if idx < self.few:
                support_subgraphs.append(subgraph_pos)
            else:
                query_subgraphs.append(subgraph_pos)


        curr_tasks_neg = self.tasks_neg[curr_rel]
        curr_tasks_neg_idx = curr_tasks_idx

        support_negative_triples = [curr_tasks_neg[i] for i in curr_tasks_neg_idx[:self.few]]
        negative_triples = [curr_tasks_neg[i] for i in curr_tasks_neg_idx[self.few:]]

        # construct support and query negative triples
        support_negative_subgraphs = []
        negative_subgraphs = []
        for idx, i in enumerate(curr_tasks_neg_idx):

            if self.mode == "test" and self.inductive:
                subgraph_neg = all_neg_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                subgraph_neg = all_neg_graphs[i]

            if (self.mode in ["train", "pretrain"] and self.dataset in ['NELL', 'FB15K-237'] and not self.inductive):
                #choose 1 neg from 50
                e1, r, e2 = curr_tasks[i]
                all_50_neg_graphs = self.dict_to_torch_geometric(self.all_negs2id[e1 + r + e2], self.all_neg_dict)
                subgraph_neg = random.choice(all_50_neg_graphs)

            if idx < self.few:
                support_negative_subgraphs.append(subgraph_neg)
            else:
                negative_subgraphs.append(subgraph_neg)

        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel


    def next_one_on_eval(self, index):
        # get current triple
        query_triple = self.eval_triples[index]
        curr_rel = query_triple[1]
        curr_rel_neg = query_triple[0] + query_triple[1] + query_triple[2]
        curr_task = self.tasks[curr_rel]

        all_pos_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.pos_dict)
        all_neg_graphs = self.dict_to_torch_geometric(self.all_rels2id[curr_rel], self.neg_dict)
        all_50_neg_graphs = self.dict_to_torch_geometric(self.all_negs2id[curr_rel_neg], self.all_neg_dict)

        # get support triples
        support_triples_idx = np.arange(0, len(curr_task), 1)[:self.few]
        support_triples = []
        support_subgraphs = []
        for idx, i in enumerate(support_triples_idx):
            support_triples.append(curr_task[i])
            if self.mode == "test" and self.inductive:
                subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                subgraph_pos = all_pos_graphs[i]
            support_subgraphs.append(subgraph_pos)

        query_triples = [query_triple]
        query_subgraphs = []

        if self.mode == "test" and self.inductive:
            subgraph_pos = all_pos_graphs[self.test_tasks_idx[curr_rel][self.eval_triples_ids[index]]]
        else:
            subgraph_pos = all_pos_graphs[self.eval_triples_ids[index]]

        query_subgraphs.append(subgraph_pos)


        # construct support negative

        curr_task_neg = self.tasks_neg[curr_rel]
        support_negative_triples_idx = support_triples_idx
        support_negative_triples = []
        support_negative_subgraphs = []
        for idx, i in enumerate(support_negative_triples_idx):
            support_negative_triples.append(curr_task_neg[i])

            if self.mode == "test" and self.inductive:
                subgraph_neg = all_neg_graphs[self.test_tasks_idx[curr_rel][i]]
            else:
                subgraph_neg = all_neg_graphs[i]

            support_negative_subgraphs.append(subgraph_neg)



        ### 50 query negs
        curr_task_50neg = self.tasks_neg_all[curr_rel_neg]
        negative_triples_idx = np.arange(0, len(curr_task_50neg), 1)
        negative_triples = []
        negative_subgraphs = []
        for idx, i in enumerate(negative_triples_idx):
            negative_triples.append(curr_task_50neg[i])
            negative_subgraphs.append(all_50_neg_graphs[i])



        return support_triples, support_subgraphs, support_negative_triples, support_negative_subgraphs, query_triples, query_subgraphs, negative_triples, negative_subgraphs, curr_rel



    def _prepare_subgraphs(self, nodes, r_label, n_labels):
#         import pdb;pdb.set_trace()
        if nodes[0] == nodes[1]:
            print(nodes)
            print("self-loop...")
            nodes = nodes[:2]
            subgraph = Data(edge_index = torch.zeros([2, 0]), edge_attr = torch.zeros([0]),  num_nodes = 2)
        else:
            subgraph = get_subgraph(self.graph, torch.tensor(nodes))
        # remove the (0,1) target edge
        index = (torch.tensor([0, 1]) == subgraph.edge_index.transpose(0,1)).all(1)
        index = index & (subgraph.edge_attr == r_label)
        if index.any():
            subgraph.edge_index = subgraph.edge_index.transpose(0,1)[~index].transpose(0,1)
            subgraph.edge_attr= subgraph.edge_attr[~index]


        # add reverse edges
        if self.rev:
            subgraph.edge_index = torch.cat([subgraph.edge_index, subgraph.edge_index.flip(0)], 1)
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, self.num_rels_bg - subgraph.edge_attr], 0)

        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.num_nodes
        n_labels = n_labels.astype(int)
        label_feats = np.zeros((n_nodes, 6))
        label_feats[0] = [1, 0, 0, 0, 1, 0]
        label_feats[1] = [0, 1, 0, 1, 0, 0]



        subgraph.x = torch.FloatTensor(label_feats)
        subgraph.x_id = torch.LongTensor(nodes)

        # sort it
        edge_index = subgraph.edge_index
        edge_attr =  subgraph.edge_attr
        row = edge_index[0]
        col = edge_index[1]
        idx = col.new_zeros(col.numel() + 1)
        idx[1:] = row
        idx[1:] *= subgraph.x.shape[0]
        idx[1:] += col
        perm = idx[1:].argsort()
        row = row[perm]
        col = col[perm]
        edge_attr = edge_attr[perm]
        edge_index = torch.stack([row,col], 0)

        subgraph.edge_index = edge_index
        subgraph.edge_attr =  edge_attr

        return subgraph



def process_files(data_path, use_cache = True, inductive = False, inductive_graph=False):

    entity2id = {}
    relation2id = {}

    postfix = "" if not inductive else "_inductive"

    relation2id_path = os.path.join(data_path, f'relation2id{postfix}.json')
    if use_cache and os.path.exists(relation2id_path):
        print("Use cache from: ", relation2id_path)
        with open(relation2id_path, 'r') as f:
            relation2id = json.load(f)



    entity2id_path = os.path.join(data_path, f'entity2id{postfix}.json')
    if use_cache and os.path.exists(entity2id_path):
        print("Use cache from: ", entity2id_path)
        with open(entity2id_path, 'r') as f:
            entity2id = json.load(f)

    triplets = {}

    ent = 0
    rel = 0

    for mode in ['bg']: # assuming only one kind of background graph for now

        if inductive_graph:
            file_path = os.path.join(data_path,f'path_graph_inductive.json')
        else:
            file_path = os.path.join(data_path, f'path_graph.json')
        data = []
        with open(file_path) as f:
            file_data = json.load(f)

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[mode] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['bg'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['bg'][:, 0][idx].squeeze(1), triplets['bg'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))


    if not os.path.exists(relation2id_path):
        with open(relation2id_path, 'w') as f:
            json.dump(relation2id, f)

    if not os.path.exists(entity2id_path):
        with open(entity2id_path, 'w') as f:
            json.dump(entity2id, f)
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation



def index_to_mask(index, size = None):
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def get_subgraph(graph, nodes):
    """ from torch_geomtric"""
#     print(nodes)
    relabel_nodes = True
    device = graph.edge_index.device

    num_nodes = graph.num_nodes
    subset = index_to_mask(nodes, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
    edge_index = graph.edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[nodes] = torch.arange(subset.sum().item(), device=device)
        edge_index = node_idx[edge_index]


    num_nodes = nodes.size(0)

    data = copy.copy(graph)

    for key, value in data:
        if key == 'edge_index':
            data.edge_index = edge_index
        elif key == 'num_nodes':
            data.num_nodes = num_nodes
        elif isinstance(value, Tensor):
            if graph.is_node_attr(key):
                data[key] = value[subset]
            elif graph.is_edge_attr(key):
                data[key] = value[edge_mask]

    return data

class SubgraphFewshotDatasetRankTail(SubgraphFewshotDataset):
    def __len__(self):
        return len(self.eval_triples)

    def __getitem__(self, index):
        return self.next_one_on_eval(index)


if __name__ == "__main__":
    pass
#     dataset = SubgraphFewshotDataset(".", dataset="NELL", mode="test", hop = 2, shot = 3, preprocess = True)
