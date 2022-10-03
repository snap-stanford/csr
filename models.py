import torch
import torch.nn as nn
from RGCN import RGCNNet
from torch_geometric.nn import MessagePassing

from torch_scatter import scatter_sum
from torch_sparse import SparseTensor, spmm
from sklearn.metrics import roc_auc_score

import pdb
import numpy as np
from torch_geometric.utils import subgraph
import os
import pickle
import json
import torch.nn.functional as F

SYNTHETIC = False

def load_embed(datapath, emb_path, dataset="NELL", embed_model = "ComplEx", use_ours = True, load_ent = False, bidir=False, inductive = False):
    tail = ""
    if inductive:
        tail += "_inductive"
    rel2id = json.load(open(datapath + f'/relation2id{tail}.json'))
    ent2id = json.load(open(datapath + f'/entity2id{tail}.json'))
    
    if inductive:
        try:
            inductive_ndoes= json.load(open(datapath + f'/inductive_nodes.json'))
        except:
            print("inductive_ndoes not found")
            inductive_ndoes = []
    else:
        inductive_ndoes = []

    if  not use_ours:    
        print("use original emb", embed_model)            
        assert dataset == "NELL" and not inductive
        theirs_rel2id = json.load(open(emb_path + f'/{dataset}/relation2ids'))
        theirs_ent2id = json.load(open(emb_path + f'/{dataset}/ent2ids'))

        print ("loading pre-trained embedding...")
        if embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
                    
            rel_embed = np.loadtxt(emb_path + f'/{dataset}/embed/relation2vec.' + embed_model)
            ent_embed = np.loadtxt(emb_path + f'/{dataset}/embed/entity2vec.' + embed_model)     

            if embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            if not load_ent:   
                embeddings = []
                id2rel = {v: k for k, v in rel2id.items()}

                for key_id in range(len(rel2id.keys())):
                    key = id2rel[key_id]
                    if key not in ['','OOV']:
                        embeddings.append(list(rel_embed[theirs_rel2id[key],:]))

                # just add a random extra one        
                embeddings.append(list(rel_embed[0,:]))        
                return np.array(embeddings)

            else:  
                embeddings = []
                
                id2ent = {v: k for k, v in ent2id.items()}

                for key_id in range(len(ent2id.keys())):
                    key = id2ent[key_id]
                    if key not in ['', 'OOV']:
                        if key in inductive_ndoes:
                            embeddings.append(np.random.normal(size = ent_embed.shape[1]))
                        else:
                            embeddings.append(list(ent_embed[theirs_ent2id[key],:]))
                            
                return np.array(embeddings)
    else:
        print("use ours emb")            

        prefix = f'{dataset}-fs'
        if bidir:
            prefix += '-bidir'
        if inductive:
            prefix += '-ind'    
        theirs_rel2id = pickle.load(open(emb_path + f'/{prefix}/rel2id.pkl', 'rb'))
        theirs_ent2id = pickle.load(open(emb_path + f'/{prefix}/ent2id.pkl', 'rb'))

        print ("loading ours pre-trained embedding...")
        if embed_model == 'TransE':
            ckpt = torch.load(emb_path + f'/{prefix}/checkpoint', map_location='cpu')
        elif embed_model == 'ComplEx':
            ckpt = torch.load(emb_path + f'/{prefix}/complex_checkpoint', map_location='cpu')
        
        if not load_ent:
            rel_embed = ckpt['model_state_dict']['relation_embedding.embedding']
            embeddings = []
            id2rel = {v: k for k, v in rel2id.items()}

            for key_id in range(len(rel2id.keys())):
                key = id2rel[key_id]
                if key not in ['','OOV']:
                    embeddings.append(list(rel_embed[theirs_rel2id[key],:]))

            # just add a random extra one        
            embeddings.append(list(rel_embed[0,:]))        
            embeddings = np.array(embeddings)

            return embeddings
    
        if load_ent:
            ent_embed = ckpt['model_state_dict']['entity_embedding.embedding']
            node_embeddings = []
            id2ent = {v: k for k, v in ent2id.items()}

            for key_id in range(len(ent2id.keys())):
                key = id2ent[key_id]
                if key not in ['','OOV']:
                    if key in inductive_ndoes:
                        node_embeddings.append(np.random.normal(size = ent_embed.shape[1]))
                    else:
                        node_embeddings.append(list(ent_embed[theirs_ent2id[key],:]))


            node_embeddings = np.array(node_embeddings)
            return node_embeddings


        
    
    
def compute_connectivity_loss(support_subgraphs, edge_mask):
    # assumed edge_mask removed head - tail and tail - head direct edge
    
    batch = support_subgraphs.batch
    if batch is None:
        return torch.tensor(0.)
    row, col = support_subgraphs.edge_index.long()
    num_nodes = scatter_sum(torch.ones(batch.shape).to(batch.device), batch)
    head_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),num_nodes[:-1]]), 0).long()
    tail_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),num_nodes[:-1]]), 0).long() + 1

    adj_m = SparseTensor.from_edge_index(support_subgraphs.edge_index.long() ,edge_mask , sparse_sizes = [support_subgraphs.x.shape[0],support_subgraphs.x.shape[0]])
    adj_m_t = SparseTensor.from_edge_index(support_subgraphs.edge_index.flip(0).long(), edge_mask, sparse_sizes = [support_subgraphs.x.shape[0],support_subgraphs.x.shape[0]])
    adj_m = adj_m + adj_m_t       

            
    A1 = adj_m
    A2 = adj_m @ adj_m
    # path = I + A + A**2
    # path0 = reachability to head within 2 hops
    path0 = A1[head_idxs].to_dense() +  A2[head_idxs].to_dense()
    path0 = torch.minimum(path0, torch.tensor(1))
    path0[range(len(head_idxs)), head_idxs] = 1
    # path1 = reachability to tail within 2 hops
    path1 = A1[tail_idxs].to_dense() +  A2[tail_idxs].to_dense()
    path1 = torch.minimum(path1, torch.tensor(1))
    path1[range(len(head_idxs)), tail_idxs] = 1

    if SYNTHETIC:
        connectivity_loss = scatter_sum((- torch.minimum((path0[batch[row], row] * path1[batch[row], row] * path0[batch[col], col] * path1[batch[col], col]), torch.tensor(1)) *edge_mask), batch[row] ) / (scatter_sum(edge_mask, batch[row] )+ 1e-5)
        return connectivity_loss
    
    # both end nodes of selected edges should be reachable to both head and tail within 2 hops
    connectivity_loss = scatter_sum((- torch.minimum((path0[batch[row], row] + path1[batch[row], row] + path0[batch[col], col] + path1[batch[col], col]), torch.tensor(1)) *edge_mask), batch[row] ) / (scatter_sum(edge_mask, batch[row] )+ 1e-5)
    return connectivity_loss

                                
def print_iou(query_subgraphs, edge_mask, print_all = False):
    print(edge_mask.min(), edge_mask.mean(), edge_mask.max(), edge_mask.sum())
    row, col = query_subgraphs.edge_index
    if print_all:
        if hasattr(query_subgraphs, "rule_mask"):
            print(query_subgraphs.edge_attr[query_subgraphs.rule_mask==1])
        print(query_subgraphs.batch[row][edge_mask>0.8])
        print(query_subgraphs.edge_attr[edge_mask>0.8])
        print(query_subgraphs.edge_index[:, edge_mask>0.8])
    print((edge_mask>0.8).sum())

    if hasattr(query_subgraphs, "rule_mask"):
        gt = query_subgraphs.edge_index[:,query_subgraphs.rule_mask==1].transpose(0,1).tolist()
        gt_batch = query_subgraphs.batch[row][query_subgraphs.rule_mask==1]

        pred = query_subgraphs.edge_index[:,edge_mask>0.8].transpose(0,1).tolist()
        pred_batch = query_subgraphs.batch[row][edge_mask>0.8]

        gt_edges = [set() for _ in range(24)]
        for idx in range(len(gt)):
            gt_edges[gt_batch[idx]].add(tuple(gt[idx]))

        pred_edges = [set() for _ in range(24)]
        for idx in range(len(pred)):
            pred_edges[pred_batch[idx]].add(tuple(pred[idx]))


        ious = []
        for i in range(24):
            iou = len(gt_edges[i].intersection(pred_edges[i])) / len(gt_edges[i].union(pred_edges[i]) )
            ious.append(iou)
        print(sum(ious)/len(ious))
        print(sum([len(gt_edges[i].intersection(pred_edges[i])) for i in range(24)]))
        print(sum([len(gt_edges[i]) for i in range(24)]))


class InnerMasks(torch.nn.Module):
    def __init__(self, edge_mask_p, edge_mask_n):
        super().__init__()
        self.pm = edge_mask_p
        self.nm = edge_mask_n
    
    def forward(self):
        return self.pm.clone(), self.nm.clone()

    
class InnerMask(torch.nn.Module):
    def __init__(self, edge_mask):
        super().__init__()
        self.m = edge_mask
    
    def forward(self):
        return self.m.clone()

class InnerRel(torch.nn.Module):
    def __init__(self, rel):
        super().__init__()
        self.rel = rel
    
    def forward(self):
        return self.rel.clone()
    


def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask


class GNNEmbeddingLearner(nn.Module):
    def __init__(self, prototype_dim, emb_dim, num_prototypes = 2, hidden_dim=128, use_subgraph = False, num_rels_bg=101, num_nodes = 1000, use_node_emb = False, debug=False, logging_dir=None):
        super(GNNEmbeddingLearner, self).__init__()
        self.edge_embedding = nn.Embedding(num_rels_bg + 1, emb_dim)
        
        self.node_embedding = nn.Embedding(num_nodes, emb_dim)
        self.prototype_dim = prototype_dim
        self.rgcn = RGCNNet(emb_dim = emb_dim, input_dim = emb_dim, edge_embedding = self.edge_embedding, node_embedding = self.node_embedding, num_rels_bg = num_rels_bg, use_node_emb = False, use_node_emb_end = use_node_emb, use_noid_node_emb = SYNTHETIC)

        self.epsilon = 1e-15
        self.debug = debug
        self.last_layer = nn.Linear(num_prototypes, 1)  
        self.egnn = RGCNNet(emb_dim =emb_dim, input_dim = emb_dim + prototype_dim, num_rels_bg = num_rels_bg, edge_embedding = self.edge_embedding, node_embedding = self.node_embedding, latent_dim = [hidden_dim]*3, use_node_emb = False, use_noid_node_emb = SYNTHETIC)
        self.egnn_post_layers = nn.Sequential(
            nn.Linear(hidden_dim , 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1))
        
        self.csg_gnn = RGCNNet(emb_dim =emb_dim, input_dim = emb_dim + prototype_dim, num_rels_bg = num_rels_bg, edge_embedding = self.edge_embedding, node_embedding = self.node_embedding, latent_dim = [128]* 10, ffn=True)
        
        self.csg_gnn_post_layers = nn.Sequential(
            nn.Linear(128 , 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1))
        self.empty_idx = num_rels_bg
        self.use_subgraph = use_subgraph
        
    def masked_embedding(self, graphs, edgemask, size_loss_beta = 0):        

        clear_masks(self.rgcn)
        set_masks(self.rgcn, edgemask)
        
        emb, _, _ = self.rgcn(graphs, edgemask)
        clear_masks(self.rgcn)
        
        size_loss = torch.sum(edgemask)
        # entropy
        mask_ent = - edgemask * torch.log(edgemask + self.epsilon) - (1 - edgemask) * torch.log(1 - edgemask + self.epsilon)
        mask_ent_loss = torch.sum(mask_ent)

        extra_loss = size_loss* size_loss_beta + mask_ent_loss 
        return emb, extra_loss, edgemask


    def gen_common_sg_mask_gnn(self, graphs, few = 3):
        
        row, col = graphs.edge_index.long()
        edge_batch = graphs.batch[row]
        batch_size = torch.div(graphs.batch.max(), 3, rounding_mode='floor') + 1
        
        graph_emb, _, edge_attr = self.rgcn(graphs)
        super_graphs = graphs.clone()
        for i in range(10):
            graph_emb = graph_emb.reshape(batch_size, few, -1)
            prototype = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
            prototype = prototype.expand(-1, few, -1).reshape(batch_size*few, -1)
            graph_emb, _, edge_attr = self.csg_gnn(super_graphs, extra_cond = prototype[edge_batch] )
            super_graphs.edge_attr = edge_attr + self.edge_embedding(graphs.edge_attr.long())
                

        h = self.csg_gnn_post_layers(edge_attr)
        h = h.sigmoid().reshape(-1)[: graphs.edge_index.shape[1]]
        
        return h

    
    def gen_mask_gnn(self, graphs, prototype):
        
        prototype = prototype[:, :self.prototype_dim] # in case it contains extra node emb
        row, col = graphs.edge_index.long()
        edge_batch = graphs.batch[row]
                
        _, _, edge_attr = self.egnn(graphs, extra_cond = prototype[edge_batch] )
        h = self.egnn_post_layers(edge_attr)
        h = h.sigmoid().reshape(-1)
        
        return h


    def get_masked_graph_embedding(self, graphs, prototype, size_loss_beta = 0):
        edgemask = self.gen_mask_gnn(graphs, prototype)

        emb, extra_loss, edgemask = self.masked_embedding(graphs, edgemask, size_loss_beta)
        return emb, extra_loss, edgemask
        
        
        
    def prototype_subgraph_distances(self, x, prototype):
        distance =  - nn.CosineSimilarity(dim = 1)(x, prototype)
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        if self.debug:
            print("neg cosine similarity", distance.mean())
        return similarity, distance

        
    def forward(self, support_subgraphs, support_negative_subgraphs, prototype, num_pos, num_neg, edge_mask_pos, edge_mask_neg, size_loss_beta = 0):
        batch_size = prototype.shape[0]
        dim = prototype.shape[2]
        extra_loss = torch.tensor(0)

        # copy
        prototype_pos = prototype.expand(-1, num_pos, -1).reshape(batch_size*num_pos, dim)
        prototype_neg = prototype.expand(-1, num_neg , -1).reshape(batch_size*num_neg, dim)
            
        edgemask = None
        if self.use_subgraph:
            if edge_mask_pos is not None:
                graph_emb, loss, edgemask  = self.masked_embedding(support_subgraphs, edge_mask_pos, size_loss_beta)
                graph_emb_neg, loss_neg, edgemask_neg = self.masked_embedding(support_negative_subgraphs, edge_mask_neg, size_loss_beta)
            else:
                graph_emb, loss, edgemask = self.get_masked_graph_embedding(support_subgraphs, prototype_pos, size_loss_beta)
                graph_emb_neg, loss_neg, edgemask_neg = self.get_masked_graph_embedding(support_negative_subgraphs, prototype_neg, size_loss_beta)  
            extra_loss = loss + loss_neg
        else:
            graph_emb, _ = self.rgcn(support_subgraphs)        
            graph_emb_neg, _ = self.rgcn(support_negative_subgraphs)      
        
        prototype_activations, pos_distances = self.prototype_subgraph_distances(graph_emb, prototype_pos)
        prototype_activations_neg, neg_distances = self.prototype_subgraph_distances(graph_emb_neg, prototype_neg)
        return pos_distances.reshape(-1 , 1), neg_distances.reshape(-1 , 1), extra_loss, edgemask, edgemask_neg, graph_emb, graph_emb_neg


class CSR(nn.Module):
    def __init__(self, dataset, parameter):
        super(CSR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.use_subgraph = parameter['use_subgraph']
        self.support_only = parameter['support_only']
        self.opt_mask = parameter['opt_mask']
        self.use_atten = parameter['use_atten']
        self.egnn_only = parameter['egnn_only']

        self.use_ground_truth = parameter['use_ground_truth']
        self.use_full_mask_rule = parameter['use_full_mask_rule']
        self.use_full_mask_query = parameter['use_full_mask_query']
        self.joint_train_mask = parameter['joint_train_mask']
        self.verbose = parameter['verbose']
        self.pdb_mode = parameter['pdb_mode']
        self.debug = parameter['debug']
        self.extra_loss_beta = parameter['extra_loss_beta']
        self.loss_mode = parameter['loss_mode']
        self.niters = parameter['niters']
        self.geo = parameter['geo']
        self.pool_mode = parameter['pool_mode']
        self.opt_mode = parameter['opt_mode']
        self.logging_dir = os.path.join(parameter['log_dir'], parameter['prefix'], 'data')
        
        self.emb_path = parameter['emb_path']
        self.emb_dim = parameter['emb_dim']
        self.hidden_dim = parameter['hidden_dim']
        self.full_kg = dataset.graph
        self.no_margin = parameter['no_margin']
        
        self.num_prototypes_per_class = 1
        self.prototype_dim = self.hidden_dim * 3

        

        self.embedding_learner = GNNEmbeddingLearner(self.prototype_dim , self.emb_dim, self.num_prototypes_per_class, self.hidden_dim, self.use_subgraph, num_rels_bg = dataset.num_rels_bg, num_nodes = dataset.num_nodes_bg, use_node_emb = parameter['use_pretrain_node_emb'] or parameter['use_rnd_node_emb'], debug=self.debug, logging_dir=self.logging_dir)
        print(self.embedding_learner)
        
        use_ours = True
        if dataset.dataset in ["NELL", "Wiki"] and not parameter['our_emb'] and not parameter['inductive']:
            use_ours = False
            
        
        if parameter['use_pretrain_edge_emb']:
            rel_embeddings =  load_embed(os.path.join(dataset.root, dataset.dataset), self.emb_path, dataset.dataset, use_ours = use_ours, embed_model=parameter["embed_model"], bidir = parameter['bidir'], inductive = parameter['inductive'])  

            print ("loading into edge embedding...")
            self.embedding_learner.edge_embedding.weight.data.copy_(torch.from_numpy(rel_embeddings))
        
        if parameter['use_pretrain_node_emb']:
            node_embeddings =  load_embed(os.path.join(dataset.root, dataset.dataset), self.emb_path, dataset.dataset, load_ent = True, use_ours = use_ours, embed_model=parameter["embed_model"], bidir = parameter['bidir'], inductive = parameter['inductive'])   

            print ("loading into node embedding...")
            self.embedding_learner.node_embedding.weight.data.copy_(torch.from_numpy(node_embeddings))
        
        
        

        self.loss_func = self.binary_loss
        self.rel_q_sharing = dict()

    def binary_loss(self, p_score, n_score, y):
        if self.debug:
            print("p score", p_score.mean())
            print("n score", n_score.mean())
        if self.no_margin:
            return  -p_score.view(-1).mean() + n_score.view(-1).mean()
        if (not self.support_only) and (not self.use_ground_truth) and p_score.shape[0] == n_score.shape[0]:
            if self.debug:
                print("use margin loss")
            return nn.MarginRankingLoss(self.margin)(p_score, n_score, y)
        if self.debug:
            print("use only positive loss")
        
        return 1 -p_score.view(-1).mean()  
    
        
    
    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def rgcn_only(self, task):
        support, support_subgraphs, support_negative, support_negative_subgraphs, query, query_subgraphs, negative, negative_subgraphs = task
        
        support_subgraphs, support_negative_subgraphs, query_subgraphs, negative_subgraphs = support_subgraphs.to(self.device), support_negative_subgraphs.to(self.device), query_subgraphs.to(self.device), negative_subgraphs.to(self.device)
        
        larger_masks = torch.ones(*support_subgraphs.edge_attr.shape, device=self.device)
        smaller_masks = self.deprecated_sample_masks(support_subgraphs.edge_attr.shape, ratio=np.random.uniform(0.02, 0.05))
        larger_graph_emb, larger_extra_loss, _ = self.embedding_learner.masked_embedding(support_subgraphs, larger_masks, size_loss_beta=0)
        smaller_graph_emb, smaller_extra_loss, _ = self.embedding_learner.masked_embedding(support_subgraphs, smaller_masks, size_loss_beta=0)
        return larger_graph_emb, smaller_graph_emb, larger_extra_loss + smaller_extra_loss
    
    def rgcn_loss_func(self, larger_graph_emb, smaller_graph_emb):
        rand_idx = torch.randperm(larger_graph_emb.shape[0])
        e_pos = torch.sum(torch.max(torch.zeros_like(larger_graph_emb, device=self.device), smaller_graph_emb - larger_graph_emb)**2, dim=1)
        e_neg = torch.sum(torch.max(torch.zeros_like(larger_graph_emb, device=self.device), smaller_graph_emb - larger_graph_emb[rand_idx])**2, dim=1)
        e_neg = torch.max(torch.tensor(0., device=self.device), self.margin - e_neg)
        
        return torch.sum(e_pos + e_neg)

    def deprecated_sample_masks(self, edge_shape, ratio=0.0257):
        return torch.tensor(np.random.choice([0., 1.], size=edge_shape, p=[1 - ratio, ratio]), device=self.device).float()
    
    def sample_masks(self, support_subgraphs, kk=10): 
        batch = support_subgraphs.batch
        device = self.device
        num_nodes = scatter_sum(torch.ones(batch.shape).to(batch.device), batch)
        head_idxs = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes[:-1]]), 0).long()
        n_edges = support_subgraphs.edge_index.shape[1]
        n_nodes = support_subgraphs.x.shape[0]
        n_node_in_batch = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),num_nodes]), 0).long()

        cnt = 1
        flag = True
        while flag:
            flag = False
                
            h = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
            h[head_idxs, 0] = 1.
            
            all_selected_nodes = []
            h1_selected = []
            for i in range(len(head_idxs)):
                h1_selected.append(n_node_in_batch[i])
            h1_selected = torch.tensor(h1_selected, device=device)
            all_selected_nodes.append(h1_selected)

            for _ in range(3):
                h1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h)
                h1_selected = []
                for i in range(len(head_idxs)):
                    if _ == 2:
                        prob = h1[n_node_in_batch[i]:n_node_in_batch[i+1]]
                        if not (prob[1] == 1).any():
                            flag = True
                            break
                        u = (prob[1] == 1).nonzero()[0][0].item()
                        h1_selected.append(all_selected_nodes[-1][i, u])
                    elif _ == 0:
                        prob = h1[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze()
                        prob[0] = 0
                        prob[1] = 0
                        if not (prob == 1).any():
                            flag = True
                            break
                        selected = torch.multinomial(prob, 1).item()
                        selected += n_node_in_batch[i]
                        h1_selected.append(selected)
                    else:
                        prob = h1[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze()
                        prob[0] = 0
                        prob[1] = 0
                        if not (prob == 1).any():
                            flag = True
                            break
                        selected = torch.multinomial(prob, kk, replacement=True)
                        selected += n_node_in_batch[i]
                        h1_selected.append(selected)
                if flag:
                    break
                if _ in [0, 2]:
                    h1_selected = torch.tensor(h1_selected, device=device)
                    h = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                    h[h1_selected, 0] = 1.
                elif _ == 1:
                    h1_selected = torch.stack(h1_selected)
                    h = torch.zeros(support_subgraphs.x.shape[0], kk, device=device)
                    h.scatter_(0, h1_selected, 1)
                all_selected_nodes.append(h1_selected)
            if flag:
                cnt += 1
                continue
            h1_selected = []
            for i in range(len(head_idxs)):
                h1_selected.append(n_node_in_batch[i] + 1)
            h1_selected = torch.tensor(h1_selected, device=device)
            all_selected_nodes.append(h1_selected)
            flag = False

        del all_selected_nodes[2]
        all_selected_nodes = torch.cat(all_selected_nodes, dim=0)
        s_edge_index, s_edge_attr = subgraph(all_selected_nodes, support_subgraphs.edge_index)
        node_mask = all_selected_nodes.new_zeros(support_subgraphs.x.shape[0], dtype=torch.bool)
        node_mask[all_selected_nodes] = True
        s_edge_mask = node_mask[support_subgraphs.edge_index[0]] & node_mask[support_subgraphs.edge_index[1]]

        return s_edge_index, s_edge_mask.float(), cnt

    def simple_sample_connected_masks(self, support_subgraphs, kk=10): 
        batch = support_subgraphs.batch
        device = self.device
        num_nodes = scatter_sum(torch.ones(batch.shape).to(batch.device), batch)
        head_idxs = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes[:-1]]), 0).long()
        n_edges = support_subgraphs.edge_index.shape[1]
        n_nodes = support_subgraphs.x.shape[0]
        n_node_in_batch = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),num_nodes]), 0).long()

        all_selected_nodes = []
        for i in range(len(n_node_in_batch)-1):
            if num_nodes[i] < kk:
                all_selected_nodes.append(torch.tensor(np.arange(n_node_in_batch[i].item(), n_node_in_batch[i+1].item()), device=device))
            else:
                h0 = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                h0[n_node_in_batch[i], 0] = 1.
                h1 = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                h1[n_node_in_batch[i] + 1, 0] = 1.
                
                num_hops = 2 # np.random.randint(2, 4)
                if num_hops == 2:
                    h0_1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h0)
                    h0_2 = spmm(support_subgraphs.edge_index[[1,0]], torch.ones(n_edges, device=device), n_nodes, n_nodes, h0)
                    h1_1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h1)
                    h1_2 = spmm(support_subgraphs.edge_index[[1,0]], torch.ones(n_edges, device=device), n_nodes, n_nodes, h1)
                    
                    prob = torch.clamp((h0_1+h0_2+h1_1+h1_2)[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze(), 0, 1)
                    
                    prob[0] = 0
                    prob[1] = 0
                    if not (prob == 1).any():
                        assert False
                        break
                    n_connected = int(prob.sum().item())
                    selected = torch.multinomial(prob, min(30, n_connected))
                    selected += n_node_in_batch[i]
                    
                    h1_selected = torch.tensor(selected, device=device)
                    all_selected_nodes.append(h1_selected)
                    all_selected_nodes.append(torch.tensor([n_node_in_batch[i], n_node_in_batch[i]+1], device=device))
        all_selected_nodes = torch.cat(all_selected_nodes)
        s_edge_index, s_edge_attr = subgraph(all_selected_nodes, support_subgraphs.edge_index)
        node_mask = all_selected_nodes.new_zeros(support_subgraphs.x.shape[0], dtype=torch.bool)
        node_mask[all_selected_nodes] = True
        s_edge_mask = node_mask[support_subgraphs.edge_index[0]] & node_mask[support_subgraphs.edge_index[1]]

        return s_edge_index, s_edge_mask.float(), 0

    def sample_connected_masks(self, support_subgraphs, kk=10): 
        batch = support_subgraphs.batch
        device = self.device
        num_nodes = scatter_sum(torch.ones(batch.shape).to(batch.device), batch)
        head_idxs = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes[:-1]]), 0).long()
        n_edges = support_subgraphs.edge_index.shape[1]
        n_nodes = support_subgraphs.x.shape[0]
        n_node_in_batch = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),num_nodes]), 0).long()
        support_subgraphs.edge_index = support_subgraphs.edge_index.long()
        all_selected_nodes = []
        for i in range(len(n_node_in_batch)-1):
            if num_nodes[i] < kk:
                all_selected_nodes.append(torch.tensor(np.arange(n_node_in_batch[i].item(), n_node_in_batch[i+1].item()), device=device))
            else:
                h0 = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                h0[n_node_in_batch[i], 0] = 1.
                h1 = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                h1[n_node_in_batch[i] + 1, 0] = 1.
                
                num_left = np.random.randint(1, 3)
                num_right = np.random.randint(1, 3)
                if num_left >= 1:
                    h0_1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h0)
                    h0_2 = spmm(support_subgraphs.edge_index[[1,0]], torch.ones(n_edges, device=device), n_nodes, n_nodes, h0)
                
                    prob = torch.clamp((h0_1+h0_2)[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze(), 0, 1)
                    
                    prob[0] = 0
                    prob[1] = 0
                    if (prob == 1).any():
                        n_connected = int(prob.sum().item())
                        if num_left == 1:
                            selected = torch.multinomial(prob, min(50, n_connected))
                        elif num_left == 2:
                            selected = torch.multinomial(prob, min(25, n_connected))
                        selected += n_node_in_batch[i]
                        
                        h1_selected = torch.tensor(selected, device=device)

                        all_selected_nodes.append(h1_selected)
                        
                        if num_left == 2:
                            h0_3 = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                            h0_3[h1_selected, 0] = 1.
                            
                            h0_3_1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h0_3)
                            h0_3_2 = spmm(support_subgraphs.edge_index[[1,0]], torch.ones(n_edges, device=device), n_nodes, n_nodes, h0_3)
                            
                            prob = torch.clamp((h0_3_1+h0_3_2)[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze(), 0, 1)
                            
                            prob[0] = 0
                            prob[1] = 0
                            if (prob == 1).any():
                                n_connected = int(prob.sum().item())
                                selected = torch.multinomial(prob, min(25, n_connected))
                                selected += n_node_in_batch[i]

                                h1_selected = torch.tensor(selected, device=device)

                                all_selected_nodes.append(h1_selected)
                    
                if num_right >= 1:
                    h0_1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h1)
                    h0_2 = spmm(support_subgraphs.edge_index[[1,0]], torch.ones(n_edges, device=device), n_nodes, n_nodes, h1)
                
                    prob = torch.clamp((h0_1+h0_2)[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze(), 0, 1)
                    
                    prob[0] = 0
                    prob[1] = 0
                    if (prob == 1).any():
                        n_connected = int(prob.sum().item())

                        if num_left == 1:
                            selected = torch.multinomial(prob, min(50, n_connected))
                        elif num_left == 2:
                            selected = torch.multinomial(prob, min(25, n_connected))
                        selected += n_node_in_batch[i]
                        
                        h1_selected = torch.tensor(selected, device=device)

                        all_selected_nodes.append(h1_selected)
                        
                        if num_left == 2:
                            h0_3 = torch.zeros(support_subgraphs.x.shape[0], 1, device=device)
                            h0_3[h1_selected, 0] = 1.
                            
                            h0_3_1 = spmm(support_subgraphs.edge_index, torch.ones(n_edges, device=device), n_nodes, n_nodes, h0_3)
                            h0_3_2 = spmm(support_subgraphs.edge_index[[1,0]], torch.ones(n_edges, device=device), n_nodes, n_nodes, h0_3)
                            
                            prob = torch.clamp((h0_3_1+h0_3_2)[n_node_in_batch[i]:n_node_in_batch[i+1]].squeeze(), 0, 1)
                            
                            prob[0] = 0
                            prob[1] = 0
                            if (prob == 1).any():
                                n_connected = int(prob.sum().item())

                                selected = torch.multinomial(prob, min(25, n_connected))
                                selected += n_node_in_batch[i]

                                h1_selected = torch.tensor(selected, device=device)

                                all_selected_nodes.append(h1_selected)
                    
                all_selected_nodes.append(torch.tensor([n_node_in_batch[i], n_node_in_batch[i]+1], device=device))
        all_selected_nodes = torch.cat(all_selected_nodes)
        s_edge_index, s_edge_attr = subgraph(all_selected_nodes, support_subgraphs.edge_index)
        node_mask = all_selected_nodes.new_zeros(support_subgraphs.x.shape[0], dtype=torch.bool)
        node_mask[all_selected_nodes] = True
        s_edge_mask = node_mask[support_subgraphs.edge_index[0]] & node_mask[support_subgraphs.edge_index[1]]

        return s_edge_index, s_edge_mask.float(), 0

    def cycle_consistency(self, task):
        support, support_subgraphs, support_negative, support_negative_subgraphs, query, query_subgraphs, negative, negative_subgraphs = task
        
        support_subgraphs, support_negative_subgraphs, query_subgraphs, negative_subgraphs = support_subgraphs.to(self.device), support_negative_subgraphs.to(self.device), query_subgraphs.to(self.device), negative_subgraphs.to(self.device)
        
        _, masks, _ = self.sample_connected_masks(support_subgraphs, kk=50)
        graph_emb_gt, extra_loss, _ = self.embedding_learner.masked_embedding(support_subgraphs, masks, size_loss_beta=0)

        reconstructed_masks = self.embedding_learner.gen_mask_gnn(support_subgraphs, graph_emb_gt)

        graph_emb, loss, edgemask = self.embedding_learner.get_masked_graph_embedding(support_subgraphs, graph_emb_gt, size_loss_beta = 0)
        graph_emb_neg, loss_neg, edgemask_neg = self.embedding_learner.get_masked_graph_embedding(support_negative_subgraphs, graph_emb_gt, size_loss_beta= 0)  
        
        p_score = nn.CosineSimilarity(dim = 1)(graph_emb[:, :self.prototype_dim], graph_emb_gt[:, :self.prototype_dim]) 
        n_score = nn.CosineSimilarity(dim = 1)(graph_emb_neg[:, :self.prototype_dim], graph_emb_gt[:, :self.prototype_dim]) 

        return masks, reconstructed_masks, p_score, n_score
    
    def cycle_loss_func(self, masks, reconstructed_masks):
        if self.loss_mode == 'inverse':
            ratio = torch.sum((masks==0).double()) / torch.sum((masks==1).double()).item()
        elif self.loss_mode == 'inverse-sqrt':
            ratio = torch.sqrt(torch.sum((masks==0).double()) / torch.sum((masks==1).double()).item())
        elif self.loss_mode == 'inverse-log':
            ratio = torch.log(torch.sum((masks==0).double()) / torch.sum((masks==1).double()).item() + 1e-7)
        elif self.loss_mode == 'normal':
            ratio = 1.
        weight = torch.where(masks == 1, ratio, 1.).float()
        return nn.BCELoss(weight = weight)(reconstructed_masks, masks)

    def forward(self, task, iseval=False, is_eval_loss = False, curr_rel='', trial = None, best_params = None):
        support, support_subgraphs, support_negative, support_negative_subgraphs, query, query_subgraphs, negative, negative_subgraphs = task


        batch_size = len(support)
        few = len(support[0])              # num of few
        num_sn = len(support_negative[0])  # num of support negative
        num_q = len(query[0])              # num of query
        num_n = len(negative[0])           # num of query negative

        support_subgraphs, support_negative_subgraphs, query_subgraphs, negative_subgraphs = support_subgraphs.to(self.device), support_negative_subgraphs.to(self.device), query_subgraphs.to(self.device), negative_subgraphs.to(self.device)
                    
        
        if self.use_atten:
            ##  CSR-GNN #################
            if not self.use_ground_truth:
                if self.opt_mode == 'no_decode_share': 
                    row, col = support_subgraphs.edge_index
                    edge_batch = support_subgraphs.batch[row]

                    graph_emb, _, edge_attr = self.embedding_learner.rgcn(support_subgraphs)
                    for i in range(self.niters):
                        graph_emb = graph_emb.reshape(batch_size, few, -1)
            
                        graph_emb_permute = graph_emb.clone()
                        graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([1,2,0]).to(graph_emb_permute.device))
                        graph_emb_permute = graph_emb_permute.reshape(batch_size*few, -1)
                         
                        prototype = graph_emb_permute
                        graph_emb, _, edge_attr = self.embedding_learner.egnn(support_subgraphs, extra_cond = prototype[edge_batch] )


                    h = self.embedding_learner.egnn_post_layers(edge_attr)
                    edge_mask = h.sigmoid().reshape(-1)[: support_subgraphs.edge_index.shape[1]]

                    graph_emb, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask, size_loss_beta = 0)
                    graph_emb = graph_emb.reshape(batch_size, few, -1)
                    rel_q = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
                    edge_mask_q = edge_mask
                    
                if self.opt_mode == 'no_decode': 
                    ####### connect 3 graphs ##########
                    edge_mask = self.embedding_learner.gen_common_sg_mask_gnn(support_subgraphs)
                    graph_emb, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask, size_loss_beta = 0)
                    graph_emb = graph_emb.reshape(batch_size, few, -1)
                    rel_q = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
                    edge_mask_q = edge_mask
                                    
                if self.opt_mode == 'iters_of_perm_min_end':
                    ####### min 2 edge masks (working version) ##########
                    n_iters = self.niters
                    graph_emb, _, _ = self.embedding_learner.rgcn(support_subgraphs)

                    pos_distances_all = []
                    pair_distances_all = []
                    size_loss_all = []
                    
                    for i in range(n_iters):
                        graph_emb = graph_emb.reshape(batch_size, few, -1)

                        graph_emb_permute = graph_emb.clone()
                        graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([1,2,0]).to(graph_emb_permute.device))
                        graph_emb_permute = graph_emb_permute.reshape(batch_size*few, -1)
                        
                        graph_emb, loss, edgemask1 = self.embedding_learner.get_masked_graph_embedding(support_subgraphs, graph_emb_permute, size_loss_beta = 0)  
                        

                    graph_emb, _, _ = self.embedding_learner.rgcn(support_subgraphs)

                    for i in range(n_iters):
                        graph_emb = graph_emb.reshape(batch_size, few, -1)

                        graph_emb_permute = graph_emb.clone()
                        graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([2,0,1]).to(graph_emb_permute.device))
                        graph_emb_permute = graph_emb_permute.reshape(batch_size*few, -1)
                        
                        
                        graph_emb, loss, edgemask2 = self.embedding_learner.get_masked_graph_embedding(support_subgraphs, graph_emb_permute, size_loss_beta = 0)  
                    
                    edge_mask = torch.minimum(edgemask1, edgemask2)
                    
                    graph_emb, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask, size_loss_beta = 0)
                    graph_emb = graph_emb.reshape(batch_size, few, -1)
                    rel_q = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
                    edge_mask_q = edge_mask

                if self.opt_mode == 'iters_of_perm_and_min':
                    ####### min 2 edge masks every iter ##########
                    n_iters = self.niters
                    graph_emb, _, _ = self.embedding_learner.rgcn(support_subgraphs)

                    pos_distances_all = []
                    pair_distances_all = []
                    
                    for i in range(n_iters):
                        graph_emb = graph_emb.reshape(batch_size, few, -1)

                        graph_emb_permute = graph_emb.clone()
                        graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([1,2,0]).to(graph_emb_permute.device))
                        graph_emb_permute = graph_emb_permute.reshape(batch_size*few, -1)

                        _, loss, edgemask1 = self.embedding_learner.get_masked_graph_embedding(support_subgraphs, graph_emb_permute, size_loss_beta = 0)  

                        graph_emb_permute = graph_emb.clone()
                        graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([2,0,1]).to(graph_emb_permute.device))
                        graph_emb_permute = graph_emb_permute.reshape(batch_size*few, -1)
                        _, loss, edgemask2 = self.embedding_learner.get_masked_graph_embedding(support_subgraphs, graph_emb_permute, size_loss_beta = 0)  
                        
                        edge_mask = torch.minimum(edgemask1, edgemask2)
                        graph_emb, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask, size_loss_beta = 0)
                    
                    graph_emb = graph_emb.reshape(batch_size, few, -1)
                    rel_q = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
                    edge_mask_q = edge_mask


                if self.opt_mode == 'iters_3_min_end':
                    ####### min 3 edge masks (probably working version) ##########
                    n_iters = self.niters
                    graph_emb, _, _ = self.embedding_learner.rgcn(support_subgraphs)

                    pos_distances_all = []
                    pair_distances_all = []
                    for i in range(n_iters):
                        graph_emb = graph_emb.reshape(batch_size, few, -1)

                        graph_emb_permute = graph_emb.clone()
                        graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([2,0,1]).to(graph_emb_permute.device))
                        graph_emb_permute = graph_emb_permute.reshape(batch_size*few, -1)
                        graph_emb, loss, edgemask2 = self.embedding_learner.get_masked_graph_embedding(support_subgraphs, graph_emb_permute, size_loss_beta = 0)  
                        
                    graph_emb = graph_emb.reshape(batch_size, few, -1)

                    graph_emb1 = graph_emb[:,0,:].view(batch_size, 1, -1)
                    pos_distances, neg_distances, extra_loss, edge_mask1, _, _,  graph_emb_neg = self.embedding_learner(support_subgraphs, support_negative_subgraphs, graph_emb1, few, num_sn, None, None, size_loss_beta = 0)

                    graph_emb2 = graph_emb[:,1,:].view(batch_size, 1, -1)
                    pos_distances, neg_distances, extra_loss, edge_mask2, _, _,  graph_emb_neg = self.embedding_learner(support_subgraphs, support_negative_subgraphs, graph_emb2, few, num_sn, None, None, size_loss_beta = 0)

                    graph_emb3 = graph_emb[:,2,:].view(batch_size, 1, -1)
                    pos_distances, neg_distances, extra_loss, edge_mask3, _, _,  graph_emb_neg = self.embedding_learner(support_subgraphs, support_negative_subgraphs, graph_emb3, few, num_sn, None, None, size_loss_beta = 0)
                    edge_mask = torch.min(torch.stack([edge_mask1, edge_mask2, edge_mask3],1),dim =  1)[0]
                        

                    graph_emb, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask, size_loss_beta = 0)
                    graph_emb = graph_emb.reshape(batch_size, few, -1)
                    rel_q = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
                    edge_mask_q = edge_mask
                
                    
                if self.support_only:
                    if self.opt_mask:
                        
                        _, _, _, _, edge_mask_neg, _,  _ = self.embedding_learner(support_subgraphs, support_negative_subgraphs, rel_q, few, num_sn, None, None, size_loss_beta = 0)
                        rule_mask = support_subgraphs.rule_mask.to(self.device)
                        loss = self.cycle_loss_func(rule_mask, edge_mask) 
                        extra_loss = - torch.sum(edge_mask) + extra_loss
                        return loss + extra_loss * self.extra_loss_beta, extra_loss , edge_mask, edge_mask_neg, 0, 0
                    
    
                    pos_distances, neg_distances, _, _, edge_mask_neg, _,  _ = self.embedding_learner(support_subgraphs, support_negative_subgraphs, rel_q, few, num_sn, None, None, size_loss_beta = 0)
                    graph_emb, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask_q, size_loss_beta = 0)
        
                    
                    rule_mask = support_subgraphs.rule_mask.to(self.device)
                    rel_gt = self.embedding_learner.masked_embedding(support_subgraphs, rule_mask)[0].view(batch_size, few, -1).mean(1).view(batch_size, 1, -1)

                    graph_emb_permute = graph_emb.clone()
                    graph_emb_permute = graph_emb_permute.reshape(batch_size, few, -1)
                    graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([1,2,0]).to(graph_emb_permute.device))
                    graph_emb_permute = graph_emb_permute.reshape(batch_size * few, -1)

                    pair_distances  = - nn.CosineSimilarity(dim = 1)(graph_emb, graph_emb_permute)
                    
                    sup_distances = - nn.CosineSimilarity(dim = 1)(rel_gt.reshape(batch_size, -1), rel_q.reshape(batch_size, -1)) 
                    print(sup_distances.mean(), pair_distances.mean())
                    pos_distances = pair_distances
                    extra_loss = - torch.sum(edge_mask) + extra_loss 
                    
                    return -pos_distances, -neg_distances, extra_loss * self.extra_loss_beta, edge_mask, edge_mask_neg
     
            else:
                # ground truth subgraph
                if self.use_full_mask_rule:
                    rule_mask = torch.ones(support_subgraphs.edge_attr.shape).to(self.device)
                    rel_q = self.embedding_learner.masked_embedding(support_subgraphs, rule_mask)[0].view(batch_size, few, -1).mean(1).view(batch_size, 1, -1)
                else:
                    rule_mask = query_subgraphs.rule_mask.to(self.device)
                    rel_q = self.embedding_learner.masked_embedding(query_subgraphs, rule_mask)[0].view(batch_size, num_q, -1).mean(1).view(batch_size, 1, -1)
            
            if self.joint_train_mask:
                rule_mask = query_subgraphs.rule_mask.to(self.device)
                rel_q = self.embedding_learner.masked_embedding(query_subgraphs, rule_mask)[0].view(batch_size, num_q, -1).mean(1).view(batch_size, 1, -1)
                    
            if not self.use_full_mask_query:        
                pm, nm = None, None
                pos_distances, neg_distances, extra_loss, edgemask, edge_mask_neg, _, _ = self.embedding_learner(query_subgraphs, negative_subgraphs, rel_q, num_q, num_n, pm, nm, size_loss_beta = 0)
            else:
                pm = torch.ones(query_subgraphs.edge_index.shape[1]).to(self.device)
                nm = torch.ones(negative_subgraphs.edge_index.shape[1]).to(self.device)
                pos_distances, neg_distances, extra_loss, edgemask, edge_mask_neg, _, _ = self.embedding_learner(query_subgraphs, negative_subgraphs, rel_q, num_q, num_n, pm, nm, size_loss_beta = 0)
            
    
            if self.joint_train_mask:
                ## end 2 end
                if self.opt_mask:
                    ### loss from support stage
                    _, extra_loss, _  = self.embedding_learner.masked_embedding(support_subgraphs, edge_mask_q, size_loss_beta = 0)
                    rule_mask = support_subgraphs.rule_mask.to(self.device)
                    loss_support = self.cycle_loss_func(rule_mask, edge_mask_q) 
                    extra_loss = - torch.sum(edge_mask_q) + extra_loss

                    rule_mask = query_subgraphs.rule_mask.to(self.device)
                    loss_query = self.cycle_loss_func(rule_mask, edgemask)
                    if not is_eval_loss:     
                        print("support:")
                        print_iou(support_subgraphs, edge_mask_q, print_all = False)

                    return loss_support + loss_query + extra_loss * self.extra_loss_beta, extra_loss , edgemask, edge_mask_neg, - pos_distances, -neg_distances
                else:
                    raise "Need opt_mask"
            else:
                if self.opt_mask:
                    rule_mask = query_subgraphs.rule_mask.to(self.device)
                    loss_query = self.cycle_loss_func(rule_mask, edgemask)
                    return loss_query + extra_loss * self.extra_loss_beta, extra_loss , edgemask, edge_mask_neg, - pos_distances, -neg_distances
            return - pos_distances, -neg_distances, extra_loss * self.extra_loss_beta , edgemask, edge_mask_neg
        
        else:
            ## CSR-OPT #################

            if not self.use_ground_truth:
                if True:                    
                    rel = nn.Parameter(torch.rand((batch_size, self.num_prototypes_per_class, self.prototype_dim)).to(self.device),
                                                          requires_grad=True)
                    edge_mask_pos = torch.nn.Parameter(torch.rand(support_subgraphs.edge_index.shape[1]).to(self.device) * (-10))
                    edge_mask_neg = torch.nn.Parameter(torch.rand(support_negative_subgraphs.edge_index.shape[1]).to(self.device)* (-10) )
                       
                    inner_mask_p = InnerMask(edge_mask_pos)
                    inner_mask_n = InnerMask(edge_mask_neg)
                    inner_rel = InnerRel(rel)
                    intermediate_masks = []
        
                    lambda_coeff = torch.tensor(1.).to(self.device)
                    lambda_coeff.requires_grad = True
                    
                    lambda_coeff2 = torch.tensor(1.).to(self.device)
                    lambda_coeff2.requires_grad = True

                    ## synthetic
                    if SYNTHETIC:
                        if trial is not None:
                            opt_mask_p = torch.optim.AdamW(inner_mask_p.parameters(), lr=trial.suggest_float('lr', 0, 1))
                            n1=1000
                            n2=1
                            n3=1
                            beta = trial.suggest_float('beta', 0, 0.001)
                            beta2 =100
                            size_loss_beta = trial.suggest_float('size_loss_beta', 0, 0.001)
                            connectivity_loss_beta = trial.suggest_float('connectivity_loss_beta', 0, 1)
                            beta3 = 1

                            # regulate for connectivity
                            lambda_coeff_max = 10
                            lambda_coeff2_max = 10

                            epsilon_perf = trial.suggest_float('epsilon_perf', 0.9, 1)
                            epsilon_con = trial.suggest_float('epsilon_con', 0.9, 1)
                        elif best_params is not None:
                            opt_mask_p = torch.optim.AdamW(inner_mask_p.parameters(), lr=best_params['lr'])
                            n1=1000
                            n2=1
                            n3=1
                            beta = best_params['beta']
                            beta2 =100
                            size_loss_beta = best_params['size_loss_beta']
                            connectivity_loss_beta = best_params['connectivity_loss_beta']
                            beta3 = 1
                            # regulate for connectivity
                            lambda_coeff_max = 10
                            lambda_coeff2_max = 10

                            epsilon_perf = best_params['epsilon_perf']
                            epsilon_con = best_params['epsilon_con']    
                        else:
                            opt_mask_p = torch.optim.AdamW(inner_mask_p.parameters(), lr=1)
                            n1=1000
                            n2=1
                            n3=1
                            beta = 0.0007
                            beta2 =100
                            size_loss_beta = 0.0001
                            connectivity_loss_beta = 0.56
                            beta3 = 1
                            # regulate for connectivity
                            lambda_coeff_max = 10
                            lambda_coeff2_max = 10

                            epsilon_perf = 0.99
                            epsilon_con = 0.95
                    else:  
                        if trial is not None:
                            opt_mask_p = torch.optim.AdamW(inner_mask_p.parameters(), lr=trial.suggest_float('lr', 0, 1))
                            n1=300
                            n2=1
                            n3=1
                            beta = trial.suggest_float('beta', 0, 1)
                            beta2 =100
                            size_loss_beta = trial.suggest_float('size_loss_beta', 0, 1)
                            connectivity_loss_beta = trial.suggest_float('connectivity_loss_beta', 0, 1)
                            beta3 = 1
                            # regulate for connectivity
                            lambda_coeff_max = 10
                            lambda_coeff2_max = 10

                            epsilon_perf = trial.suggest_float('epsilon_perf', 0.9, 1)
                            epsilon_con = trial.suggest_float('epsilon_con', 0.9, 1)
                        elif best_params is not None:
                            opt_mask_p = torch.optim.AdamW(inner_mask_p.parameters(), lr=best_params['lr'])
                            n1=300
                            n2=1
                            n3=1
                            beta = best_params['beta']
                            beta2 =100
                            size_loss_beta = best_params['size_loss_beta']
                            connectivity_loss_beta = best_params['connectivity_loss_beta']
                            beta3 = 1
                            # regulate for connectivity
                            lambda_coeff_max = 10
                            lambda_coeff2_max = 10

                            epsilon_perf = best_params['epsilon_perf']
                            epsilon_con = best_params['epsilon_con']
                        else:
                            opt_mask_p = torch.optim.AdamW(inner_mask_p.parameters(), lr=0.7625)
                            n1=300
                            n2=1
                            n3=1
                            beta = 0.3554
                            beta2 =100
                            size_loss_beta = 0.3958
                            connectivity_loss_beta = 0.5734
                            beta3 = 1
                            # regulate for connectivity
                            lambda_coeff_max = 10
                            lambda_coeff2_max = 10

                            epsilon_perf = 0.9864
                            epsilon_con = 0.9391
                    
                    
                    if self.pdb_mode:
                        pdb.set_trace()
                    with torch.enable_grad():
                        for i in range(n1):
                            for j in range(n2):
                                opt_mask_p.zero_grad()
                                pm = inner_mask_p()
                                nm = inner_mask_n()
                                pos_distances, neg_distances, extra_loss, edge_mask, edge_mask_neg, graph_emb, graph_emb_neg = self.embedding_learner(support_subgraphs, support_negative_subgraphs, inner_rel(), few, num_sn, pm.sigmoid(), nm.sigmoid(), size_loss_beta = 0)
                                # permute graph_emb
                                graph_emb_permute = graph_emb.clone()
                                graph_emb_permute = graph_emb_permute.reshape(batch_size, few, -1)
                                graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([1,2,0]).to(graph_emb_permute.device))
                                graph_emb_permute = graph_emb_permute.reshape(batch_size * few, -1)

                                pos_distances  = - nn.CosineSimilarity(dim = 1)(graph_emb, graph_emb_permute)
                                neg_distances  = - nn.CosineSimilarity(dim = 1)(graph_emb, graph_emb_neg)

                                
                                size_loss = torch.sum(edge_mask)
                                connectivity_loss = compute_connectivity_loss(support_subgraphs, edge_mask)
                                connectivity_thresh_loss = (connectivity_loss + epsilon_con).sum()
                                
                                perf_loss = (pos_distances + epsilon_perf).sum()
                                loss = - size_loss * size_loss_beta + extra_loss * beta + connectivity_thresh_loss * lambda_coeff2 * connectivity_loss_beta + lambda_coeff * perf_loss
    
                                if (self.pdb_mode or self.verbose) and not is_eval_loss:
                                    print(i, pos_distances.mean(), neg_distances.mean(), extra_loss, size_loss, connectivity_loss.mean())
                                    print_iou(support_subgraphs, edge_mask, print_all = False)
                                loss.backward()
                                opt_mask_p.step()


                            for j in range(n3):
                                pm = inner_mask_p()
                                nm = inner_mask_n()
                                lambda_coeff = lambda_coeff.detach()
                                lambda_coeff.requires_grad = True
                                lambda_coeff2 = lambda_coeff2.detach()
                                lambda_coeff2.requires_grad = True
                                pos_distances, neg_distances, extra_loss, edge_mask, edge_mask_neg, graph_emb, graph_emb_neg = self.embedding_learner(support_subgraphs, support_negative_subgraphs, inner_rel(), few, num_sn, pm.sigmoid(), nm.sigmoid(), size_loss_beta = 0)

                                graph_emb_permute = graph_emb.clone()
                                graph_emb_permute = graph_emb_permute.reshape(batch_size, few, -1)
                                graph_emb_permute = torch.index_select(graph_emb_permute, 1, torch.LongTensor([1,2,0]).to(graph_emb_permute.device))
                                graph_emb_permute = graph_emb_permute.reshape(batch_size * few, -1)

                                pos_distances  = - nn.CosineSimilarity(dim = 1)(graph_emb, graph_emb_permute)

                                
                                size_loss = torch.sum(edge_mask)
                                connectivity_loss = compute_connectivity_loss(support_subgraphs, edge_mask)
                                connectivity_thresh_loss = (connectivity_loss + epsilon_con).sum()
                                perf_loss = (pos_distances + epsilon_perf).sum()
                                
                                loss = - (- size_loss * size_loss_beta + extra_loss * beta + lambda_coeff * perf_loss +  connectivity_thresh_loss *connectivity_loss_beta* lambda_coeff2)
                                loss.backward()

                                lambda_coeff = lambda_coeff.detach() - lambda_coeff.grad.detach() * beta2
                                lambda_coeff = torch.clamp(lambda_coeff, 0, lambda_coeff_max)
                                lambda_coeff.requires_grad = True
            
                                lambda_coeff2 = lambda_coeff2.detach() - lambda_coeff2.grad.detach() * beta3
                                lambda_coeff2 = torch.clamp(lambda_coeff2, 0, lambda_coeff2_max)
                                lambda_coeff2.requires_grad = True     
                                
                                graph_emb = graph_emb.reshape(batch_size, few, -1)        
                                rel = torch.mean(graph_emb, 1).view(batch_size, 1, -1)
                                
                                if (self.pdb_mode or self.verbose) and not is_eval_loss:
                                    print(lambda_coeff)
                                    print(lambda_coeff2)


                        if (self.pdb_mode or self.verbose) and not is_eval_loss:
                            print(pos_distances)
                            print_iou(support_subgraphs, edge_mask, print_all = True)
                
                rel_q = rel.detach()
                
                if self.support_only:
                    return -pos_distances, -neg_distances, torch.tensor(0), edge_mask, edge_mask_neg


            else:
                # ground truth subgraph
                if self.use_full_mask_rule:
                    rule_mask = torch.ones(support_subgraphs.edge_attr.shape).to(self.device)
                    rel_q = self.embedding_learner.masked_embedding(support_subgraphs, rule_mask)[0].view(batch_size, few, -1).mean(1).view(batch_size, 1, -1)
                else:
                    rule_mask = query_subgraphs.rule_mask.to(self.device)
                    rel_q = self.embedding_learner.masked_embedding(query_subgraphs, rule_mask)[0].view(batch_size, num_q, -1).mean(1).view(batch_size, 1, -1)

            if not self.use_full_mask_query:
                edge_mask_pos = torch.nn.Parameter(torch.rand(query_subgraphs.edge_index.shape[1]).to(self.device)*0.1)
                edge_mask_neg = torch.nn.Parameter(torch.rand(negative_subgraphs.edge_index.shape[1]).to(self.device)*0.1)
                
                intermediate_masks = []

                inner_mask = InnerMasks(edge_mask_pos, edge_mask_neg)
                with torch.enable_grad():

                    n_e = 300
                    
                    opt_mask = torch.optim.AdamW(inner_mask.parameters(), lr=0.5)
                    ## synthetics
                    if trial is not None:
                        beta =  trial.suggest_float('beta_2', 0, 0.001)
                        size_loss_beta = trial.suggest_float('size_loss_beta_2', 0, 5)
                    elif best_params is not None:
                        beta =  best_params['beta_2']
                        size_loss_beta = best_params["size_loss_beta_2"]
                    else:
                        # nell
                        beta = 2e-5
                        size_loss_beta = 4.928
                        
                    if self.pdb_mode:
                        pdb.set_trace() 
                    
                    for i in range(n_e):
                        opt_mask.zero_grad()
                        pm, nm = inner_mask()
                        pos_distances, neg_distances, extra_loss, edge_mask, edge_mask_neg, _, _ = self.embedding_learner(query_subgraphs, negative_subgraphs, rel_q, num_q, num_n, pm.sigmoid(), nm.sigmoid(), size_loss_beta = size_loss_beta)                        

                        loss = (pos_distances.sum() + neg_distances.sum()) + extra_loss * beta
                        loss.backward()
                        opt_mask.step()
                        intermediate_masks.append(inner_mask())
                        
                        if i% 10 ==0 and (self.pdb_mode or self.verbose) and not is_eval_loss:
                            print(i, pos_distances.mean(), neg_distances.mean(), extra_loss)
                            print_iou(query_subgraphs, edge_mask, print_all = False)
                if (self.pdb_mode or self.verbose)and not is_eval_loss:
                    print_iou(query_subgraphs, edge_mask, print_all = True)
                    p_score = - pos_distances.reshape(-1).detach()
                    n_score = - neg_distances.reshape(-1).detach()
                    print(roc_auc_score(torch.cat([torch.ones(p_score.shape), torch.zeros(n_score.shape)]) , torch.cat([p_score, n_score]).cpu() ))

                pm, nm = intermediate_masks[-1]
                pos_distances, neg_distances, extra_loss, edgemask, edge_mask_neg, _, _ = self.embedding_learner(query_subgraphs, negative_subgraphs, rel_q, num_q, num_n, pm.sigmoid(), nm.sigmoid(), size_loss_beta = 0)
            else:
                pm = torch.ones(query_subgraphs.edge_index.shape[1]).to(self.device)*1000000
                nm = torch.ones(negative_subgraphs.edge_index.shape[1]).to(self.device)*1000000
                pos_distances, neg_distances, extra_loss, edgemask, edge_mask_neg, _, _ = self.embedding_learner(query_subgraphs, negative_subgraphs, rel_q, num_q, num_n, pm.sigmoid(), nm.sigmoid(), size_loss_beta = 0)
                
            return -pos_distances, -neg_distances, extra_loss, edgemask, edge_mask_neg

