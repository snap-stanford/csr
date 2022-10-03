import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv, FastRGCNConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.pool.topk_pool import topk

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import MetaLayer
import json
import numpy as np
import pickle

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)

class PathCon(torch.nn.Module):
    def __init__(self, input_dim, output_dim, node_dim = 6):
        super().__init__()
        self.edge_mlp = nn.Linear(input_dim * 3 + node_dim * 2, output_dim)
        nn.init.xavier_uniform_(self.edge_mlp.weight)
 

    def forward(self, x, num_nodes, edge_index, edge_attr, mask):
        row = edge_index[0].long()
        col = edge_index[1].long()
   
        ## sum edges around node
        
        node_rep = scatter_sum(edge_attr * mask.unsqueeze(-1), col, dim=0, dim_size=num_nodes)
        node_rep = node_rep/(scatter_sum(mask.unsqueeze(-1), col, dim=0, dim_size=num_nodes) + 1)
        node_rep = torch.cat([node_rep, x], 1)
        edge_rep = torch.cat([node_rep[row], node_rep[col], edge_attr], -1)
        edge_rep = self.edge_mlp(edge_rep)

        return node_rep, edge_rep


class PathConFFN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128, node_dim = 6):
        super().__init__()        
        self.edge_mlp = nn.Linear(input_dim * 3 + 12, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim*2)
        self.linear2 = nn.Linear(output_dim*2, output_dim)
        self.output_dim = output_dim
        self.norm1 = nn.LayerNorm(output_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(output_dim, eps=1e-5)
 

    def forward(self, x, num_nodes, edge_index, edge_attr, mask):
        row = edge_index[0]
        col = edge_index[1]
           
        
        ## sum edges around node
        node_rep = scatter_sum(edge_attr * mask.unsqueeze(-1), col, dim=0, dim_size=num_nodes)
        node_rep = node_rep/(scatter_sum(mask.unsqueeze(-1), col, dim=0, dim_size=num_nodes) + 1)
        
        node_rep = torch.cat([node_rep, x], 1)
        edge_rep = torch.cat([node_rep[row], node_rep[col], edge_attr], -1)
    
        x = self.norm1(edge_attr[:, :self.output_dim] + self.edge_mlp(edge_rep))
        edge_rep = self.norm2(x + self.linear2(F.relu(self.linear1(x))))
            
        
        return node_rep, edge_rep


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


def normalize_emb(data, node_embedding):
    batch = data.batch
    device = batch.device
    num_nodes = scatter_sum(torch.ones(batch.shape).to(device), batch)
    head_idxs = torch.cumsum(torch.cat([torch.tensor([0], device=device), num_nodes[:-1]]), 0).long()
    x_embed = node_embedding[data.x_id]
    head_embed = node_embedding[data.x_id[head_idxs]]
    all_diff = []
    for i, num in enumerate(num_nodes):
        all_diff.append(head_embed[i].repeat(int(num.item()), 1))
    all_diff = torch.cat(all_diff, 0)
    x_embed -= all_diff
    assert (x_embed[head_idxs] == 0).all()
    return x_embed


# GCN
class RGCNNet(nn.Module):
    def __init__(self, emb_dim, input_dim, num_rels_bg, edge_embedding, node_embedding, latent_dim = [128, 128, 128], ffn = False, use_node_emb = False, use_noid_node_emb = False,use_node_emb_end = False, normalize_node_emb = False):
        super(RGCNNet, self).__init__()
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.num_gnn_layers = len(self.latent_dim)
        self.dense_dim = self.latent_dim[-1]
        self.readout_layers = get_readout_layers('max')
        self.num_rels_bg = num_rels_bg
        self.edge_embedding = edge_embedding 
        self.node_embedding = node_embedding 
        self.use_node_emb = use_node_emb
        self.use_noid_node_emb = use_noid_node_emb
        self.normalize_node_emb = normalize_node_emb
        self.pair_node_mlp = nn.Linear(2 * emb_dim , 2*emb_dim)

        self.use_node_emb_end = use_node_emb_end
        
        node_dim = 6
        if self.use_node_emb:
            node_dim = emb_dim + 6
        self.gnn_layers = nn.ModuleList()

        if ffn:
            self.gnn_layers.append(PathConFFN(int(input_dim), self.latent_dim[0], node_dim))
            for i in range(1, self.num_gnn_layers):
                self.gnn_layers.append(PathConFFN(self.latent_dim[i - 1], self.latent_dim[i], node_dim))
            
        else:
            self.gnn_layers.append(PathCon(int(input_dim), self.latent_dim[0], node_dim))
            for i in range(1, self.num_gnn_layers):
                self.gnn_layers.append(PathCon(self.latent_dim[i - 1], self.latent_dim[i], node_dim))

            
            
        self.gnn_non_linear = nn.ReLU()


    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance+1) / (distance + self.epsilon))
        return similarity, distance

    def prototype_subgraph_distances(self, x, prototype):
        distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def forward(self, data, mask = None,  extra_cond = None, protgnn_plus=False, similarity=None):
        
        if protgnn_plus:
            logits = self.last_layer(similarity)
            probs = self.Softmax(logits)
            return logits, probs, None, None, None

        x_orig, edge_index, edge_attr, batch, num_nodes = data.x, data.edge_index.long(), data.edge_attr, data.batch, data.num_nodes

        batch_num_nodes = scatter_sum(torch.ones(batch.shape).to(batch.device), batch)
        head_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),batch_num_nodes[:-1]]), 0).long()
        tail_idxs = torch.cumsum(torch.cat([torch.tensor([0]).to(batch.device),batch_num_nodes[:-1]]), 0).long() + 1

        if mask is None:
            mask = torch.ones(edge_attr.shape[0]).to(x_orig.device)
        
        x = torch.zeros(x_orig.shape).to(x_orig.device)
        x[head_idxs] = x_orig[head_idxs]  
        x[tail_idxs] = x_orig[tail_idxs]         
        
        if self.normalize_node_emb:
            batch_embedding = normalize_emb(data, self.node_embedding)
        else:
            batch_embedding = self.node_embedding(data.x_id)
                    
        if self.use_node_emb:
            if self.use_noid_node_emb:
                x = torch.cat([x,  torch.rand([x_orig.shape[0], self.emb_dim]).to(x_orig.device)], 1)
            else:
                x = torch.cat([x, batch_embedding], 1)
        
        # translate to embedding    
        if len(edge_attr.shape) == 1:
            edge_attr = self.edge_embedding(edge_attr.long())
        if extra_cond is not None:
            edge_attr = torch.cat([edge_attr, extra_cond],1)
        for i in range(self.num_gnn_layers):
            node_rep, edge_attr = self.gnn_layers[i](x, num_nodes, edge_index, edge_attr, mask)
            edge_attr = self.gnn_non_linear(edge_attr)

        
        col = edge_index[1].long()
        node_rep = scatter_sum(edge_attr * mask.unsqueeze(1), col, dim=0, dim_size=num_nodes)
        node_rep = node_rep/(scatter_sum(mask.unsqueeze(1), col, dim=0, dim_size=num_nodes) + 1)

        
        head_emb = node_rep[head_idxs]  
        tail_emb = node_rep[tail_idxs]          
    
        if batch is None:
            pooled = []
            pooled.append(node_rep.mean(0, keepdim=True).repeat(head_idxs.shape[0], 1))
            
            pooled.append(head_emb)
            pooled.append(tail_emb)
            graph_emb = torch.cat(pooled, dim=-1)   
            return graph_emb, (None, None), edge_attr
    
        if node_rep.shape[0] != batch.shape[0]:
            return None, node_rep, edge_attr
        
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(node_rep, batch))

        pooled.append(head_emb)
        pooled.append(tail_emb)
        graph_emb = torch.cat(pooled, dim=-1)        
        
        if self.use_node_emb_end:            
            graph_emb = torch.cat([graph_emb, batch_embedding[head_idxs] , batch_embedding[tail_idxs]], 1)
        
        return graph_emb, node_rep, edge_attr
