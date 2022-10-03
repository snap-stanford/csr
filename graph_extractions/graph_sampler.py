import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
# import multiprocessing as mp
import ray.util.multiprocessing as mp
from functools import partial

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, serialize
import networkx as nx
from load_kg_dataset import process_files
import json
import numpy as np
import ray

'''adpated from https://github.com/kkteru/grail/blob/master/subgraph_extraction/graph_sampler.py'''

FIX2 = False

def generate_subgraph_datasets(root, dataset, splits, kind, hop, sample_neg = False, all_negs = False, sample_all_negs = False, all_candidate_negs = False, onek_negs = False, two_hun_negs = False, neg_triplet_as_task = False, subset = None, inductive = False, no_candidates = False):
    raw_data_paths = os.path.join(root, dataset)
    if sample_neg or all_negs:
        rel2candidates = json.load(open(os.path.join(raw_data_paths, 'rel2candidates.json')))
    e1rel_e2 = json.load(open(os.path.join(raw_data_paths, 'e1rel_e2.json')))
    
    postfix = "" if not inductive else "_inductive"
    

    path_graph = json.load(open(os.path.join(raw_data_paths, f"path_graph{postfix}.json")))
        
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(raw_data_paths, inductive = inductive)
    
    links = {}
    print(splits)
    for split_name in splits:
        split = {}
        if no_candidates:
            assert split_name == "test"

        
        tasks = json.load(open(os.path.join(raw_data_paths, split_name + '_tasks.json')))
        if split_name == "pretrain":
            tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks{postfix}.json')))
        if split_name == "train" and inductive:
            tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks{postfix}.json')))
        
        pos = {}
        if not all_negs:
            # don't need to extract pos again normally
            for rel, task in tasks.items():
                t = []
                for e1, rel, e2 in task:
                    try:
                        t.append([entity2id[e1], entity2id[e2]])
                    except:
                        print("nop")
                pos[rel] = t

        
        neg = {}
        if not all_negs:
            if not sample_neg:
                print("reuse negatives")
                tasks = json.load(open(os.path.join(raw_data_paths, split_name + '_tasks_neg.json')))
                if no_candidates:
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_neg_nocandidates.json')))
                if split_name == "train" and inductive:
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_neg{postfix}.json')))
                if split_name == "pretrain":
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_neg{postfix}.json')))
                   
                for rel, task in tasks.items():
                    t = []
                    for e1, rel, e2 in task:
                        try:
                            t.append([entity2id[e1], entity2id[e2]])
                        except:
                            print("nop neg")
                    neg[rel] = t
            else:
                print("sampling negatives")
                d = {}
                count = {}
                #sample 1 true neg for each pos edge
                for rel, task in tqdm(tasks.items()):
                    t = []
                    d[rel] = []
                    for e1, rel, e2 in tqdm(task):           
                        while True:
                            if rel in rel2candidates and not no_candidates:
                                negative = random.choice(rel2candidates[rel])
                                negative_condition = negative not in e1rel_e2[e1 + rel]
                            else:
                                negative = random.choice(list(entity2id.keys()))
                                negative_condition = [e1, rel, negative] not in path_graph
                                
                            if (negative_condition) \
                                    and negative != e2 and negative != e1 and [e1, rel, negative] not in d[rel]:
                                t.append([entity2id[e1], entity2id[negative]]) 
                                d[rel].append([e1,rel, negative])
                                break
                    neg[rel] = t   
                if split_name == "pretrain":    
                    json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_neg{postfix}.json'), "w"))
                elif split_name == "train" and inductive:
                    json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_neg{postfix}.json'), "w"))
                elif no_candidates:
                    json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_neg_nocandidates.json'), "w"))
                else:
                    json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_neg.json'), "w"))
                                
                            
        elif neg_triplet_as_task:
            ## only for 50 negs
            print("50negs (neg_triplet_as_task) ")
            if not sample_all_negs:
                print("reuse negatives")
                tasks = json.load(open(os.path.join(raw_data_paths, split_name + '_tasks_50neg_triplet_as_task.json')))
                for rel_t, task in tasks.items():
                    t = []
                    for e1, rel, e2 in task:
                        try:
                            t.append([entity2id[e1], entity2id[e2]])
                        except:
                            print("nop neg")
                    neg[rel_t] = t
            else:
                print("sampling negatives")
                # for eval and test, generate 50 negs for all pos edge each rel first
                d = {}
                all_triplets = {}
                for rel, task in tqdm(tasks.items()):
                    for e1, rel, e2 in task:
                        t = []
                        d[e1+rel +e2] = []
                        # sample negs among existing ones
                        for r in all_triplets.keys():
                            if len(t) < 50:
                                e1_neg, _, negative = all_triplets[r][0]
                                if e1_neg != e1:
                                    continue
                                if (negative not in e1rel_e2[e1 + rel]) \
                                        and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                    d[e1+rel+e2].append([e1,rel, negative])
                            else:
                                break
                                
                        num_current_negs = len(d[e1+rel+e2])
                        
                        if num_current_negs < 50:
                            # sample new negs 
                            d_e = []        
                            for negative in rel2candidates[rel]:
                                if (negative not in e1rel_e2[e1 + rel]) \
                                        and negative != e2 and negative != e1 and [e1, rel, negative] not in d_e and [e1, rel, negative] not in d[e1+rel +e2]:
                                    d_e.append([e1,rel, negative])
                                    
        #                     print(len(t))
                            indices = np.random.choice(range(len(d_e)), 50 - num_current_negs, replace = False)
                            d[e1+rel+e2] = d[e1+rel+e2] + np.array(d_e)[indices].tolist()
                            for e1,rel, negative in np.array(d_e)[indices].tolist():
                                all_triplets[e1 + negative] = [[e1,rel, negative]]
                                neg[e1+negative] = [[entity2id[e1], entity2id[negative]]]
                    
                json.dump(d,open(os.path.join(raw_data_paths, split_name + '_tasks_50neg.json'), "w"))
                json.dump(all_triplets,open(os.path.join(raw_data_paths, split_name + '_tasks_50neg_triplet_as_task.json'), "w"))
        
        else:
            if all_candidate_negs:
                print("all_candidate_negs")
                if not sample_all_negs:
                    print("reuse negatives")
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + '_tasks_allneg.json')))
                    for rel_t, task in tasks.items():
                        t = []
                        for e1, rel, e2 in task:
                            try:
                                t.append([entity2id[e1], entity2id[e2]])
                            except:
                                print("nop neg")
                        neg[rel_t] = t
                else:
                    print("sampling negatives")
                    # for eval and test, generate 50 negs for all pos edge each rel first
                    d = {}
                    for rel, task in tasks.items():
                        for e1, rel, e2 in task:
                            t = []
                            d[e1+rel +e2] = []
                            # sample all negs for dev and test
                            for negative in rel2candidates[rel]:
                                if (negative not in e1rel_e2[e1 + rel]) \
                                        and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                    t.append([entity2id[e1], entity2id[negative]]) 
                                    d[e1+rel+e2].append([e1,rel, negative])
        #                     print(len(t))
                            neg[e1+rel+e2] = t
                    json.dump(d,open(os.path.join(raw_data_paths, split_name + '_tasks_allneg.json'), "w"))
            if onek_negs:
                print("1000 negs")
                if not sample_all_negs:
                    print("reuse negatives")
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + '_tasks_1000neg.json')))
                    for rel_t, task in tasks.items():
                        t = []
                        for e1, rel, e2 in task:
                            try:
                                t.append([entity2id[e1], entity2id[e2]])
                            except:
                                print("nop neg")
                        neg[rel_t] = t
                else:
                    print("sampling negatives")
                    # for eval and test, generate 50 negs for all pos edge each rel first
                    d = {}
                    for rel, task in tqdm(tasks.items()):
                        for e1, rel, e2 in tqdm(task):
                            t = []
                            d[e1+rel +e2] = []
                            # sample all negs for dev and test
                            if dataset != "FB15K-237":
                                print("candidates")
                                for negative in rel2candidates[rel]:
                                    if (negative not in e1rel_e2[e1 + rel]) \
                                            and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])
            #                     print(len(t))
                                indices = np.random.choice(range(len(t)), min(1000, len(t)), replace = False)
                                neg[e1+rel+e2] = np.array(t)[indices].tolist()
                                d[e1+rel+e2] = np.array(d[e1+rel +e2])[indices].tolist()
                    
                            else:
                                while len(t) < 1000:
                                    negative = random.choice(list(entity2id.keys()))
                                    if (negative not in e1rel_e2[e1 + rel]) \
                                            and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])

                                neg[e1+rel+e2] = t 

                    json.dump(d,open(os.path.join(raw_data_paths, split_name + '_tasks_1000neg.json'), "w"))
            elif two_hun_negs:
                print("200 negs")
                if not sample_all_negs:
                    print("reuse negatives")
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + '_tasks_200neg.json')))
                    for rel_t, task in tasks.items():
                        t = []
                        for e1, rel, e2 in task:
                            try:
                                t.append([entity2id[e1], entity2id[e2]])
                            except:
                                print("nop neg")
                        neg[rel_t] = t
                else:
                    print("sampling negatives")
                    # for eval and test, generate 50 negs for all pos edge each rel first
                    d = {}
                    for rel, task in tqdm(tasks.items()):
                        for e1, rel, e2 in tqdm(task):
                            t = []
                            d[e1+rel +e2] = []
                            if rel in rel2candidates and dataset != "ConceptNet":        
                                # sample all negs for dev and test
                                for negative in rel2candidates[rel]:
                                    if (negative not in e1rel_e2[e1 + rel]) \
                                            and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])
            #                     print(len(t))
                                indices = np.random.choice(range(len(t)), min(200, len(t)), replace = False)
                                neg[e1+rel+e2] = np.array(t)[indices].tolist()
                                d[e1+rel+e2] = np.array(d[e1+rel +e2])[indices].tolist()
                            else:
                                while len(t) < 200:
                                    negative = random.choice(list(entity2id.keys()))
                                    if ([e1, rel, negative] not in path_graph) \
                                            and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])
                                neg[e1+rel+e2] = t 
                    json.dump(d,open(os.path.join(raw_data_paths, split_name + '_tasks_200neg.json'), "w"))     
            else:
                print("50negs")
                if not sample_all_negs:
                    print("reuse negatives")
                    tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg.json')))
                    if split_name == "pretrain":    
                        tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg{postfix}.json')))
                    if split_name == "train" and inductive:
                        tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg{postfix}.json')))
                    if no_candidates:
                        tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg_nocandidates.json')))
                    
                    if dataset == "Wiki":
                        print(f"subset {subset} triplets")
                        tasks = json.load(open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg_subset{subset}.json')))
                    for rel_t, task in tasks.items():
                        t = []
                        for e1, rel, e2 in task:
                            try:
                                t.append([entity2id[e1], entity2id[e2]])
                            except:
                                print("nop neg")
                        neg[rel_t] = t

                else:
                    print("sampling negatives")
                    # for eval and test, generate 50 negs for all pos edge each rel first
                    d = {}
                    for rel, task in tqdm(tasks.items()):
                        for e1, rel, e2 in tqdm(task):
                            t = []
                            d[e1+rel +e2] = []
                            # sample all negs for dev and test                                
                            if rel in rel2candidates and dataset not in ["ConceptNet", "FB15K-237"] and not no_candidates:            
                                for negative in rel2candidates[rel]:
                                    if (negative not in e1rel_e2[e1 + rel]) \
                                                and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])
        #                     print(len(t))
                                indices = np.random.choice(range(len(t)), 50, replace = False)
                                neg[e1+rel+e2] = np.array(t)[indices].tolist()
                                d[e1+rel+e2] = np.array(d[e1+rel +e2])[indices].tolist()
                            elif e1 + rel in e1rel_e2:
                                while len(t) < 50:
                                    negative = random.choice(list(entity2id.keys()))
                                    if (negative not in e1rel_e2[e1 + rel]) \
                                            and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])
                                    
                                neg[e1+rel+e2] = t 
                            
                            else:
                                print("no e1rel_e2")
                                while len(t) < 50:
                                    negative = random.choice(list(entity2id.keys()))
                                    negative_condition = [e1, rel, negative] not in path_graph

                                    if (negative_condition) \
                                            and negative != e2 and negative != e1 and [e1, rel, negative] not in d[e1+rel+e2]:
                                        t.append([entity2id[e1], entity2id[negative]]) 
                                        d[e1+rel+e2].append([e1,rel, negative])
                                neg[e1+rel+e2] = t   
                   
                    
                    if split_name == "pretrain":    
                        json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg{postfix}.json'), "w"))
                    elif split_name == "train" and inductive:
                        json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg{postfix}.json'), "w"))
                    elif no_candidates:
                        json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg_nocandidates.json'), "w"))
                    else:
                        json.dump(d,open(os.path.join(raw_data_paths, split_name + f'_tasks_50neg.json'), "w"))
            
        split['pos'] = pos
        split['neg'] = neg
        
        links[split_name] = split

     
    if dataset == "Wiki":
        postfix += f"_{subset}"
    if all_negs:
        if neg_triplet_as_task:
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix_new_{kind}_50negs_triplet_as_task_hop={hop}' + postfix)  
            
        elif all_candidate_negs:
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix_new_{kind}_allnegs_hop={hop}' + postfix)  
        elif onek_negs:
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix_new_{kind}_1000negs_hop={hop}'+ postfix)  
        elif two_hun_negs:
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix_new_{kind}_200negs_hop={hop}'+ postfix)   
        else:   
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix_new_{kind}_50negs_hop={hop}'+ postfix)  
    else:
        db_path = os.path.join(raw_data_paths, f'subgraphs_fix_new_{kind}_hop={hop}'+ postfix)  
    
    if FIX2:   
        if all_negs:
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix2_new_{kind}_50negs_hop={hop}'+ postfix)  
        else:
            db_path = os.path.join(raw_data_paths, f'subgraphs_fix2_new_{kind}_hop={hop}'+ postfix)  
    print(db_path)
    links2subgraphs(adj_list, links, kind, hop, db_path)


def links2subgraphs(A, links, kind, hop, db_path):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []
    print("avg")
    BYTES_PER_DATUM = 200000
    # BYTES_PER_DATUM = get_average_subgraph_size(100, links['dev']['pos'], A, kind, hop) * 1.5
    print(BYTES_PER_DATUM)

    links_length = 0        
        
    for split_name, split in links.items():
        for rel, task in split['pos'].items():
            links_length += len(task)

        for rel, task in split['neg'].items():
            links_length += len(task)

    map_size = links_length * BYTES_PER_DATUM * 1000

    env = lmdb.open(db_path, map_size=map_size, max_dbs=8)
    

    A_ = ray.put(A)    

    def extraction_helper(A, links_all, r_label_all, g_labels_all, split_env, ids_all, hop, prefix_all):

        thread_n =6000000
        for idx in tqdm(range(0, len(links_all), thread_n), leave = True):
            
            end = idx+thread_n
            if end > len(links_all):
                end = len(links_all)
            ids = ids_all[idx:end]
            links = links_all[idx:end]
            r_label = r_label_all[idx:end]
            g_labels = g_labels_all[idx:end]
            prefix = prefix_all[idx:end]                

            with mp.Pool(processes=None) as p:
                args_ = zip(ids, links, r_label,g_labels, [kind] *len(links), [hop] *len(links), prefix, [A_] * len(links))
                
                for (str_id, datum) in tqdm(p.imap_unordered(extract_save_subgraph, list(args_)), total=len(links), leave = True):
                    max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                    subgraph_sizes.append(datum['subgraph_size'])
                    enc_ratios.append(datum['enc_ratio'])
                    num_pruned_nodes.append(datum['num_pruned_nodes'])

                    with env.begin(write=True, db=split_env) as txn:
                        txn.put(str_id, serialize(datum))   
                        
    
                  
    for split_name, split in links.items():
        print(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        ls = []
        rs = []
        prefix = []
        ids = []
        count = 0 
        with env.begin(write=False, db=split_env) as txn:
            for rel, task in split['pos'].items(): 
                
#                 missing = False
#                 for idx in range(len(task)):
#                     str_id = (rel).encode() + '{:08}'.format(idx).encode('ascii')
#                     if txn.get(str_id) is None: 
#                         missing = True
#                         break
#                 if not missing:
#                     print(rel, "already exists")
#                     continue                      
                
                ls.extend(task)
                rs.extend([rel] * len(task))
                prefix.extend([rel] * len(task))
                ids.extend(list(range(len(task))))
                count += len(task)
            labels= np.ones(count)
        extraction_helper(A, ls, rs, labels, split_env, ids, hop, prefix)
        
        print(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        ls = []
        rs = []
        prefix = []
        ids = []
        count = 0 
        with env.begin(write=False, db=split_env) as txn:
            for rel, task in split['neg'].items(): 
                
                # more finegrained missing
                missing_ids = list(range(len(task)))
#                 missing_ids = []
#                 missing = False
#                 for idx in range(len(task)):
#                     str_id = (rel).encode() + '{:08}'.format(idx).encode('ascii')
#                     if txn.get(str_id) is None: 
#                         missing = True
#                         missing_ids.append(idx)
# #                         break
#                 if not missing:
#                     print(rel, "already exists")
#                     continue                        
                
                ls.extend(np.array(task)[missing_ids].tolist())
                rs.extend([rel] * len(missing_ids))
                prefix.extend([rel] * len(missing_ids))
                ids.extend(np.array(list(range(len(task))))[missing_ids].tolist())
                count += len(task)
            labels= np.ones(count)
        print(count)
        extraction_helper(A, ls, rs, labels, split_env, ids, hop, prefix)

 
    max_n_label['value'] = max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_subgraph_size(sample_size, pos, A, kind, hop):
    total_size = 0
    for r_label in np.random.choice(list(pos.keys()), sample_size):
        tasks = pos[r_label]
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling(tasks[np.random.choice(len(tasks))], A, kind, hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A):
    global A_
    A_ = A
    
def extract_save_subgraph(args_):
    idx, (n1, n2), r_label, g_label, kind, hop, prefix, A_ = args_
    A_ = ray.get(A_)
    str_id = '{:08}'.format(idx).encode('ascii')

    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), A_, kind, hop)

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}

    return (prefix.encode() + str_id, datum)



def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, A_list, kind, h=1, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    if ind[0] in subgraph_nei_nodes_int:
        subgraph_nei_nodes_int.remove(ind[0])
    if ind[1] in subgraph_nei_nodes_int:
        subgraph_nei_nodes_int.remove(ind[1])
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    if ind[0] in subgraph_nei_nodes_un:
        subgraph_nei_nodes_un.remove(ind[0])
    if ind[1] in subgraph_nei_nodes_un:
        subgraph_nei_nodes_un.remove(ind[1])

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if kind == "intersection":
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
        
        
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    if kind == "union_prune" or kind == "union_prune_plus": 
        while len(enclosing_subgraph_nodes) != len(subgraph_nodes):
            subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes]
            subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
            labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
            
        pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes]
        pruned_labels = labels[enclosing_subgraph_nodes]            
    else:
        pruned_subgraph_nodes = subgraph_nodes
        pruned_labels = labels
        
    if kind == "union_prune_plus":
        if not FIX2:
            root1_nei_1 = get_neighbor_nodes(set([ind[0]]), A_incidence, 1, 50)
            root2_nei_1 = get_neighbor_nodes(set([ind[1]]), A_incidence, 1, 50)
        else:
            root1_nei_1 = get_neighbor_nodes(set([ind[0]]), A_incidence, 2, 50)
            root2_nei_1 = get_neighbor_nodes(set([ind[1]]), A_incidence, 2, 50)
        
        root1_nei_1 = root1_nei_1 - set(pruned_subgraph_nodes)
        root2_nei_1 = root2_nei_1 - set(pruned_subgraph_nodes) - root1_nei_1
        
        pruned_subgraph_nodes_after = np.array(list(pruned_subgraph_nodes) + list(root1_nei_1) + list(root2_nei_1))
        pruned_labels_after = np.zeros((len(pruned_subgraph_nodes_after), 2)) 
        pruned_labels_after[:len(pruned_subgraph_nodes)] = pruned_labels
        pruned_labels_after[len(pruned_subgraph_nodes): len(pruned_subgraph_nodes)+ len(root1_nei_1)] = [1, h]
        pruned_labels_after[len(pruned_subgraph_nodes)+ len(root1_nei_1):] = [h, 1]
        
        pruned_subgraph_nodes = pruned_subgraph_nodes_after
        pruned_labels = pruned_labels_after
    
    

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [1, 0]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


if __name__ == "__main__":
    ray.init()

    # run export PYTHONPATH=.:$PYTHONPATH if there is dependency issue
    # the code bellow will extract subgraphs for transductive setting

    # for inductive, set inductive = True
    # set sample_neg/sample_all_negs = True to resample negatives
    # by default, all cores are used in parallel; you can change this on L459

    # after the subgraph extraction is completed, run SubgraphFewshotDataset in load_kg_dataset 
    # with preprocess/preprocess_50negs = True to pre cache the dataset (generate the preprocessed_* dirs)
    # e.g. SubgraphFewshotDataset(".", shot = 3, dataset="NELL", mode="pretrain", kind="union_prune_plus", hop=2, preprocess = True, preprocess_50negs = True)

    # generate_subgraph_datasets(".", dataset="NELL", splits = ['pretrain', 'dev','test'], kind = "union_prune_plus", hop=2, sample_neg = False)
    # generate_subgraph_datasets(".", dataset="NELL", splits = ['dev','test'], kind = "union_prune_plus", hop=2, all_negs = False, sample_all_negs = False)

    # generate_subgraph_datasets(".", dataset="FB15K-237", splits = ['pretrain', 'dev','test'], kind = "union_prune_plus", hop=1, sample_neg = False)
    # generate_subgraph_datasets(".", dataset="FB15K-237", splits = ['dev','test'], kind = "union_prune_plus", hop=1, all_negs = True, sample_all_negs = False)

    # generate_subgraph_datasets(".", dataset="ConceptNet", splits = ['pretrain', 'dev','test'], kind = "union_prune_plus", hop=1, sample_neg = False)
    # generate_subgraph_datasets(".", dataset="ConceptNet", splits = ['dev','test'], kind = "union_prune_plus", hop=1, all_negs = True, sample_all_negs = False)
