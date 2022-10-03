from models import *
import wandb
import os
import sys
import torch
import shutil
import logging
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import transformers
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pdb
import numpy as np
from load_kg_dataset import PairSubgraphsFewShotDataLoader
import torch.nn.functional as F
import optuna

class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.dev_data_loader_ranktail = data_loaders[3]
        self.test_data_loader_ranktail = data_loaders[4]
        self.pretrain_data_loader = data_loaders[5]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.niters = parameter['niters']
        self.threshold = parameter['threshold']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']
        self.coefficient = parameter['coefficient']
        self.coefficient2 = parameter['coefficient2']
        self.finetune = parameter['finetune']
        self.finetune_on_train = parameter['finetune_on_train']
        self.margin = parameter['margin']
        self.debug = parameter['debug']
        self.pretrain_on_bg = parameter['pretrain_on_bg']

        self.support_only = parameter['support_only']
        self.opt_mask = parameter['opt_mask']
        self.use_atten = parameter['use_atten']
        self.egnn_only = parameter['egnn_only']
        orig_name = self.parameter['prefix'] 
        self.parameter['prefix'] = self.parameter['prefix'] + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.metaR = CSR(self.train_data_loader.dataset, parameter)

        self.metaR.to(self.device)
        # optimizer
        self.metaR.forward_res = None
        self.metaR.backward_res = None
        # freeze rgcn
        if parameter['freeze_edge_emb']:
            for para in self.metaR.embedding_learner.edge_embedding.parameters():
                para.requires_grad = False
        if parameter['freeze_node_emb']:
            for para in self.metaR.embedding_learner.node_embedding.parameters():
                para.requires_grad = False    
        if parameter['freeze_rgcn']:
            for para in self.metaR.embedding_learner.rgcn.parameters():
                para.requires_grad = False
        


        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.metaR.parameters()), lr=self.learning_rate)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, 0, self.epoch)

        if self.parameter['step'] == "pretrain":
            self.rgcn_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.metaR.embedding_learner.rgcn.parameters()), lr=self.learning_rate)
            self.rgcn_scheduler = transformers.get_linear_schedule_with_warmup(self.rgcn_optimizer, 0, self.epoch)


        if self.parameter['final']:
            # for grouping
            wandb.init(project="fewshotKG", entity=self.parameter['wandb_name'], name =orig_name + "_final", config = parameter)
        else:
            wandb.init(project="fewshotKG", entity=self.parameter['wandb_name'], name = self.parameter['prefix'], config = parameter)
        # wandb.run.log_code(".") # could be too slow
        wandb.save("main.py")
        wandb.save("trainer.py")
        wandb.save("protgnn_models.py")
        wandb.save("RGCN.py")
        wandb.watch(self.metaR, log_freq=100)

        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        self.logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'data')
        
        if not os.path.isdir(self.logging_dir):
            os.makedirs(self.logging_dir)
        else:
            print(self.logging_dir, "already exists!!!")
            sys.exit()
            
        self.html_f = open(os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'data',"display.html"), "w")


        if self.parameter['encoder_state_dir'] is not None:
            encoder_ckpt = torch.load(self.parameter['encoder_state_dir'], map_location='cpu')
            self.metaR.embedding_learner.rgcn.load_state_dict(encoder_ckpt)
        if self.parameter['prev_state_dir'] is not None:
            prev_ckpt = torch.load(self.parameter['prev_state_dir'], map_location='cpu')
            self.metaR.load_state_dict(prev_ckpt, strict=False)
            
        if self.parameter['transfer_state_dir'] is not None:
            prev_ckpt = torch.load(self.parameter['transfer_state_dir'], map_location='cpu')
            del prev_ckpt["embedding_learner.edge_embedding.weight"]
            del prev_ckpt["embedding_learner.rgcn.edge_embedding.weight"]
            del prev_ckpt["embedding_learner.egnn.edge_embedding.weight"]
            del prev_ckpt["embedding_learner.csg_gnn.edge_embedding.weight"]
            if "embedding_learner.node_embedding.weight" in prev_ckpt:
                del prev_ckpt["embedding_learner.node_embedding.weight"]
                del prev_ckpt["embedding_learner.rgcn.node_embedding.weight"]
                del prev_ckpt["embedding_learner.egnn.node_embedding.weight"]
                del prev_ckpt["embedding_learner.csg_gnn.node_embedding.weight"]
            self.metaR.load_state_dict(prev_ckpt, strict=False)

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def save_rgcn_checkpoint(self, epoch):
        torch.save(self.metaR.embedding_learner.rgcn.state_dict(), os.path.join(self.ckpt_dir, 'rgcn_state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch, is_eval_loss = False):
        wandb.log({'Training_Loss' + ("_eval" if is_eval_loss else ""): data['Loss'], "epoch": epoch})
        wandb.log({'Extra_Loss'+ ("_eval" if is_eval_loss else ""): data['Extra_Loss'], "epoch": epoch})


    def write_validating_log(self, data, epoch):
        wandb.log({'Acc': data['Acc'], "epoch": epoch})
        wandb.log({'F1': data['F1'], "epoch": epoch})
        wandb.log({'IOU': data['IOU'], "epoch": epoch})
        wandb.log({'ROC': data['ROC'], "epoch": epoch})
        wandb.log({'AVG ROC': data['AVG_ROC'], "epoch": epoch})
        wandb.log({'coverage': data['coverage'], "epoch": epoch})

    def write_validating_rank_log(self, data, epoch):
        wandb.log({'Validating_MRR': data['MRR'], "epoch": epoch})
        wandb.log({'Validating_Hits_10': data['Hits@10'], "epoch": epoch})
        wandb.log({'Validating_Hits_5': data['Hits@5'], "epoch": epoch})
        wandb.log({'Validating_Hits_1': data['Hits@1'], "epoch": epoch})

            
    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, iseval=False, is_eval_loss = False , curr_rel='', trial = None, best_params = None):
        loss, p_score, n_score = 0, 0, 0
        if not iseval and not is_eval_loss:
            if self.use_atten:
                self.optimizer.zero_grad()
            if self.opt_mask:
                loss, extra_loss, edgemask, edge_mask_neg, p_score, n_score = self.metaR(task, False, False, curr_rel, trial, best_params)
            else:
                p_score, n_score, extra_loss, edgemask, edge_mask_neg = self.metaR(task, False, False, curr_rel, trial, best_params)
                y = torch.Tensor([1]).to(self.device)
                loss = self.metaR.loss_func(p_score, n_score, y) + extra_loss
                if self.debug:
                    pdb.set_trace()
            if self.use_atten:
                loss.backward()
                self.optimizer.step()
        elif is_eval_loss and len(task[-4][0]) < 100 and len(task[-2][0]) < 100:
            with torch.no_grad():
                if self.opt_mask:
                    loss, extra_loss, edgemask, edge_mask_neg, p_score, n_score = self.metaR(task, False, True, curr_rel, trial, best_params)
                else:
                    p_score, n_score, extra_loss, edgemask, edge_mask_neg = self.metaR(task, False, True, curr_rel, trial, best_params)
                    y = torch.Tensor([1]).to(self.device)
                    loss = self.metaR.loss_func(p_score, n_score, y) + extra_loss          
        elif is_eval_loss and len(task[-4][0]) == 1:
            ### batch queries for ranking, where there less much pos than negs
            with torch.no_grad():
                if self.use_atten:
                    eval_bs = 100
                else:
                    eval_bs = 100 # for opt
                all_p_scores = []
                all_n_scores = []
                for idx in range(0, len(task[-2][0]), eval_bs):

                    end = idx + eval_bs
                    if end > len(task[-2][0]):
                        end = len(task[-2][0])
                    # repeat the evaluation of positives
                    sub_task = task[:-2] + ([task[-2][0][idx:end]], Batch.from_data_list(task[-1].to_data_list()[idx:end]))
                    p_score, n_score, extra_loss, edgemask, edge_mask_neg = self.metaR(sub_task, iseval, is_eval_loss, curr_rel, trial, best_params)
                    
                    all_n_scores.append(n_score.detach())
                n_score = torch.cat(all_n_scores, 0)
                y = torch.Tensor([1]).to(self.device)
                loss = self.metaR.loss_func(p_score, n_score, y)+ extra_loss       
        else:
            
            ### batch queries for roc, where there are paired pos and negs
            with torch.no_grad():
                if self.use_atten:
                    eval_bs = 10
                    if self.train_data_loader.dataset.dataset == "Wiki":
                        eval_bs = 1
                else:
                    eval_bs = 100 # for opt
#                     eval_bs = 200 # for opt
                all_p_scores = []
                all_n_scores = []
                for idx in range(0, len(task[-4][0]), eval_bs):
                    end = idx + eval_bs
                    if end > len(task[-4][0]):
                        end = len(task[-4][0])
                    sub_task = task[:-4] + ([task[-4][0][idx:end]], Batch.from_data_list(task[-3].to_data_list()[idx:end]), [task[-2][0][idx:end]], Batch.from_data_list(task[-1].to_data_list()[idx:end]))
                    p_score, n_score, extra_loss, edgemask, edge_mask_neg = self.metaR(sub_task, iseval, is_eval_loss, curr_rel, trial, best_params)
                    all_p_scores.append(p_score.detach())
                    all_n_scores.append(n_score.detach())
                p_score = torch.cat(all_p_scores, 0)
                n_score = torch.cat(all_n_scores, 0)
                y = torch.Tensor([1]).to(self.device)
                loss = self.metaR.loss_func(p_score, n_score, y)+ extra_loss
        return loss, extra_loss, p_score, n_score, edgemask, edge_mask_neg

    
    def pretrain_one_step(self, task, iseval=False, is_eval_loss = False , curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval and not is_eval_loss:
            self.optimizer.zero_grad()
            masks, reconstructed_masks, p_score, n_score = self.metaR.cycle_consistency(task)

            recon_loss = self.metaR.cycle_loss_func(masks, reconstructed_masks)
            contrastive_loss = torch.nn.MarginRankingLoss(self.margin)(p_score, n_score, torch.Tensor([1]).to(self.device))
            loss = self.coefficient * recon_loss + self.coefficient2 * contrastive_loss
            loss.backward()
            self.optimizer.step()
        elif is_eval_loss:
            assert False      
        return loss, recon_loss, contrastive_loss, (torch.sum(masks)/masks.shape[0]).item(), (torch.sum(reconstructed_masks)/masks.shape[0]).item(), (torch.sum((reconstructed_masks > 0.5).float())/masks.shape[0]).item(), masks, reconstructed_masks

    def pretrain(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        valid_data = self.eval_roc(istest=False, epoch=0)            
        self.eval_roc(istest=True, epoch=0)   
        self.eval(istest=False, epoch=0)   
        self.eval(istest=True, epoch=0)   
    
        # training by epoch
        t_load, t_one_step = 0, 0
        pbar = tqdm(range(self.epoch))
        for e in pbar:
            self.metaR.train()
            # sample one batch from data_loader
            t1 = time.time()
            if self.pretrain_on_bg:
                train_task, curr_rel = self.pretrain_data_loader.next_batch()
            else:
                train_task, curr_rel = self.train_data_loader.next_batch()
            t2 = time.time()
            loss, recon_loss, contrastive_loss, ones_in_masks, ones_in_reconstructed_masks, thresholded_ones_in_reconstructed_masks, masks, reconstructed_masks = self.pretrain_one_step(train_task, iseval=False, curr_rel=curr_rel)
            if self.finetune:
                finetune_loss, finetune_extra_loss, p_score, n_score, edgemask, edge_mask_neg = self.do_one_step(train_task, iseval=False, is_eval_loss=False)
            else:
                finetune_loss = 0

            ## eval iou
            subgraph = train_task[1]
            row, col = subgraph.edge_index
            gt = subgraph.edge_index[:, masks == 1].transpose(0, 1).tolist()
            gt_batch = subgraph.batch[row][masks == 1]
            pred = subgraph.edge_index[:, reconstructed_masks > 0.5].transpose(0, 1).tolist()
            pred_batch = subgraph.batch[row][reconstructed_masks > 0.5]
            gt_edges = [set() for _ in range(24)]
            for idx in range(len(gt)):
                gt_edges[gt_batch[idx]].add(tuple(gt[idx]))
            pred_edges = [set() for _ in range(24)]
            for idx in range(len(pred)):
                pred_edges[pred_batch[idx]].add(tuple(pred[idx]))

            ious = []
            for i in range(24):
                iou = len(gt_edges[i].intersection(pred_edges[i])) / (len(gt_edges[i].union(pred_edges[i]) ) + 0.001)
                ious.append(iou)
            iou = (sum(ious)/len(ious))

            if self.finetune_on_train:
                train_task, curr_rel = self.train_data_loader.next_batch()
                finetune_on_train_loss, finetune_extra_loss, p_score, n_score, edgemask, edge_mask_neg = self.do_one_step(train_task, iseval=False, is_eval_loss=False)
            else:
                finetune_on_train_loss = 0
            
            t3 = time.time()
            t_load += t2 - t1
            t_one_step += t3 - t2
            pbar.set_description("masks: %.4f, recon: %.4f, t_rec: %.4f, iou: %.4f" % (ones_in_masks, ones_in_reconstructed_masks, thresholded_ones_in_reconstructed_masks, iou))
            self.scheduler.step()
            # print the loss on specific epoch
            wandb.log({"train_loss": loss, "epoch": e, "iou": iou})
            wandb.log({"train_recon_loss": recon_loss, "epoch": e, "train_contrastive_loss": contrastive_loss})
            wandb.log({"train_finetune_loss": finetune_loss, "epoch": e})
            wandb.log({"train_finetune_on_train_loss": finetune_on_train_loss, "epoch": e})
        
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            if e % self.eval_epoch == 0 and e != 0:
                self.eval_roc(istest=False, epoch=e)
                self.eval_roc(istest=True, epoch=e)
                self.eval(istest=False, epoch=e)
                self.eval(istest=True, epoch=e)
        pass
    

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        t_load, t_one_step = 0, 0
        pbar = tqdm(range(self.epoch))

    
        for e in pbar:
            self.metaR.train()
            # sample one batch from data_loader
            t1 = time.time()
            train_task, curr_rel = self.train_data_loader.next_batch()
            t2 = time.time()
            loss, extra_loss,_, _, edge_mask, edge_mask_neg = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel)
            t3 = time.time()
            t_load += t2 - t1
            t_one_step += t3 - t2
            pbar.set_description("load: %s, step: %s" % (t_load/(e+1), t_one_step/(e+1)))
            self.scheduler.step()
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                if self.support_only:
                    query_subgraphs = train_task[1]
                    query_subgraphs_neg = train_task[3]
                else:
                    query_subgraphs = train_task[5]
                    query_subgraphs_neg = train_task[7]

                loss_num = loss.item()
                
                threshold = self.threshold
                print(self.metaR.forward_res)
                print(self.metaR.backward_res)
                self.write_training_log({'Loss': loss_num, 'Extra_Loss':extra_loss.item()}, e)
                print("Epoch: {}\tLoss: {:.4f} Loss extra: {:.4f}".format(e, loss_num, extra_loss.item()))
                
                if not self.egnn_only:
                    print(edge_mask.min(), edge_mask.mean(), edge_mask.max(), edge_mask.sum())
                    self.html_f.write("<p>Epoch: {}\tLoss: {:.4f} Loss extra: {:.4f} </p>".format(e, loss_num, extra_loss.item()))
                    self.html_f.write("<p>Mask min: {:.4f} Mask mean: {:.4f} Mask max: {:.4f} Mask sum: {:.4f}  </p>".format(edge_mask.min(), edge_mask.mean(), edge_mask.max(), edge_mask.sum()))


                    if hasattr(query_subgraphs, "rule_mask"):
                        print(query_subgraphs.edge_attr[query_subgraphs.rule_mask==1])
                    row, col = query_subgraphs.edge_index
                    print(query_subgraphs.batch[row][edge_mask>threshold])
                    print(query_subgraphs.edge_attr[edge_mask>threshold])
                    print((edge_mask>threshold).sum())

                    if hasattr(query_subgraphs, "rule_mask") and edge_mask is not None:
                        gt = query_subgraphs.edge_index[:,query_subgraphs.rule_mask==1].transpose(0,1).tolist()
                        gt_batch = query_subgraphs.batch[row][query_subgraphs.rule_mask==1]

                        pred = query_subgraphs.edge_index[:,edge_mask>threshold].transpose(0,1).tolist()
                        pred_batch = query_subgraphs.batch[row][edge_mask>threshold]

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
                        avg_iou = sum(ious)/len(ious)
                        coverage = sum([len(gt_edges[i].intersection(pred_edges[i])) for i in range(24)]) / sum([len(gt_edges[i]) for i in range(24)])
                        print(avg_iou)
                        print(coverage)
                        wandb.log({"train_iou": avg_iou, "epoch": e})
                        wandb.log({"train_coverage": coverage, "epoch": e})
                        self.html_f.write("<p>iou: {:.4f} intersection: {:.4f} </p>".format(avg_iou, coverage))

                    torch.save([query_subgraphs.detach().cpu(), edge_mask.detach().cpu()],os.path.join( self.logging_dir, f"train_{e}.pt") )
                    torch.save([query_subgraphs_neg.detach().cpu(), edge_mask_neg.detach().cpu()],os.path.join( self.logging_dir, f"train_neg_{e}.pt") )
                
                    self.html_f.write(f"<img src=\"train_{e}.jpg\" width=\"1500\" height=\"150\" />")
                    self.html_f.write(f"<img src=\"train_neg_{e}.jpg\" width=\"1500\" height=\"150\" />")
                    if not self.use_atten:
                        for i in range(1,8):
                            self.html_f.write(f"<img src=\"train_{e}_task_{i}.jpg\" width=\"1500\" height=\"150\" />")
                            self.html_f.write(f"<img src=\"train_neg_{e}_task_{i}.jpg\" width=\"1500\" height=\"150\" />")
                    self.html_f.flush()

                
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            if e % self.eval_epoch == 0 and e != 0:
                valid_data = self.eval_roc(istest=False, epoch=e)
                self.write_training_log(valid_data, e, is_eval_loss = True)
            
                self.eval_roc(istest=True, epoch=e)
                self.eval(istest=True, epoch=e)
                self.write_validating_log(valid_data, e)

                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current besnnnt
                    self.save_checkpoint(best_epoch)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1



        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        print('Finish')

    def hyperparameter_tune(self, istest=False, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()
        
        def objective(trial):
            data = self.eval_roc(istest=istest, trial = trial)
            return - data["ROC"]
        study = optuna.create_study()
        study.optimize(objective, n_trials=50)
        print(study.best_params)    

        self.eval_roc(istest=False, best_params = study.best_params)
        self.eval_roc(istest=True, best_params = study.best_params)
#         self.eval(istest=False, best_params = study.best_params)
        self.eval(istest=True, best_params = study.best_params)
        return
            
    def eval_roc(self, istest=False, epoch=None, trial = None, best_params = None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader

        # initial return data of validation
        data = {'Loss': 0, 'Extra_Loss': 0, "Acc" : 0, "F1": 0, "ROC": 0, "AVG_ROC": 0}
        ranks = []
        
        t = 0
        temp = dict()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        IOU = 0
        coverage = 0
        all_scores_pos = []
        all_scores_neg = []
        all_rocs = []
        thresh = torch.log(torch.tensor(0.5))
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            # sample all the eval tasks
            
            eval_task, curr_rel = batch
            loss, extra_loss, p_score, n_score, edge_mask, edge_mask_neg = self.do_one_step(eval_task, iseval=False, is_eval_loss = True , curr_rel=curr_rel, trial = trial, best_params = best_params)
            data['Loss'] += loss.item() 
            data["Extra_Loss"] += extra_loss.item()
            
            
            
            if self.support_only:
                n = 3 * len(curr_rel)
                query_subgraphs = eval_task[1]
                query_subgraphs_neg = eval_task[3]
            else:
                n = 10 * len(curr_rel)
                query_subgraphs = eval_task[5]
                query_subgraphs_neg = eval_task[7]
            
            
            if hasattr(query_subgraphs, "rule_mask") and edge_mask is not None:
                threshold = self.threshold
                row, col = query_subgraphs.edge_index
                gt = query_subgraphs.edge_index[:,query_subgraphs.rule_mask==1].transpose(0,1).tolist()
                gt_batch = query_subgraphs.batch[row][query_subgraphs.rule_mask==1]

                pred = query_subgraphs.edge_index[:,edge_mask>threshold].transpose(0,1).tolist()
                pred_batch = query_subgraphs.batch[row][edge_mask>threshold]             

                gt_edges = [set() for _ in range(n)]
                for idx in range(len(gt)):
                    gt_edges[gt_batch[idx]].add(tuple(gt[idx]))

                pred_edges = [set() for _ in range(n)]
                for idx in range(len(pred)):
                    pred_edges[pred_batch[idx]].add(tuple(pred[idx]))

                ious = []
                coverages = []
                for i in range(n):
                    iou = len(gt_edges[i].intersection(pred_edges[i])) / len(gt_edges[i].union(pred_edges[i]) )

                    ious.append(iou)
                IOU += sum(ious)/len(ious)
                coverage += sum([len(gt_edges[i].intersection(pred_edges[i])) for i in range(n)]) / sum([len(gt_edges[i]) for i in range(n)])
            else:
                IOU = 0
            all_scores_pos.append(p_score)
            all_scores_neg.append(n_score)
            if not self.support_only:
                cur_roc = roc_auc_score(torch.cat([torch.ones(p_score.shape), torch.zeros(n_score.shape)]) , torch.cat([p_score, n_score]).cpu() )
                print(cur_roc)
                all_rocs.append(cur_roc)
            
            if query_subgraphs is not None and edge_mask is not None:
                torch.save([query_subgraphs.detach().cpu(), edge_mask.detach().cpu()],os.path.join( self.logging_dir, f"eval_batch_{batch_idx}_{epoch}.pt") )
                torch.save([query_subgraphs_neg.detach().cpu(), edge_mask_neg.detach().cpu()],os.path.join( self.logging_dir, f"eval_neg_batch_{batch_idx}_{epoch}.pt") )
            
                
                
                    
            TP += (p_score > thresh).float().sum()
            TN += (n_score < thresh).float().sum()
            FP += (n_score > thresh).float().sum()
            FN += (p_score < thresh).float().sum()
            
        data["Loss"] = data["Loss"] / len(data_loader)
        data["Extra_Loss"] = data["Extra_Loss"] / len(data_loader)
        data["Acc"] =  (TP + TN ) / (TP + TN + FP + FN)
        data["precision"] = TP / (TP + FP + 1e-5)
        data["recall"] = TP / (TP + FN)
        data["F1"] = 2 * data["precision"] * data["recall"]  / (data["precision"] + data["recall"])
        data["IOU"] = IOU / len(data_loader)
        data["coverage"] = coverage / len(data_loader)
        
        if not self.support_only:

            p_score = torch.cat(all_scores_pos).reshape(-1)
            n_score = torch.cat(all_scores_neg).reshape(-1)
                
            data["ROC"] = roc_auc_score(torch.cat([torch.ones(p_score.shape), torch.zeros(n_score.shape)]) , torch.cat([p_score, n_score]).cpu() )
            data["AVG_ROC"] = np.mean(all_rocs)
        print("Eval Epoch: {}\tLoss: {:.4f} Loss extra: {:.4f} AVG_ROC: {:.4f} IOU: {:.4f} ROC: {:.4f} coverage: {:.4f}".format(epoch, data['Loss'], data["Extra_Loss"], data["AVG_ROC"], data["IOU"],  data["ROC"],  data["coverage"]))
        if istest:
            wandb.log({"test_auc": data['ROC'], "epoch": epoch})
            wandb.log({"test_avg_auc": data['AVG_ROC'], "epoch": epoch})
        else:
            wandb.log({"valid_auc": data['ROC'], "epoch": epoch})
            wandb.log({"valid_avg_auc": data['AVG_ROC'], "epoch": epoch})
        
        self.html_f.write("<p>Eval Epoch: {}\tLoss: {:.4f} Loss extra: {:.4f} Acc: {:.4f} IOU: {:.4f} ROC: {:.4f} AVG ROC: {:.4f} </p>".format(epoch, data['Loss'], data["Extra_Loss"], data["Acc"], data["IOU"],  data["ROC"], data["AVG_ROC"]))
        for i in range(5):
            self.html_f.write(f"<img src=\"eval_batch_{i}_{epoch}.jpg\" width=\"1500\" height=\"150\" />") 
            self.html_f.write(f"<img src=\"eval_neg_batch_{i}_{epoch}.jpg\" width=\"1500\" height=\"150\" />") 
        self.html_f.flush()
        
        return data        
        
    def eval(self, istest=False, epoch=None, trial = None, best_params = None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader_ranktail
        else:
            data_loader = self.dev_data_loader_ranktail
        
        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []
        ranks_x = []

        t = 0
        temp = dict()
        
        # convert to next_on_eval version
        for batch_idx, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
            # sample all the eval tasks
            
            eval_task, curr_rel = batch
            # at the end of sample tasks, a symbol 'EOT' will return

            loss, extra_loss, p_score, n_score, edge_mask, edge_mask_neg = self.do_one_step(eval_task, iseval=False, is_eval_loss = True , curr_rel=curr_rel, trial = trial, best_params = best_params)
            
            x = torch.cat([n_score.reshape(len(curr_rel), -1), p_score.reshape(-1, 1)], 1)
            for idx in range(x.shape[0]):
                t += 1
                self.rank_predict(data, x[idx], ranks)
                ranks_x.append(x[idx])
                # print current temp data dynamically
                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
                sys.stdout.write("{}\tLoss: {:.4f} Loss extra: {:.4f}".format(t, loss.item(), extra_loss.item()))
                sys.stdout.flush()

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)
    
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
               t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))
        if istest:
            prefix='test'
        else:
            prefix = 'valid'
        wandb.log({
            "%s-mrr"%prefix: data['MRR'],
            "%s-h1"%prefix: data['Hits@1'],
            "%s-h5"%prefix: data['Hits@5'],
            "%s-h10"%prefix: data['Hits@10'],
            "epoch": epoch
        })
        return data

