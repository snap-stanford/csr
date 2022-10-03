import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("--final", default=False, type=bool) 
    args.add_argument("--wandb_name", type=str, required = True) 
    
    args.add_argument("-data", "--dataset", default="NELL", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-pretrain_data", "--pretrain_dataset", default=None, type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("--real", default=True, type=bool) 
    args.add_argument("--fix2", default=False, type=bool) 
    args.add_argument("--num_rank_negs", default=50, type=int) 

    args.add_argument("-path", "--data_path", default=".", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("--emb_path", default="./embedding", type=str)  # ["./NELL", "./Wiki"]
    
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=3, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("--hop", default=2, type=int)
    args.add_argument("-metric", "--metric", default="Acc", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    args.add_argument("-rev", "--rev", default=False, type=bool) ## add rev edges

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default= 8, type=int)
    args.add_argument("-rtbs", "--rank_tail_batch_size", default= 1, type=int)
    args.add_argument("-lr", "--learning_rate", default=1e-5, type=float)
    args.add_argument("-ebeta", "--extra_loss_beta", default=0, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)

    args.add_argument("-epo", "--epoch", default=5000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=10, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=100, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=100, type=int)

    args.add_argument("-b", "--beta", default=0.5, type=float)
    args.add_argument("-m", "--margin", default=0.5, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)
    
    args.add_argument("-sub", "--use_subgraph", default=True, type=bool)
    args.add_argument("-so", "--support_only", default=False, type=bool)
    args.add_argument("-om", "--opt_mask", default=False, type=bool)
    args.add_argument("-gt", "--use_ground_truth", default=False, type=bool)
    args.add_argument("-fr", "--use_full_mask_rule", default=False, type=bool)
    args.add_argument("--use_full_mask_query", default=False, type=bool)
    args.add_argument("-jt", "--joint_train_mask", default=False, type=bool)
    
    args.add_argument("--use_pretrain_edge_emb", default=True, type=bool)
    args.add_argument("--use_rnd_node_emb", default=False, type=bool)
    args.add_argument("--use_pretrain_node_emb", default=False, type=bool)
    
    args.add_argument("-att", "--use_atten", default=False, type=bool)
    args.add_argument("--egnn_only", default=False, type=bool)
    args.add_argument("-pdb", "--pdb_mode", default=False, type=bool)
    args.add_argument("-verbose", "--verbose", default=False, type=bool)
    args.add_argument("-freeze_node_emb", "--freeze_node_emb", default=False, type=bool)
    args.add_argument("-freeze_edge_emb", "--freeze_edge_emb", default=True, type=bool)
    args.add_argument("-freeze_rgcn", "--freeze_rgcn", default=False, type=bool)

    args.add_argument("-gpu", "--device", default=0, type=int)
    args.add_argument("-niters", "--niters", default=1, type=int)
    args.add_argument("-emb_dim", "--emb_dim", default=100, type=int)
    args.add_argument("--our_emb", default=False, type=bool)
    
    args.add_argument("-hidden_dim", "--hidden_dim", default=128, type=int)
    args.add_argument("-geo", "--geo", default="vec", type=str, choices=['vec', 'box'])
    args.add_argument("-threshold", "--threshold", default=0.8, type=float)
    args.add_argument("-coefficient", "--coefficient", default=0.1, type=float)
    args.add_argument("-coefficient2", "--coefficient2", default=1, type=float)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="pretrain", type=str, choices=['train', 'test', 'dev', 'test_roc', 'pretrain' , 'tune', 'opt_test'])
    args.add_argument("-lm", "--loss_mode", default="normal", type=str, choices=['inverse', 'inverse-sqrt', 'inverse-log', 'normal'])
    args.add_argument("-pool", "--pool_mode", default="min", type=str, choices=['min', 'mean'])
    args.add_argument("-opt", "--opt_mode", default="iters_of_perm_and_min", type=str, choices=['iters_of_perm_min_end', 'iters_of_perm_and_min', 'no_decode', 'no_decode_share','iters_3_min_end'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-encoder_state_dir", "--encoder_state_dir", default=None, type=str)
    args.add_argument("-prev_state_dir", "--prev_state_dir", default=None, type=str)
    args.add_argument("-transfer_state_dir", "--transfer_state_dir", default=None, type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-finetune", "--finetune", default=True, type=bool)
    args.add_argument("-debug", "--debug", default=False, type=bool)
    args.add_argument("-pretrain_on_bg", "--pretrain_on_bg", default=True, type=bool)
    args.add_argument("-skip_training_dataset", "--skip_training_dataset", default=False, type=bool)
    args.add_argument("-finetune_on_train", "--finetune_on_train", default=False, type=bool)

    args.add_argument("-emb_model", "--embed_model", default="TransE", choices=["TransE", "ComplEx"])
    args.add_argument("-bidir", "--bidir", default=True, type=bool)
    args.add_argument("--inductive", default=False, type=bool)
    args.add_argument("--orig_test", default=False, type=bool)

    args.add_argument("-nn", "--normalize_node_emb", default=False, type=bool)
    args.add_argument("-nm", "--no_margin", default=False, type=bool)
    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    params['device'] = torch.device('cuda:'+str(args.device))

    return params


data_dir = {
    'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates.json',

    'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2.json',

    'ent2ids': '/ent2ids',
    'ent2vec': '/ent2vec.npy',
}
