from trainer import *
from params import *
import json
from CommonSubgraph.load_CSG_dataset import CSGDataset
import models
from load_kg_dataset import *
import pdb


if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = ''
    if params['data_form'] == 'In-Train':
        tail = '_in_train'

    if params['real']:
        n_query = 10
        if params['step'] in ['test_roc', 'pretrain', 'tune']:
            n_query = None
        kind = "union_prune_plus"
        if params['dataset'] in ['NELL', 'NELL_newsplit']: 
            hop = 2
        elif params['dataset'] in ['FB15K-237', 'ConceptNet']:
            hop = 1
        elif params['dataset'] == 'ConceptNet':
            hop = 1
        else:
            assert False
        if params['skip_training_dataset']:
            train_data_loader = None
        else:
            train_data_loader = PairSubgraphsFewShotDataLoader(SubgraphFewshotDataset(params["data_path"], shot = params['few'], dataset=params['dataset'], mode="train", rev = params['rev'], kind=kind, hop=hop, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test'] ), batch_size= params["batch_size"], shuffle = True)
        if params['pretrain_dataset'] is not None:
            pretrain_data_loader = PairSubgraphsFewShotDataLoader(CSGDataset(params["data_path"], dataset=params['pretrain_dataset'], shot = params['few'],  mode="train"), batch_size= params["batch_size"], shuffle = True)
        else:
            pretrain_data_loader = PairSubgraphsFewShotDataLoader(SubgraphFewshotDataset(params["data_path"], shot = params['few'], dataset=params['dataset'], mode="pretrain", rev = params['rev'], kind=kind, hop=hop, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], orig_test= params['orig_test']), batch_size= params["batch_size"], shuffle = True)
        test_data_loader = PairSubgraphsFewShotDataLoader(SubgraphFewshotDataset(params["data_path"], shot = params['few'], n_query = n_query, dataset=params['dataset'], mode="test", rev = params['rev'], kind=kind, hop=hop, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test']), batch_size= 1)
        dev_data_loader = PairSubgraphsFewShotDataLoader(SubgraphFewshotDataset(params["data_path"], shot = params['few'], n_query = n_query, dataset=params['dataset'], mode="dev", rev = params['rev'], kind=kind, hop=hop, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test']), batch_size= 1)
        

        if params['num_rank_negs'] > 50:
            params["rank_tail_batch_size"] = 1 
        test_data_loader_ranktail = PairSubgraphsFewShotDataLoader(SubgraphFewshotDatasetRankTail(params["data_path"], hop = hop, shot = params['few'], n_query = n_query, dataset=params['dataset'], mode="test", rev = params['rev'], kind=kind, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test']), batch_size=  params["rank_tail_batch_size"])
        dev_data_loader_ranktail = PairSubgraphsFewShotDataLoader(SubgraphFewshotDatasetRankTail(params["data_path"], hop = hop, shot = params['few'], n_query = n_query, dataset=params['dataset'], mode="dev", rev = params['rev'], kind=kind, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test']), batch_size=  params["rank_tail_batch_size"])
        

    else:
        models.SYNTHETIC = True
        pretrain_data_loader = None
        train_data_loader = PairSubgraphsFewShotDataLoader(CSGDataset(params["data_path"], dataset=params['dataset'], shot = params['few'],  mode="train"), batch_size= params["batch_size"], shuffle = True)
        test_data_loader = PairSubgraphsFewShotDataLoader(CSGDataset(params["data_path"], dataset=params['dataset'],shot = params['few'], n_query = 10,  mode="test"), batch_size= 1)
        dev_data_loader = PairSubgraphsFewShotDataLoader(CSGDataset(params["data_path"], dataset=params['dataset'], shot = params['few'], n_query = 10,  mode="dev"), batch_size= 1)
        dev_data_loader_ranktail = None
        test_data_loader_ranktail = None
        
        if params['step'] == 'tune': 
            test_data_loader = PairSubgraphsFewShotDataLoader(CSGDataset(params["data_path"], dataset=params['dataset'],shot = params['few'], n_query = 1,  mode="test"), batch_size= 50)
            dev_data_loader = PairSubgraphsFewShotDataLoader(CSGDataset(params["data_path"], dataset=params['dataset'], shot = params['few'], n_query = 1,  mode="dev"), batch_size= 50)

    data_loaders = [train_data_loader, dev_data_loader, test_data_loader, dev_data_loader_ranktail, test_data_loader_ranktail, pretrain_data_loader]
    trainer = Trainer(data_loaders, None, params)


    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)   
    elif params['step'] == 'tune':
        trainer.hyperparameter_tune(istest=False) 
        
    elif params['step'] == 'opt_test':
        if params['dataset'] == "NELL": 
            if not params['inductive']:
                best_params = {'lr': 0.4272434952703357, 'beta': 0.3805215812608637, 'size_loss_beta': 0.7752281142633662, 'connectivity_loss_beta': 0.4146276876754283, 'epsilon_perf': 0.9647842106125186, 'epsilon_con': 0.9065198167178523, 'beta_2': 3.87129625201692e-06, 'size_loss_beta_2': 4.984069437892537}
            else:
                best_params = {'lr': 0.1850696414351297, 'beta': 0.32915219548599417, 'size_loss_beta': 0.8165748329935867, 'connectivity_loss_beta': 0.06001236228092588, 'epsilon_perf': 0.918779607986352, 'epsilon_con': 0.9127479732929106, 'beta_2': 2.160946245436929e-06, 'size_loss_beta_2': 1.208465690353429}
        
        if params['dataset'] == "FB15K-237": 
            if not params['inductive']:
                best_params = {'lr': 0.3696665591248449, 'beta': 0.05115163009934731, 'size_loss_beta': 0.19087754211720234, 'connectivity_loss_beta': 0.3392274350938487, 'epsilon_perf': 0.9828856661949582, 'epsilon_con': 0.9156964479826757, 'beta_2': 0.00047325746186136473, 'size_loss_beta_2': 0.07787902045316168}
            else:
                best_params = {'lr': 0.9035027218912838, 'beta': 0.26527158675073803, 'size_loss_beta': 0.43571144720812716, 'connectivity_loss_beta': 0.3786326023325238, 'epsilon_perf': 0.9837608563003981, 'epsilon_con': 0.9843290665639614, 'beta_2': 0.0005317224659912181, 'size_loss_beta_2': 3.109305928410248}

        if params['dataset'] == "ConceptNet":
            if not params['inductive']:
                best_params = {'lr': .2183281548658058, 'beta': 0.0046903119105187185, 'size_loss_beta': 0.3511572280264563, 'connectivity_loss_beta': 0.863809562498436, 'epsilon_perf': 0.9604720443653948, 'epsilon_con': 0.9838573384929162, 'beta_2': 3.324708061927636e-05, 'size_loss_beta_2': 4.121786235504798}
            else:
                best_params = {'lr': 0.4497209556202594, 'beta': 0.12953118594940427, 'size_loss_beta': 0.7979398629258933, 'connectivity_loss_beta': 0.48238750266710756, 'epsilon_perf': 0.9327420393503416, 'epsilon_con': 0.9163594885458736, 'beta_2': 0.00010688160374155669, 'size_loss_beta_2': 4.547521693636153}

        trainer.eval(istest=True, best_params=best_params) 
            
    elif params['step'] == 'test_roc':
        data = trainer.eval_roc(istest=True)
        trainer.eval(istest=True)
        
    elif params['step'] == 'test':        
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            data = trainer.eval(istest=True)
            trainer.write_validating_rank_log(data, 0)
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)
    elif params['step'] == 'pretrain':
        trainer.pretrain()

