from configs.config import load_config_data
from train import train_model
import numpy as np
import torch
import time
import pickle

def SGAP_objective(trial, idx): 
    
    # define the parameter search space
    model_parameters = {}
    
    model_parameters['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model_parameters['layer_num'] = trial.suggest_categorical('layer_num', [2, 3, 4])
    model_parameters['feature_num'] = trial.suggest_categorical('feature_num', [8, 16, 32, 64, 128])
    model_parameters['hidden_size'] = trial.suggest_categorical('hidden_size', [8, 16, 32, 64, 128, 256, 512])
    model_parameters['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    model_parameters['dropout'] = trial.suggest_float('dropout', 0, 1)
    model_parameters['encoder_layer'] = trial.suggest_categorical('encoder_layer', [1, 2, 3])
    
    
    # load the model config
    cfg_model_train = load_config_data("configs/SGAP_Model.yaml")
    data_path = 'data/' + cfg_model_train['dataset']
    model_parameters['window_size'] = cfg_model_train['avg_len']
    model_parameters['activity_num'] = cfg_model_train['activity_num']
    model_parameters['model_name'] = cfg_model_train['model_parameters']['model_name'] 
    model_parameters['encoder_type'] = cfg_model_train['model_parameters']['encoder_type'] 
    model_parameters['num_epochs'] = cfg_model_train['model_parameters']['num_epochs']
    
    if model_parameters['encoder_type'] == 'transformer':
        model_parameters['num_head'] = trial.suggest_int('num_head', 1, 4)
    else: model_parameters['num_head'] = 1
    
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
     
    start_time = time.time()
    
        
    train_data_list = np.load(f'{data_path}/kfold{idx}_train.npy', allow_pickle=True).tolist()
    val_data_list = np.load(f'{data_path}/kfold{idx}_valid.npy', allow_pickle=True).tolist()
        
    print(f"fold: {idx+1}, dataset: {cfg_model_train['dataset']}, train size: {len(train_data_list)}, valid size:{len(val_data_list)}")
    
    best_model, best_val_accurace, train_loss_plt, train_accuracy_plt, val_accuracy_plt = train_model(trial, idx, train_data_list, val_data_list, model_parameters, device)
        
    save_folder = 'results_kfold_'+ str(cfg_model_train['k_fold_num'])+'/'+cfg_model_train['model_parameters']['model_name'] + '/' + cfg_model_train['dataset']
     
    current_best = trial.study.best_value if trial.number > 0 else 0
    if best_val_accurace > current_best:
        with open( f'{save_folder}/model/best_model_kfd{idx}.pickle', 'wb') as fout:
            pickle.dump(best_model, fout)

    duartime = time.time() - start_time
   
    record_file = open(f'{save_folder}/optimize/opt_history_{idx}.txt', 'a')
    record_file.write(f"\n{trial.number},{best_val_accurace},{model_parameters['learning_rate']},{model_parameters['layer_num']},{model_parameters['feature_num']},{model_parameters['hidden_size']},{model_parameters['batch_size']},{model_parameters['dropout']},{model_parameters['window_size']},{model_parameters['encoder_layer']},{model_parameters['num_head']},{duartime}")
    record_file.close()
    
    return best_val_accurace