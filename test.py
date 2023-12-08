import torch
import numpy as np
import pickle
from configs.config import load_config_data
from train import test_model

if __name__ == "__main__":
    
    cfg_model = load_config_data("configs/SGAP_Model.yaml")
    model_parameters = cfg_model['model_parameters']
    
    data_path = 'data/' + cfg_model['dataset']
    save_folder = 'results_kfold_'+ str(cfg_model['k_fold_num'])+'/'+cfg_model['model_parameters']['model_name']+ '/' + cfg_model['dataset']
    
    cfg_model['model_parameters']['activity_num'] = cfg_model['activity_num']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
           
                
    for idx in range(cfg_model['k_fold_num']):
        test_data_list = np.load(f'{data_path}/kfold{idx}_test.npy', allow_pickle=True).tolist()
        train_data_list = np.load(f'{data_path}/kfold{idx}_train.npy', allow_pickle=True).tolist()
        
        # Load the best model.
        with open(f'{save_folder}/model/best_model_kfd{idx}.pickle', 'rb') as fin:
            best_model = pickle.load(fin).to(device)
        
        test_accurace = test_model(train_data_list, test_data_list, best_model, cfg_model['avg_len'], device)
        print(f"kfold: {idx}, test size: {len(test_data_list)}, test_accurace:{test_accurace} ")