import torch
import random
import pickle
import optuna
import numpy as np
from configs.config import load_config_data
import os
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset.SGAP_dataset import SGAPDataset
from model.SGAP import SGAP
from utils.util import get_w_list, generate_graph
from copy import deepcopy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False # 
    torch.backends.cudnn.deterministic = True

# Test the test data(val data)
def test_model(train_data, test_data, model, window_size, device):
    train_seqs = torch.tensor([get_w_list(i[0], window_size) for i in train_data], device=device)
    test_seqs = torch.tensor([get_w_list(i[0], window_size) for i in test_data], device=device)
    test_target = [i[1] for i in test_data]
    
    with torch.no_grad():
        model.eval()
        logits = model(train_seqs, test_seqs)
        predictions = torch.argmax(logits, dim=1)
            
    avg_accuracy = accuracy_score(predictions.cpu().numpy().tolist(), test_target)
    return avg_accuracy

def train_model(trial, idx, train_data, val_data, model_parameters, device):
    
    print("************* Training Model ***************")
    
    train_dataset = SGAPDataset(train_data, shuffle=True, window_size=model_parameters['window_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=model_parameters['batch_size'])
    
    if model_parameters['model_name'] =='SGAP':
        model = SGAP(activity_num=model_parameters['activity_num'] + 1, # total_activities_num
                    layer_num=model_parameters['layer_num'],
                    hidden_size=model_parameters['hidden_size'],
                    feature_num=model_parameters['feature_num'],
                    encoder_type=model_parameters['encoder_type'],
                    encoder_layer=model_parameters['encoder_layer'],
                    num_heads= 1,
                    dropout=model_parameters['dropout']).to(device)
    else:
        raise Exception("This Model Don't exit")
    
    optimizer = optim.AdamW(model.parameters(), lr=model_parameters['learning_rate'])
    crossentropy = nn.CrossEntropyLoss()
    
    train_loss_plt = []
    train_accuracy_plt = []
    val_accuracy_plt = []
    
    best_val_accurace = 0

    
    # Train Model
    for epoch in range(model_parameters['num_epochs']):
        model.train()
        predictions_list = [] 
        true_list = []
        training_loss = 0
        num_train = 0
        
        for inputs in train_dataloader:
            optimizer.zero_grad()
            batch_data = inputs.to(device)
            seq, targets = batch_data[:, :-1], batch_data[:, -1]
            
            logits = model(seq)
            loss = crossentropy(logits, targets) # nn.crossentropy involves softmax
        
            loss.backward()
            optimizer.step()

            true_list.extend(targets.cpu().numpy().tolist())
            predictions_list.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            num_train += 1
            training_loss += loss.item()
        
        train_loss_plt.append(training_loss/num_train)
        train_accuracy_plt.append(accuracy_score(true_list, predictions_list))

        # test the accurace in val dataset
        val_accurace = test_model(train_data, val_data, model, model_parameters['window_size'], device)
        val_accuracy_plt.append(val_accurace)
        print(f"epoch: {epoch}, val_accurace:{val_accurace} ")

       
        if epoch == 0 or val_accurace > best_val_accurace:
            best_val_accurace = val_accurace
            best_model = deepcopy(model)
            

        # Report intermediate objective value.
        trial.report(best_val_accurace, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return best_model, best_val_accurace, train_loss_plt, train_accuracy_plt, val_accuracy_plt

if __name__ == "__main__":
    
    # load the model config
    cfg_model_train = load_config_data("configs/SGAP_Model.yaml")
    
    setup_seed(cfg_model_train['seed'])
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
        
    data_path = 'data/' + cfg_model_train['dataset']
    save_folder = 'results_kfold_'+ str(cfg_model_train['k_fold_num'])+'/'+cfg_model_train['model_parameters']['model_name'] +'/'+ cfg_model_train['dataset']
    
    os.makedirs(save_folder + f'/model', exist_ok=True)
    os.makedirs(save_folder + f'/train', exist_ok=True)
    os.makedirs(save_folder + f'/curves', exist_ok=True)
    
    cfg_model_train['model_parameters']['activity_num'] = cfg_model_train['activity_num']
    
    
    print("************* Training in different k-fold dataset ***************")
    for idx in range(cfg_model_train['k_fold_num']):
        
        train_data_list = np.load(f'{data_path}/kfold{idx}_train.npy', allow_pickle=True).tolist()
        val_data_list = np.load(f'{data_path}/kfold{idx}_valid.npy', allow_pickle=True).tolist()
        
        best_model, best_val_accurace, train_loss_plt, train_accuracy_plt, val_accuracy_plt = train_model(idx, train_data_list, val_data_list, cfg_model_train['model_parameters'], device)

        # print the loss and accurace curve
        generate_graph(save_folder + f'/curves/curve_kfd{idx}.jpg', train_loss_plt, train_accuracy_plt, val_accuracy_plt)
        
        with open( f'{save_folder}/train/best_model_kfd{idx}.pickle', 'wb') as fout:
            pickle.dump(best_model, fout)
    