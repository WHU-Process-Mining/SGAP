import optuna
from optuna.visualization import plot_optimization_history
from optuna.samplers import TPESampler
import gc
import os
from train import setup_seed
from configs.config import load_config_data
from parameter_search import SGAP_objective


if __name__ == "__main__":
    
    # load the model config
    cfg_model_train = load_config_data("configs/SGAP_Model.yaml")

    # Fixed random number seed
    setup_seed(cfg_model_train['seed']) 
    
    data_path = 'data/' + cfg_model_train['dataset']
    save_folder = 'results_kfold_'+ str(cfg_model_train['k_fold_num'])+'/'+cfg_model_train['model_parameters']['model_name'] + '/' + cfg_model_train['dataset']
    
    os.makedirs(f'{save_folder}/optimize', exist_ok=True)
    os.makedirs(f'{save_folder}/model', exist_ok=True)
    
    for idx in range(cfg_model_train['k_fold_num']):
        # record optimization
        record_file = open(f'{save_folder}/optimize/opt_history_{idx}.txt', 'w')
        record_file.write("tid,score,learning_rate,layer_num,feature_num,hidden_size,batch_size,dropout,window_size,encoder_layer,num_head,duartime")
        record_file.close()
        
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=cfg_model_train['seed'])) # fixed parameter
        study.optimize(lambda trial: SGAP_objective(trial, idx), n_trials=20, gc_after_trial=True, callbacks=[lambda study, trial:gc.collect()])
        
        # record optimization history
        history = optuna.visualization.plot_optimization_history(study)
        plot_optimization_history(study).write_image(f"{save_folder}/optimize/opt_history_{idx}.png")
        
        outfile = open(f'{save_folder}/model/best_model_kfd{idx}.txt', 'w')
        best_params = study.best_params
        best_accuracy = study.best_value

        print("Best hyperparameters:", best_params)
        print("Best accuracy:", best_accuracy)
        
        outfile.write('Best trail:' + str(study.best_trial.number))
        outfile.write('\nBest hyperparameters:' + str(best_params))
        outfile.write('\nBest accuracy:' + str(best_accuracy))
        outfile.close()
        