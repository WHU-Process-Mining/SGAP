import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import os
from configs.config import load_config_data

class EventLogData():
    def __init__(self, path):
        df = pd.read_csv(path, usecols=['case', 'activity_id'], dtype={
            'activity_id': int, 'case': str})
        
        self.event_log = df
        self.start = 0
        self.all_activities = np.unique(df['activity_id'])
        self.all_activities = np.insert(self.all_activities, 0, values=0)

        self.total_activities_num = len(np.unique(df['activity_id'])) + 1
        self.total_data_list, self.adj = self._generate_data_for_input()

    def _generate_data_for_input(self):
        total_data_list = [] # [sequence, current, target] 
        current_case_list = [0] 
        end = self.total_activities_num + 1
        adj = np.zeros(shape=(self.total_activities_num, self.total_activities_num)) # node adj

        for idx, row in self.event_log.iterrows():
            case_id, activity_id = row
            
            if len(current_case_list) == 1: # add start node to adj matrix
                adj[self.start, activity_id] += 1
            
            if idx + 1 < self.event_log.shape[0]: 
                next_case_id, next_activity_id = self.event_log.loc[idx+1]
                next_activity_id = end if next_case_id != case_id else next_activity_id
            else: # The end of whole log
                next_activity_id = end
            
            
            if next_activity_id == end:
                current_case_list = [0]
                continue
            
            adj[activity_id, next_activity_id] += 1
            current_case_list.append(activity_id)
            total_data_list.append([copy.copy(current_case_list), next_activity_id])


        adj = np.where(np.array(adj) > 0, 1, 0) # obtain non-frequence adj
        return total_data_list, adj
    
    def split_valid_data(self, valid_ratio):
        valid_n = int(valid_ratio * len(self.total_data_list))
        
        # valid_data_list = random.sample(self.total_data_list, valid_n)
        
        train_data_list , valid_data_list = train_test_split(
            self.total_data_list, test_size=valid_n, shuffle=True
        )
        self.train_data_list = train_data_list
        self.valid_data_list = valid_data_list
        
if __name__ == "__main__":
    
    print("************* process event log ***************")
    cfg_model_input = load_config_data("configs/model_input.yaml")
    
    for name in cfg_model_input['dataset_names']:
        print(name)
        os.makedirs(f'data/{name}', exist_ok=True)
        path = cfg_model_input['dataset_path'] + '/' + name + '/kfold_data/'
        for idx in range(cfg_model_input['k_fold_num']):
            train_file_name = path + name + '_kfoldcv_' + str(idx) + '_train.csv'   
            test_file_name = path + name + '_kfoldcv_' + str(idx) + '_test.csv'
            
            event_log = EventLogData(train_file_name)
            event_log.split_valid_data(cfg_model_input['valid_ratio']) # split valid set 
            
            test_log = EventLogData(test_file_name)
            np.save(f'data/{name}/kfold{idx}_train.npy', np.array(event_log.train_data_list, dtype=object))
            np.save(f'data/{name}/kfold{idx}_valid.npy', np.array(event_log.valid_data_list, dtype=object))
            np.save(f'data/{name}/kfold{idx}_test.npy', np.array(test_log.total_data_list, dtype=object))