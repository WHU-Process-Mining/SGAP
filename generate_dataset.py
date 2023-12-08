import pandas as pd
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
import copy
from configs.config import load_config_data

date_format_str = '%Y/%m/%d %H:%M:%S'

def format_time(t, format):
    if type(format) is str:
        t_struct = time.strptime(str(t), format)
    elif type(format) is list:
        try:
            t_struct = time.strptime(str(t), format[0])
        except:
            t_struct = time.strptime(str(t), format[1])
    return time.strftime(date_format_str, t_struct)

def get_time(t):
    t_struct = time.strptime(str(t), date_format_str)
    return time.mktime(t_struct)

def generate_length_graph(long_list, dataset, file_name):
    plt.figure(figsize = (8, 4))
    x = long_list[:, 0]
    y = long_list[:, 1]
    plt.bar(x, y)
    for a,b in zip(x, y):
        plt.text(a, b+0.001, b, ha='center', va= 'bottom',fontsize=6)
    plt.xlabel("case_len")
    plt.ylabel("count")
    plt.legend(labels=['count'], loc='best')
    os.makedirs('Data/' + dataset, exist_ok=True)
    plt.savefig('Data/' + dataset + "/" + file_name + '_len.png')
    plt.close()

def minus (all_list, l):
    if len(l):
        return [a for a in all_list if a not in l]
    else:
        return all_list

def iter_sample (l1, l2):
    same_x = [x for x in l1 if x in l2]  
    if len(same_x) == 0:
        return random.sample(l1, 1)[0]
    else:
        return None

if __name__ == "__main__":
    divide = "--------------------------------------------------------------------------------"
    cfg_dataset = load_config_data("configs/dataset.yaml")
    log_str = ''
    for name in cfg_dataset['dataset_names']:
        print(name)
        log_str += divide + "database: " + name + divide + "\n"

        event_log = pd.DataFrame()
        csv_file = cfg_dataset['dataset_path'] + '/' + name  + '.csv'
        case_all = []

        if name == 'helpdesk':
            event_log = pd.read_csv(csv_file)
            event_log.columns = cfg_dataset['cols']['new']
            event_log['case'] = event_log['case'].map(lambda x: int(x))
            event_log['timestamp'] = event_log['timestamp'].map(
                lambda x: format_time(x, '%Y-%m-%d %H:%M:%S'))
        else:
            if name == 'BPI_challenge_2020_PermitLog':
                columns = cfg_dataset['cols']['default'] +  cfg_dataset['cols']['bpi20_p']
            elif name == 'Receipt':
                columns = cfg_dataset['cols']['default'] +  cfg_dataset['cols']['receipt']
            elif name == 'BPI_challenge_2020_InternationalDeclarations':
                columns = cfg_dataset['cols']['default'] +  cfg_dataset['cols']['bpi20_id_cols']
            elif name == 'BPI_challenge_2020_DomesticDeclarations':
                columns = cfg_dataset['cols']['default'] +  cfg_dataset['cols']['bpi20_dd_cols']
            elif name == 'BPI_challenge_2020_PrepaidTravelCost':
                columns = cfg_dataset['cols']['default'] +  cfg_dataset['cols']['bpi20_ptco']
            else:
                columns = cfg_dataset['cols']['default']

            event_log = pd.read_csv(csv_file, usecols=columns)[columns]
            event_log.rename(columns=dict(zip(cfg_dataset['cols']['default'], cfg_dataset['cols']['new'])), inplace=True)

            event_log['case'] = event_log['case'].map(lambda x: str(x).strip())
            event_log['activity'] = event_log['activity'].map(lambda x: str(x).strip())
            event_log['timestamp'] = event_log['timestamp'].map(
                lambda x: format_time(x, ["%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S.%f%z"]))
            
            event_log.rename(columns={'org:resource': 'resource'}, inplace=True)
            event_log.columns = event_log.columns.str.replace('org:', 'org_')
            event_log.columns = event_log.columns.str.replace('case:', '')

            event_log.rename(columns={'AMOUNT_REQ': 'amount'}, inplace=True)
            event_log.rename(columns={'travel permit number': 'travelPermitNumber'}, inplace=True)
            event_log.rename(columns={'Permit travel permit number': 'travelPermitNumber'}, inplace=True)

            event_log.columns = event_log.columns.str.replace(' ', '')
            
        case_all_list = np.unique(event_log['case'])
        activity_all_list = np.unique(event_log['activity'])

        case_log = event_log.groupby('case', sort=False)
        
        # remove case_len=1
        case_all = [x for x in case_all_list if case_log.get_group(x).shape[0] >= 2]
        if len(case_all) > len(case_all_list): log_str += "The number of case length =1 : {}, will be removed\n".format(len(case_all) - len(case_all_list))

        # activity transition：{case}/
        activity_case_dict = {}
        for index, item in event_log.iterrows():
            case, activity = item['case'], item['activity']
            activity_case_dict.setdefault(activity, [])

            activity_case_dict[activity].append(case)
            activity_case_dict[activity] = list(set(activity_case_dict[activity]))

        k = cfg_dataset['k_fold_num']
        # handle activitiy-case frequence < the number of kfold
        low_freq_activities = [x for x in activity_all_list if len(activity_case_dict[x]) < k]

        low_freq_case = []
        low_freq_activities = list(set(low_freq_activities))

        # for the case the low_freq_activities involved, remove it in the activity_case_dict
        for activity in low_freq_activities:
            case_list = activity_case_dict.pop(activity)
            low_freq_case += case_list

            for case in case_list:
                activity_list = case_log.get_group(case)['activity']
                print("low_freq_activity: {}".format(activity))
                log_str += "low_freq_activity: " + str(activity) + "\n"
                log_str += "case-{}:{}\n".format(case, list(activity_list))

                for a in activity_list:
                    if (not a == activity) and case in activity_case_dict[a]:
                        activity_case_dict[a].remove(case)

        
        freq_list = np.array(
            sorted(activity_case_dict.items(), key=lambda x: len(x[1])), dtype=object) # sorted by value

        low_freq_case = list(set(low_freq_case))

        # divided to initial k fold to ensure each activity appear in every fold
        
        kfold=[[] for _ in range(k)]
        activity_remain_list = activity_case_dict.keys()
        print("all_activity:%s" % (len(activity_case_dict.keys())))
        
        is_init = True
        kfold_case_all = minus(case_all, low_freq_case)
        kfold_remain_list = kfold_case_all
        for activity, _ in freq_list:
            case_list = activity_case_dict[activity]
            if len(case_list) >= k: # Consider the situation that deleted case causes the activity-case to be less than k
                if is_init:
                    sample_list = random.sample(case_list, k)
                    for idx in range(k):
                        kfold[idx].append(sample_list[idx])
                    is_init = False
                    kfold_remain_list = minus(kfold_remain_list, sample_list)

                else:
                    for idx in range(k):
                        sample = iter_sample(case_list, kfold[idx])
                        if sample: # the activity never appear in this fold,otherwise no need to append
                            kfold[idx].append(sample)
                            case_list = minus(case_list, [sample])
                            kfold_remain_list = minus(kfold_remain_list, [sample])
        
        kfold_num = int(len(kfold_case_all) / k)
        kfold_num_remain = len(kfold_case_all) % k

        # remaining cases divide into k fold
        for idx in range(k):          
            sample_list = random.sample(kfold_remain_list ,kfold_num - len(kfold[idx]))
            kfold[idx] += sample_list
            kfold_remain_list = minus(kfold_remain_list, sample_list)

        for i in range(kfold_num_remain):
            sample_list = random.sample(kfold_remain_list,  1)
            kfold[i] += sample_list
            kfold_remain_list = minus(kfold_remain_list, sample_list)
        
        # generate the complete dataframe
        df = pd.DataFrame(columns=event_log.columns, dtype=object)
        for idx in range(k):
            for i in kfold[idx]:
                current = copy.copy(case_log.get_group(i))
                current['kfold'] = idx
                current['time'] = current['timestamp'].map(lambda x: get_time(x))
                current.sort_values(by="time", ascending=True, inplace=True)

                df = pd.concat([df, current], ignore_index=True)
        
        # Map activity name to int category  1~activity  _num                                                           
        activity2id = dict(zip(activity_remain_list, range(1, len(activity_remain_list) + 1)))
        df['activity_id'] = df['activity'].map(lambda x: activity2id[x])
        
        for idx in range(k):
            test_file_name = name + '_kfoldcv_' + str(idx) + '_test.csv'
            train_file_name = name + '_kfoldcv_' + str(idx) + '_train.csv'

            test_df = df[df['kfold'] == idx].copy()
            train_df= df[df['kfold'] != idx].copy()

            print("train:%s, test:%s" % (len(train_df['case'].unique()), len(test_df['case'].unique())))
            print("train_activity:%s" % (len(train_df['activity'].unique())))
            print("test_activity:%s" % (len(test_df['activity'].unique())))


            os.makedirs(cfg_dataset['dataset_path'] + '/' + name + '/kfold_data', exist_ok=True)
            test_df.to_csv(cfg_dataset['dataset_path'] + '/' + name + '/kfold_data/' + test_file_name, index=False, columns=list(event_log.columns)+['activity_id'])
            train_df.to_csv(cfg_dataset['dataset_path'] + '/' + name + '/kfold_data/' + train_file_name, index=False, columns=list(event_log.columns)+['activity_id'])
        
    with open(cfg_dataset['output_log'], "a") as f:
        f.write(log_str)
        f.close()
