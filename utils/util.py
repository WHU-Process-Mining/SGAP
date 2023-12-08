import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_graph(file_path, train_loss_list, train_acc_list, val_acc_list):
    assert len(train_loss_list) == len(train_acc_list) == len(val_acc_list), "data list length not consistent, please check out d data"
    
    epochs_list = [i for i in range(1, len(train_loss_list)+1)]
    fig = plt.figure(figsize=(12,6)) 
    ax1 = fig.add_subplot(121)  

    ###绘图
    ax1.plot(epochs_list, train_loss_list, linewidth=2, label='train_loss', color='Blue')
    ###添加图例
    ax1.legend(loc='upper right')

    ####添加X，Y坐标轴
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")


    ax2 = fig.add_subplot(122)  
    ###绘图
    ax2.plot(epochs_list, train_acc_list, linewidth=2, label='train_acc', color='Blue')
    ax2.plot(epochs_list, val_acc_list, linewidth=2, label='test_acc', color='Red')
    ###添加图例
    ax2.legend(loc='upper right')

    ####添加X，Y坐标轴
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")

    ####保存文件
    plt.savefig(file_path)
    plt.close()

# Gets the case list of the most recent window size ws
def get_w_list(current_list, ws):
    current_len = len(current_list)
    if ws > current_len:
        w_list = np.pad(current_list, (ws - current_len, 0), 'constant')
    else:
        w_list = current_list[-ws:]

    return list(w_list)
