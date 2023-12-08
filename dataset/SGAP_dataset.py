
from dataset.ap_dataset import APDataset
from utils.util import get_w_list
import numpy as np

class SGAPDataset(APDataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(self, data_list, shuffle=False,  window_size = None):
        super(SGAPDataset, self).__init__(data_list, shuffle)
        self.window_size = window_size

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: (history sequence, next activity)
        """
        history_ws_seq = get_w_list(self.data_list[idx][0], self.window_size)
            
        return np.array(history_ws_seq + [self.data_list[idx][1]])