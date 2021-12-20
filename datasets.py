from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class BLCADataset(Dataset):
    #directory format has 2 directories: dictionaries + features and the excel for labels
    def __init__(self, directory, excel_name):
        self.root_dir = directory
        self.excel_name = excel_name
        self.dictionary_path = os.path.join(directory, 'dictionaries')
        self.excel_path = os.path.join(directory, excel_name)
        self.features_path = os.path.join(directory, 'features')
        list_dir = os.listdir(self.features_path)
        self.alphabetical_items = sorted(list_dir)
        self.excel_info = pd.read_excel(self.excel_path, 'Master table')

    def __len__(self):
        return len(self.alphabetical_items)

    def __getitem__(self, idx, feature_name = 'Histologic subtype'):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.alphabetical_items[idx]
        features = np.load(os.path.join(self.features_path, item_name)).astype(np.float)
        item_name_excel = item_name[:12]
        target_row = self.excel_info[self.excel_info['Case ID'] == item_name_excel]
        target_label = target_row[feature_name].item()
        if(feature_name == 'Histologic subtype'):
            manual_dict = {'Non-Papillary': 0, 'Papillary':1, 'ND':2}
            target_label = manual_dict[target_label]
        
        sample = {'features' : features, 'target' : target_label}
        #sample = {'features' : features}
        return sample

