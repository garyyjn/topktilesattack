# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from models import GatedAttention
import torch
from datasets import BLCADataset
model = GatedAttention()

data_path = ""
datasets = BLCADataset(data_path, 'Table_S1.2017_08_05.xlsx')
datasize = len(datasets)
train_p = .8
test_p = 1 - train_p
train_size = int(train_p*datasize)
test_size = int(test_p*datasize)
waist_size = datasize - train_size - test_size
train_set, test_set, na = torch.utils.data.random_split(datasets, [train_size, test_size, waist_size])

epoch_limit = 10
batch_size = 1

train_loader = torch.utils.data.dataloader(train_set)
test_loader = torch.utils.data.dataloader(test_set)

