# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from models import GatedAttention
import torch
from datasets import BLCADataset
import torch.optim
model = GatedAttention()

data_path = "/home/ext_yao_gary_mayo_edu/CLAM/extraction_output"
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

train_loader = torch.utils.data.DataLoader(train_set)
test_loader = torch.utils.data.DataLoader(test_set)

loss = torch.nn.NLLLoss()
optimizer = torch.optim.SGD([
                {'params': model.parameters()}],
                lr=1e-2, momentum=0.9)

def train_epoch(model, loader):
    total_num = 0
    correct_num = 0
    for i_batch, sample_batched in enumerate(loader):
        model.zero_grad()
        features = sample_batched['features']
        target = sample_batched['target']
        output = model(features[0,:,:].float())
        batch_loss = loss(output, target)
        batch_loss.backward()
        optimizer.step()
        #print(output)
        #print(target)


def test_epoch(model, loader):
    pass

for i in range(epoch_limit):
    train_epoch(model, loader=train_loader)
