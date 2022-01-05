# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from models import GatedAttention
import torch
from datasets import BLCADataset
import torch.optim
import torch
torch.manual_seed(0)
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

epoch_limit = 50
batch_size = 1

train_loader = torch.utils.data.DataLoader(train_set)
test_loader = torch.utils.data.DataLoader(test_set)



loss = torch.nn.CrossEntropyLoss(weight = torch.tensor([.4,1,.7]))
optimizer = torch.optim.SGD([
                {'params': model.attention_V.parameters(), 'lr': 200},
                {'params': model.attention_U.parameters(), 'lr': 200},
                {'params': model.attention_weights.parameters(), 'lr':7e-1},
                {'params': model.classifier.parameters()}],
                lr=5e-2, momentum=0.9)

def train_epoch(model, loader):
    total_num = 0
    correct_num = 0
    for i_batch, sample_batched in enumerate(loader):
        model.zero_grad()
        features = sample_batched['features']
        target = sample_batched['target']
        #print(features[0,0,:] - features[0,1,:])
        target_onehot = torch.nn.functional.one_hot(target, num_classes = 3).float()
        output = model(features[0,:,:].float())
        y_prob, a = output
        print(a)
        #print(y_prob)
        batch_loss = loss(y_prob, target_onehot)
        batch_loss.backward()
        #print(model.attention_V[0].weight.grad)
        optimizer.step()
        model_choice = torch.argmax(y_prob)
        if model_choice == target:
            correct_num += 1
        total_num += 1
    print("train batch, accu: {} correct: {} total: {}".format(correct_num/total_num, correct_num, total_num))
        #print(target)


def test_epoch(model, loader):
    total_num = 0
    correct_num = 0
    answer_list = []
    truth_list = []
    prob_list = []
    for i_batch, sample_batched in enumerate(loader):
        model.zero_grad()
        features = sample_batched['features']
        target = sample_batched['target']
        output = model(features[0,:,:].float())
        y_prob = output[0]
        model_choice = torch.argmax(y_prob)
        if model_choice == target:
            correct_num += 1
        answer_list.append(int(model_choice))
        prob_list.append(list(y_prob))
        truth_list.append(int(target))
        total_num += 1
    print(prob_list)
    print(answer_list)
    print(truth_list)
    print("test batch, accu: {} correct: {} total: {}".format(correct_num/total_num, correct_num, total_num))

for i in range(epoch_limit):
    train_epoch(model, loader=train_loader)
    test_epoch(model, loader=test_loader)

torch.save(model.state_dict(), './modelsave.dict')
