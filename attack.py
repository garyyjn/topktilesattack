from models import *
import torch

load_path = ''
model = GatedAttention()
model.load_state_dict(torch.load(load_path))

def singletopkattack(model, features, topk = 1):
    features_copy = torch.tensor(torch.clone(features), requires_grad=True)
    y, a = model(features_copy)
    res, ind = a.topk(topk)
    print("model prediciton {}".format(y))
    print("top k attented values {}".format(res))
    print("top k attention index {}".format(ind))


singletopkattack(model, features = ,)