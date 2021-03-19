import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
from copy import deepcopy


def get_pruned_model(model, normed=False, amount=0.5):
    model_pruned = deepcopy(model)
    for i in range(len(model)):
        if isinstance(model_pruned[i], nn.Conv2d):
            prune.ln_structured(model_pruned[i], name='weight', amount=amount, n=2, dim=0)
            if normed:
                model_pruned[i].weight = model_pruned[i].weight / (1 - amount)
        elif isinstance(model_pruned[i], nn.Linear) and i != len(model) - 2:
            prune.ln_structured(model_pruned[i], name='weight', amount=amount, n=2, dim=0)
            if normed:
                model_pruned[i].weight = model_pruned[i].weight / (1 - amount)
    return model_pruned


def remove_prune_params(model):
    for key, value in model.named_modules():
        if isinstance(value, nn.Conv2d):
            prune.remove(value, name='weight')
        elif isinstance(value, nn.Linear):
            prune.remove(value, name='weight')
    return model