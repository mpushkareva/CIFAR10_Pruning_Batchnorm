import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


# Исходная модель
def model():
    net = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(3, 32, 3, stride=2)),
              ('relu1', nn.ReLU()),

              ('conv2', nn.Conv2d(32, 64, 3, stride=2)),
              ('relu2', nn.ReLU()),

              ('conv3', nn.Conv2d(64, 128, 3, stride=2)), # добавил третий конволюционный слой
              ('relu3', nn.ReLU()),

              ('conv4', nn.Conv2d(128, 128, 3, stride=1)), # добавил четвёртый конволюционный слой
              ('relu4', nn.ReLU()),

              ('flatten', nn.Flatten()),
              ('fc1', nn.Linear(128, 64)),
              ('relu5', nn.ReLU()),

    #           ('flatten', nn.Flatten()), # добавил третий полносвязный слой
    #           ('fc2', nn.Linear(256, 128)),
    #           ('relu6', nn.ReLU()),

              ('fc3', nn.Linear(64, 10)),
              ('softmax', nn.LogSoftmax())
            ]))
    return net

def add_dropout_or_batchnorm(model, drop=0.5, add_dropout=False, add_batchnorm=False):
    if not isinstance(model, nn.Sequential):
        print('error: only for sequential models')
        return
    net = []
    for i, module in enumerate([*model]):
        net.append(deepcopy(module))
        if isinstance(module, nn.Conv2d): # после всех конволюционных
            if add_dropout:
                net.append(nn.Dropout2d(drop))
            if add_batchnorm:
                net.append(nn.BatchNorm2d(module.out_channels))
        if isinstance(module, nn.Linear) and i < len([*model]) - 2: # после всех полносвязных кроме последнего
            if add_dropout:
                net.append(nn.Dropout2d(drop))
            if add_batchnorm:
                net.append(nn.BatchNorm1d(module.out_features))
    return nn.Sequential(*net)


def remove_dropout_or_batchnorm(model, remove_dropout=False, remove_batchnorm=False):
    if not isinstance(model, nn.Sequential):
        print('error: only for sequential models')
        return
    net = [*deepcopy(model)]
    if remove_dropout:
        net = [item for item in net if not isinstance(item, nn.Dropout2d)]
    if remove_batchnorm:
        net = [item for item in net if not isinstance(item, nn.BatchNorm2d) and not isinstance(item, nn.BatchNorm1d)]
    return nn.Sequential(*net)