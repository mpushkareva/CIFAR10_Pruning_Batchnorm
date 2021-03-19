import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from train_test import test


def get_accuracies(network, loader, device, model_name=None):
    if model_name is not None:
        print(model_name)
    total_len = len(loader.dataset)
    test_loss, correct = test(network, loader, device=device, is_print=False)
    print(f'Accuracy in eval mode:  test loss: {np.mean(test_loss):.4f}, accuracy: {np.sum(correct)}/{total_len} ({100. * np.sum(correct)/total_len:.2f}%)')
    test_loss, correct = test(network, loader, device=device,   is_train=True, is_print=False)
    print(f'Accuracy in train mode: test loss: {np.mean(test_loss):.4f}, accuracy: {np.sum(correct)}/{total_len} ({100. * np.sum(correct)/total_len:.2f}%)')
    print('____________________________________________')
    
    
def get_accuracies_pd(models, loader, device):
    columns = [ 'Accuracy eval', 'Accuracy train', 'Loss eval', 'Loss train']
    index = [m for m in models]
    data = pd.DataFrame(index=index, columns=columns)
    total_len = len(loader.dataset)
    for i, model in enumerate(models):
        test_loss, correct = test(models[model], loader, device=device, is_print=False)
        data['Loss eval'].iloc[i] = np.mean(test_loss)
        data['Accuracy eval'].iloc[i] = 100. * np.sum(correct) / total_len
        test_loss, correct = test(models[model], loader, device=device, is_train=True, is_print=False)
        data['Loss train'].iloc[i] = np.mean(test_loss)
        data['Accuracy train'].iloc[i] = 100. * np.sum(correct) / total_len
    return data.rename_axis('Model Name', axis=1)


def acc_on_batch(network, data, target, device):
    network = network.to(device)
    data = data.to(device)
    output = network(data)
    target = target.to(device)
    pred = output.data.max(1, keepdim=True)[1]
    return (pred.eq(target.data.view_as(pred)).sum().item() / len(data))


def running_stats(model, num):
    model1 = deepcopy(model)
    print('infere net in eval mode ...')
    test(model1, test_loader)
    run_mean = deepcopy(model1[num].running_mean)
    run_var = deepcopy(model1[num].running_var)
    print('infere net in train mode ...')
    test(model1, test_loader, is_train=True)
    print('Mean relation mean ', torch.mean(mode1[num].running_mean / run_mean))
    print('Mean relation var ', torch.mean(model1[num].running_var / run_var))
    
def running_stats_batch(network, train_loader, device):
    eval_acc = []
    train_acc = []
    bn_list = []
    means_eval = []
    vars_eval = []
    means_rel = defaultdict(list)
    vars_rel = defaultdict(list)
    for i, layer in enumerate(network):
        if isinstance(layer, torch.nn.BatchNorm2d)\
            or isinstance(layer, torch.nn.BatchNorm1d):
                bn_list.append(i)
    with torch.no_grad():
        network.eval()
        for i in bn_list:
            means_eval.append(network[i].running_mean.clone().detach().cpu())  # по абс и по отдельным каналам
            vars_eval.append(network[i].running_var.clone().detach().cpu())
        for data, target in train_loader:
            network.eval()
            eval_acc = eval_acc + [acc_on_batch(network, data, target, device)]

            network.train()
            train_acc = train_acc + [acc_on_batch(network, data, target, device)]
            for i, b in enumerate(bn_list):
                means_rel[i].append(torch.mean(torch.abs(network[b].running_mean.cpu() / means_eval[i])))
                vars_rel[i].append(torch.mean(torch.abs(network[b].running_var.cpu() / vars_eval[i])))

    return eval_acc, train_acc, means_rel, vars_rel