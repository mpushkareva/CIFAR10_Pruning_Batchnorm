import numpy as np
import torch.nn as nn
import torch


def change_conv_weights(layer, layer_masked, in_mask):
    copy_weight = layer.weight
    copy_bias = layer.bias
    out_mask = (layer_masked.weight_mask.detach().numpy() == 1)[:, 0, 0, 0]
    in_channels = len(np.where(in_mask == True)[0])
    out_channels = len(np.where(out_mask == True)[0])
    kernel_size = layer.kernel_size
    stride = layer.stride
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    with torch.no_grad():
        layer.weight.copy_(copy_weight[out_mask][:, in_mask])
        layer.bias.copy_(copy_bias[out_mask])
    return layer, out_mask

def change_lin_weights(layer, in_mask, first_linear):
    copy_weight = layer.weight
    copy_bias = layer.bias
    if first_linear:
        hw = int(layer.in_features / len(in_mask))
        in_mask = np.repeat(in_mask, hw)
        in_features = len(np.where(in_mask == True)[0])
        
    else:
        in_features = len(np.where(in_mask == True)[0])
    if not first_linear:
        out_features = layer.out_features
        out_mask = np.repeat(True, out_features)
    else:
        out_mask = (layer.weight_mask.detach().numpy() == 1)[:, 0]
        out_features = len(np.where(out_mask == True)[0])
    layer = nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.copy_(copy_weight[out_mask][:, in_mask])
        layer.bias.copy_(copy_bias[out_mask])
    first_linear = False
    return layer, out_mask, first_linear

def change_bn(layer, in_mask, prune_rate, norm_mean_var):
    if norm_mean_var and i != 1:
        copy_mean = layer.running_mean * (1 - prune_rate)
        copy_var = layer.running_var * (1 - prune_rate)
    else:
        copy_mean = layer.running_mean
        copy_var = layer.running_var
    copy_weight = layer.weight
    copy_bias = layer.bias
    channels = len(np.where(in_mask == True)[0])
    if isinstance(layer, torch.nn.BatchNorm2d):
        layer = nn.BatchNorm2d(channels)
    else:
        layer = nn.BatchNorm1d(channels)
    with torch.no_grad():
        layer.weight.copy_(copy_weight[in_mask])
        layer.bias.copy_(copy_bias[in_mask])
        layer.running_mean.copy_(copy_mean[in_mask])
        layer.running_var.copy_(copy_var[in_mask])
    return layer

def resize_model(model, device, norm_mean_var=False):
    model = model.cpu()
    in_mask = np.array([True] * 3) 
    first_linear = True
    # define prune rate by first conv layer
    prune_rate = len(np.where(model[0].weight_mask[:, 0, 0, 0] == 0)[0]) /\
                 len(model[0].weight_mask[:, 0, 0, 0])
    for i in range(len(model)):
        if isinstance(model[i], torch.nn.Conv2d):
            model[i], in_mask= change_conv_weights(model[i], model[i], in_mask)
        if isinstance(model[i], torch.nn.Linear):
            model[i], in_mask, first_linear =\
                change_lin_weights(model[i], in_mask, first_linear)
        if isinstance(model[i], torch.nn.BatchNorm2d) or isinstance(model[i], torch.nn.BatchNorm1d):
            model[i] = change_bn(model[i], in_mask, prune_rate, norm_mean_var)
    return model.to(device)