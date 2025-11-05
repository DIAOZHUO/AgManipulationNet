import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix


def test_model(model, test_data_loader, device, th1=0.5, th2=0.5):
    should_swtich_to_train = model.training
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        total = 0
        correct1 = 0
        correct2 = 0
        for item in test_data_loader:
            input = item['input'].to(device)
            I = item['I'].to(device)
            Z = item['Z'].to(device)
            labels = item['label'].to(device)
            outputs = model(Z, I, input)
            predicted = outputs.data
            # print("predict:", predicted.data)
            # print("label:", labels.data)
            predict1 = predicted.data[:,0] > th1
            predict2 = predicted.data[:,1] > th2
            # print(labels.data[:, 1])
            total += len(labels)
            correct1 += (predict1 == labels.data[:, 0]).sum().item()
            correct2 += (predict2 == labels.data[:, 1]).sum().item()

        # print('Test Accuracy of the model on the {} test images,  1:{} %, 2:{} %'.format(total, 100 * correct1 / total, 100 * correct2 / total))
    if should_swtich_to_train:
        model.train()
    return correct1 / total, correct2 / total



def test_iz_model(model, test_data_loader, device, th1=0.5):
    should_swtich_to_train = model.training
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        total = 0
        correct1 = 0
        correct2 = 0
        for item in test_data_loader:
            I = item['I'].to(device)
            Z = item['Z'].to(device)
            labels = item['label'].to(device)
            outputs = model(Z, I)
            predicted = outputs.data
            # print("predict:", predicted.data)
            # print("label:", labels.data)
            predict1 = predicted.data[:,0] > th1
            # print(labels.data[:, 1])
            total += len(labels)
            correct1 += (predict1 == labels.data[:, 0]).sum().item()

        # print('Test Accuracy of the model on the {} test images,  1:{} %, 2:{} %'.format(total, 100 * correct1 / total, 100 * correct2 / total))
    if should_swtich_to_train:
        model.train()
    return correct1 / total


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, weight_decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer
