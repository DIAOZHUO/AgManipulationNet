import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix


def test_model(model, test_data_loader, device):
    should_swtich_to_train = model.training
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_data_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted.data)
            # print(labels.data)
            total += len(labels)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    if should_swtich_to_train:
        model.train()
    return correct / total

def get_two_state_check(outputs, threshold):
    pred_value = np.argmax(outputs)
    if pred_value == 6:
        return True

    for j in range(15):
        if outputs[j] < threshold:
            outputs[j] = 0
    sum = np.sum(outputs) - outputs[6]
    if sum > 0:
        pred_value1 = (np.sum(outputs[:6]) / sum) > 0.5
        if (pred_value < 6) != pred_value1:
            return False
    else:
        return False
    return True


def test_model_accuracy(model, test_data_loader, device, test_time=1):
    model.eval()
    error_list = []
    real_value_list = []
    pred_value_list = []
    with torch.no_grad():
        correct = 0
        total = 0
        for _ in range(test_time):
            for index, item in enumerate(test_data_loader):
                images = item['image'].to(device)
                labels = item['label'].to(device)

                outputs = model(images).to('cpu').detach().numpy().copy()
                total += len(labels)

                for i in range(len(labels)):
                    pred_value = np.argmax(outputs[i])
                    label_value = labels[i].item()
                    real_value_list.append(label_value)
                    pred_value_list.append(pred_value)

                    if pred_value == label_value:
                        correct += 1
                    else:
                        error_list.append([i + index * 16, pred_value])
                        # print("error in", labels[i], i + index * 16)
                        # print("error in", labels[i], i + index * 16)


        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

    return error_list, real_value_list, pred_value_list



def test_goodbad_model_accuarcy(model, test_data_loader, device, test_time=1, threshold=2.4):
    model.eval()
    error_list = []
    real_value_list = []
    pred_value_list = []
    with torch.no_grad():
        correct = 0
        total = 0
        for _ in range(test_time):
            for index, item in enumerate(test_data_loader):
                images = item['image'].to(device)
                labels = item['label'].to(device)
                # print("batch", index, len(labels))

                outputs = model(images).to('cpu').detach().numpy().copy()
                total += len(labels)

                for i in range(len(labels)):
                    for j in range(15):
                        if outputs[i, j] < threshold:
                            outputs[i, j] = 0
                    if labels[i].item() == 6:
                        total -= 1
                        continue
                    label_value = labels[i].item() < 6
                    sum = np.sum(outputs[i]) - outputs[i][6]
                    if sum <= 0:
                        total -= 1
                        continue
                    pred_value = (np.sum(outputs[i][:6]) / sum) > 0.5
                    real_value_list.append(label_value)
                    pred_value_list.append(pred_value)

                    if pred_value == label_value:
                        correct += 1
                    else:
                        error_list.append([i + index * 16, pred_value])
                        # print("error in", labels[i], i + index * 16)
        print('Test Good or Bad Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

    return error_list, real_value_list, pred_value_list




# def test_siamese_model(model, test_data_loader, device):
#     should_swtich_to_train = model.training
#     model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for input1, input2, target in test_data_loader:
#             output1, output2 = model(input1['image'].to(device), input2['image'].to(device))
#             outputs = None
#             _, predicted = torch.max(outputs.data, 1)
#             # print(predicted.data)
#             # print(labels.data)
#             total += len(labels)
#             correct += (predicted == labels).sum().item()
#
#         print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
#     if should_swtich_to_train:
#         model.train()
#     return correct / total



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
