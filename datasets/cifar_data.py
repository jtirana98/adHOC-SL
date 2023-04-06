import torch
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
import torch
import time
from torch.autograd import Function


# REGARDING DATA
def get_dataset(type=10):
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    if type == 10:
        trainset = torchvision.datasets.CIFAR10(root=f'~/joana/', train=True,
                                                download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR100(root=f'/home/joana/Documents/trial code/cifar/cifar-{type}', train=True,
                                                download=True, transform=transform)

    if type == 10:
        testset = torchvision.datasets.CIFAR10(root=f'~/joana/', train=False,
                                              download=True, transform=transform)
    else:
         testset = torchvision.datasets.CIFAR10(root=f'/home/joana/Documents/trial code/cifar/cifar-{type}', train=False,
                                              download=True, transform=transform)                                           
    return (trainset, testset)


def get_dataloaders(trainset, testset, batch_size=64):
    train_size = int(0.9 * len(trainset))
    test_size = len(trainset) - train_size
    train_set, val_set = torch.utils.data.random_split(trainset, [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return (trainloader, validloader, testloader)

# REGARDING TRAINING
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
