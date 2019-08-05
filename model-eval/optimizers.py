import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler


def resnet_imagenet_sgd_default(model: torch.nn.Module):
    LR = 0.1
    MOMENTUM = 0.9
    WD = 1e-4
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])
    return optimizer, scheduler


def resnet_imagenet_sgd_tune(model: torch.nn.Module):
    LR = 0.1
    MOMENTUM = 0.9
    WD = 1e-4
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10])
    return optimizer, scheduler


def resnet_cifar_sgd(model, wd=1e-4):
    LR = 0.1
    MOMENTUM = 0.9
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150])
    return optimizer, scheduler


def resnet_cifar_sgd_wd5(model):
    return resnet_cifar_sgd(model, wd=5e-4)


def mnist_opt(model: nn.Module):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20])
    return optimizer, scheduler


def mobilenet_v2_opt(model: nn.Module):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, alpha=0.9, momentum=0.9, weight_decay=0.00004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    return optimizer, scheduler
