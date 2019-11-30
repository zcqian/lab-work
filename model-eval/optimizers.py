import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler
from lr_scheduler import LinearDecayLR


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


def shufflenet_imagenet_default(model: torch.nn.Module):
    LR = 0.5
    END_LR = 1e-6
    SLOPE = (LR-END_LR) / 240  # more than 240 epochs and boom
    MOMENTUM = 0.9
    WD = 4e-5
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = LinearDecayLR(optimizer, SLOPE)
    return optimizer, scheduler


def shufflenet_v2_imagnet_sgd_tune(model: torch.nn.Module):
    LR = 0.1
    MOMENTUM = 0.9
    WD = 1e-4
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20])
    return optimizer, scheduler


def shufflenet_v2_imagnet_sgd_tune_2(model: torch.nn.Module):
    LR = 0.1
    MOMENTUM = 0.9
    WD = 4e-5
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40])
    return optimizer, scheduler

def shufflenet_v2_imagnet_sgd_tune_3(model: torch.nn.Module):
    LR = 0.1
    MOMENTUM = 0.9
    WD = 4e-5
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20])
    return optimizer, scheduler


def resnet_cifar_sgd(model, wd=1e-4):
    LR = 0.1
    MOMENTUM = 0.9
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150])
    return optimizer, scheduler


def resnet_cifar_inc_batch(model, wd=1e-4):
    LR = 0.01
    MOMENTUM = 0.9
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=wd
    )
    scheduler = None
    return optimizer, scheduler

def resnet_cifar_sgd_wd5(model):
    return resnet_cifar_sgd(model, wd=5e-4)


def mnist_opt(model: nn.Module):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20])
    return optimizer, scheduler


def mobilenet_v2_opt(model: nn.Module):
    """Optimizer/LR_Scheduler for MobileNet v2

    Uses parameters comparable to the paper, but LR is lower because our batch size is smaller.
    They use batch size of 96 x 16 (I think), we are using 256. We should decrease from 0.045 by a factor of sqrt(6),
    but 0.02 seems close enough.

    Args:
        model: MobileNet_v2 or similar

    Returns:

    """
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.02, alpha=0.9, momentum=0.9, weight_decay=0.00004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    return optimizer, scheduler
