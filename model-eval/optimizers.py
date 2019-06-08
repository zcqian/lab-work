import torch
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


def resnet_cifar_sgd(model, start_epoch=0):
    LR = 0.1
    MOMENTUM = 0.9
    WD = 1e-4
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150])
    return optimizer, scheduler
