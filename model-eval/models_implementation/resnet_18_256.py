import torch
import torch.nn as nn

from .resnet_orthogonal import RepackagedResNet18, LambdaLayer
from .hadamard_3rdpty import HadamardProj


__all__ = ['rn18_256_fc', 'rn18_256_hd', 'rn18_256_id', 'rn18_256_or']

def rn18_256_fc():
    model = RepackagedResNet18(pretrained = False)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        LambdaLayer(lambda x: x[:, :256]),
        nn.Linear(256, 256, bias=True)
    )
    return model


def rn18_256_hd():
    model = rn18_256_fc()
    model.classifier[-1] = HadamardProj(256, 256)
    return model


def rn18_256_id():
    model = rn18_256_fc()
    model.classifier[-1] = nn.Identity()
    return model


def rn18_256_or():
    model = rn18_256_fc()
    fc = nn.Linear(256, 256, bias=True)
    nn.init.orthogonal_(fc.weight.data)
    fc.weight.requires_grad_(False)
    model.classifier[-1] = fc
    return model
