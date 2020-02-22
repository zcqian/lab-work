import torch.nn as nn
from torchvision.models import DenseNet

from .hadamard_3rdpty import HadamardProj

__all__ = ['dn_bc_100_12_fc', 'dn_bc_100_12_hd']


def dn_bc_100_12_fc():
    model = DenseNet(growth_rate=12, block_config=(16, 16, 16), num_init_features=24, num_classes=100)
    conv = nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    model.features.conv0 = conv
    return model


def dn_bc_100_12_hd():
    model = dn_bc_100_12_fc()
    in_features, out_features = model.classifier.in_features, model.classifier.out_features
    model.classifier = HadamardProj(in_features, out_features)
    return model
