import torch.nn as nn
from torchvision.models import DenseNet

from .hadamard_3rdpty import HadamardProj
from .resnet_orthogonal import LambdaLayer

__all__ = ['dn_bc_100_12_fc', 'dn_bc_100_12_hd', 'dn_bc_100_12_id_last100']


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


def dn_bc_100_12_id_last100():
    """Returns DenseNet-BC 100 layers channels, with identity classifier using last 100 input channels

    This model uses a fixed identity classifier (i.e. no classifier).
    Contrary to the usual way of using the first 100 input channels, this uses the last 100 channels.
    This is due to the nature of DenseNets, of the 342 channels of input,
    it is the last 100 channels that are the "deepest".
    (Where the first 100 merely goes through two DenseBlocks and two Transition layers. )

    Returns:
        nn.Module: model of said description
    """
    model = dn_bc_100_12_fc()
    model.classifier = nn.Sequential(
        LambdaLayer(lambda x: x[:, -100:]),
        nn.Identity()
    )
    return model
