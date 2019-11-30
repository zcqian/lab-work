from torchvision.models import vgg13_bn
from torchvision.models.vgg import vgg16
import torch.nn as nn

from .clsf_utils import generate_orthoplex
from collections import OrderedDict

__all__ = ['vgg13_bn_cf100_fixed_orthoplex', 'vgg13_bn_cf100',
           'vgg16_nc_1k',
           ]


def vgg13_bn_cf100():
    """VGG-13 model with batch normalization for CIFAR-100 classification

    This tries to implement the CIFAR-100 classifier/model in the Fix Your Features paper (arXiv: 1902.10441).

    Their paper is a bit ambiguous about what hyper-parameters they used.
    "All the networks use Batch Normalization and ReLU if not otherwise specified."
    "... and the same hyper-parameters used in the original work."
    So I am guessing that they keep the number of channels in each Conv2d the same, and adjusted FC layers accordingly,
    also replacing Dropout with BatchNorm in the classifier.
    Also, this uses d=50, for the input dimension to the final classifier layer.

    Returns:
        nn.Module that implements VGG-13 w/ BN for CIFAR-100 using learned fully connected final classifier
    """
    m = vgg13_bn()
    m.avgpool = nn.Sequential()  # given the input 32x32, features already output 1x1, no point doing avg pooling
    m.classifier = nn.Sequential(
        nn.Linear(512, 50),
        nn.BatchNorm1d(50),
        nn.ReLU(inplace=True),
        nn.Linear(50, 50),
        nn.BatchNorm1d(50),
        nn.ReLU(inplace=True),
        nn.Linear(50, 100)
    )
    return m


def vgg13_bn_cf100_fixed_orthoplex():
    """VGG-13 w/ BatchNorm for CIFAR-100 using fixed orthoplex as last weight matrix

    Returns:
        nn.Module
    """
    m = vgg13_bn_cf100()
    classifier = nn.Linear(50, 100)
    classifier.weight.data = generate_orthoplex(50, 100)
    classifier.weight.requires_grad_(False)
    m.classifier[6] = classifier
    return m


def vgg16_nc_1k():
    model = vgg16(pretrained=True)
    model.features[28] = nn.Conv2d(512, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    del model.features[30]  # remove max pool and relu
    del model.features[29]
    model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    model.classifier = nn.Identity()
    return model
