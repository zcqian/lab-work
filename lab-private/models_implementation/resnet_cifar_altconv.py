import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from textwrap import dedent
import math
from models_implementation.clsf_utils import __fixed_eye, __no_bias

from .clsf_utils import generate_cube_random

__all__ = ['rn32_cf100_exg_d_sep_fixed_eye']


def _weights_init(m):
    if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and not isinstance(m, FixedConv2d)):
        init.kaiming_normal_(m.weight)


class FixedConv2d(nn.Conv2d):
    pass  # just a hack to change signature


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ClsfBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_classes=100, option='G', groups=1):
        super(ClsfBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, num_classes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != num_classes:
            if option == 'G':  # d-cube random
                w = generate_cube_random(64, 100)
                w = w.view(num_classes, in_planes, 1, 1)
                conv = FixedConv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False)
                conv.weight.data = w.float()
                conv.weight.requires_grad_(False)
                self.shortcut = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(num_classes)
                )
            else:
                raise NotImplementedError

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class ResNet_alt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, option='G', clsf_groups=1):
        super(ResNet_alt, self).__init__()
        self.clsf_expansion_option = option
        self.clsf_groups = clsf_groups
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, num_classes=num_classes)
        self.fc = nn.Linear(100, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, num_classes=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            if (num_classes is not None) and (idx == len(strides) - 1):
                layers.append(ClsfBlock(self.in_planes, planes, stride, num_classes,
                                        self.clsf_expansion_option, self.clsf_groups))
            else:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


for groups in [2, 4]:
    code = f"""\
    def rn32_cf100_exg_groups_{groups}_fixed_eye():
        model = ResNet_alt(BasicBlock, [5, 5, 5], clsf_groups={groups})
        model = __fixed_eye(model)
        return model
    """
    exec(dedent(code))
    __all__ += [f'rn32_cf100_exg_groups_{groups}_fixed_eye']


def rn32_cf100_exg_d_sep_fixed_eye():
    model = ResNet_alt(BasicBlock, [5, 5, 5])
    model.layer3[4].conv2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=64),
        nn.Conv2d(64, 100, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    )
    model = __fixed_eye(model)
    return model
