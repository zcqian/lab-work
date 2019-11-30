'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from textwrap import dedent
import math
from models_implementation.clsf_utils import __fixed_eye, __no_bias, \
    generate_hadamard, generate_orthoplex, generate_cube_ordered, generate_cube_random


__all__ = []


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

    def __init__(self, in_planes, planes, stride=1, option='A', num_classes=100):
        super(ClsfBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != num_classes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                pad_size = int((num_classes - in_planes)//2)
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x, [0, 0, 0, 0, pad_size, pad_size], "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(num_classes)
                )
            elif option == 'C':  # Hadamard and scaling
                h = generate_hadamard(in_planes, num_classes)
                h = h.view(num_classes, in_planes, 1, 1)
                conv = FixedConv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False)
                conv.weight.data = h.float()
                conv.weight.requires_grad_(False)

                init_scale = 1. / math.sqrt(num_classes)
                self.scale = nn.Parameter(torch.tensor(init_scale))
                self.shortcut = nn.Sequential(
                    conv,
                    LambdaLayer(lambda x: - self.scale * x),
                    nn.BatchNorm2d(num_classes)
                )
            elif option == 'D':  # Fixed Orthoplex
                w = torch.tensor(generate_orthoplex(in_planes, num_classes))
                w = w.view(num_classes, in_planes, 1, 1)
                conv = FixedConv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False)
                conv.weight.data = w.float()
                conv.weight.requires_grad_(False)
                self.shortcut = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(num_classes)
                )
            elif option == 'E':  # shuffled fixed Orthoplex, using all channels at least once (64x +1 then 36x -1)
                w = torch.zeros(num_classes, in_planes)
                for row in range(num_classes):
                    col = row % in_planes
                    w[row, col] = 1 if row < in_planes else -1
                w = w .view(num_classes, in_planes, 1, 1)
                conv = FixedConv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False)
                conv.weight.data = w.float()
                conv.weight.requires_grad_(False)
                self.shortcut = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(num_classes)
                )
            elif option == 'F':  # d-cube ordered
                w = generate_cube_ordered(64, 100)
                w = w.view(num_classes, in_planes, 1, 1)
                conv = FixedConv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False)
                conv.weight.data = w.float()
                conv.weight.requires_grad_(False)
                self.shortcut = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(num_classes)
                )
            elif option == 'G':  # d-cube random
                w = generate_cube_random(64, 100)
                w = w.view(num_classes, in_planes, 1, 1)
                conv = FixedConv2d(in_planes, num_classes, kernel_size=1, stride=stride, bias=False)
                conv.weight.data = w.float()
                conv.weight.requires_grad_(False)
                self.shortcut = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(num_classes)
                )
            elif option == 'H':  # d-cube some better ordering that I can think of
                raise NotImplementedError  # don't know how to do it yet
            else:
                raise NotImplementedError

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class ResNet_alt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, option='A'):
        super(ResNet_alt, self).__init__()
        self.clsf_expansion_option = option
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
                layers.append(ClsfBlock(self.in_planes, planes, stride, self.clsf_expansion_option, num_classes))
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


for option in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    code = f"""\
    def rn32_cf100_ex{option}():
        model = ResNet_alt(BasicBlock, [5, 5, 5], option='{option.upper()}')
        return model
    
    def rn32_cf100_ex{option}_fixed_eye():
        model = rn32_cf100_ex{option}()
        model = __fixed_eye(model)
        return model
    
    def rn32_cf100_ex{option}_no_bias():
        model = rn32_cf100_ex{option}()
        model = __no_bias(model)
        return model
    
    def rn32_cf100_ex{option}_fixed_eye_no_bias():
        model = rn32_cf100_ex{option}()
        model = __no_bias(model)
        model = __fixed_eye(model)
        return model
    """
    exec(dedent(code))
    __all__ += [f'rn32_cf100_ex{option}', f'rn32_cf100_ex{option}_fixed_eye',
                f'rn32_cf100_ex{option}_no_bias', f'rn32_cf100_ex{option}_fixed_eye_no_bias']



