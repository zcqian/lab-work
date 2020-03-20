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
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models_implementation.clsf_utils import __fixed_eye, __no_bias, generate_orthoplex

__all__ = ['rn32_cf10', 'rn32_cf100', 'resnet32_cifar10_fixed_eye',
           'resnet32_cifar64_fixed_eye', 'resnet32_cifar64', 'resnet32_cifar64_fixed_eye_no_bias',
           'resnet32_cifar100_no_bias', 'resnet32',
           'rn32_fixed_orthoplex', 'rn32_cf64_fx_or',
           'ResNet', 'BasicBlock']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
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


def rn32_cf10():
    return ResNet(BasicBlock, [5, 5, 5])


def rn32_cf100():
    return ResNet(BasicBlock, [5, 5, 5], num_classes=100)


def resnet32(**kwargs):
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet32_cifar10_fixed_eye():
    model = rn32_cf10()
    model = __fixed_eye(model)
    return model


def resnet32_cifar64():
    return ResNet(BasicBlock, [5, 5, 5], num_classes=64)


def resnet32_cifar64_fixed_eye():
    model = resnet32_cifar64()
    model = __fixed_eye(model)
    return model


def resnet32_cifar64_fixed_eye_no_bias():
    model = resnet32_cifar64()
    model = __no_bias(model)
    model = __fixed_eye(model)
    return model


def resnet32_cifar100_no_bias():
    model = rn32_cf100()
    model = __no_bias(model)
    return model


def rn32_fixed_orthoplex():
    model = rn32_cf100()
    classifier = nn.Linear(model.fc.in_features, 100)
    classifier.weight.data = generate_orthoplex(classifier.in_features, classifier.out_features)
    classifier.weight.requires_grad_(False)
    model.fc = classifier
    return model


def rn32_cf64_fx_or():
    model = rn32_cf100()
    classifier = nn.Linear(model.fc.in_features, 100)
    nn.init.orthogonal_(classifier.weight.data)
    classifier.weight.requires_grad_(False)
    model.fc = classifier
    return model
