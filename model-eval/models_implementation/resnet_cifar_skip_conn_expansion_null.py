from types import MethodType

from torch import nn as nn
from torch.nn import functional as F

from models_implementation.resnet_cifar import ResNet, BasicBlock
from models_implementation.clsf_utils import __no_bias, __fixed_eye


__all__ = ['rn32_cf100_exnull', 'rn32_cf100_exnull_no_bias',
           'rn32_cf100_exnull_fixed_eye', 'rn32_cf100_exnull_fixed_eye_no_bias',
           'rn32_cf100_exnull_no_last_relu', 'rn32_cf100_exnull_no_last_relu_fixed_eye',
           'rn32_cf100_exnull_no_last_relu_fixed_eye_no_bias',
           ]


def rn32_cf100_exnull():
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=100)
    model.layer3[4].conv2 = nn.Conv2d(64, 100, kernel_size=3, stride=1, padding=1, bias=False)
    model.layer3[4].bn2 = nn.BatchNorm2d(100)
    model.fc = nn.Linear(100, 100)

    def forward_without_skip_conn(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

    model.layer3[4].forward = MethodType(forward_without_skip_conn, model.layer3[4])
    return model


def rn32_cf100_exnull_no_bias():
    model = rn32_cf100_exnull()
    model = __no_bias(model)
    return model


def rn32_cf100_exnull_fixed_eye():
    model = rn32_cf100_exnull()
    model = __fixed_eye(model)
    return model


def rn32_cf100_exnull_fixed_eye_no_bias():
    model = rn32_cf100_exnull()
    model = __no_bias(model)
    model = __fixed_eye(model)
    return model


def rn32_cf100_exnull_no_last_relu():
    model = rn32_cf100_exnull()

    def forward_without_skip_conn_no_relu(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        return out

    model.layer3[4].forward = MethodType(forward_without_skip_conn_no_relu, model.layer3[4])
    return model


def rn32_cf100_exnull_no_last_relu_fixed_eye_no_bias():
    model = rn32_cf100_exnull_no_last_relu()
    model = __no_bias(model)
    model = __fixed_eye(model)
    return model


def rn32_cf100_exnull_no_last_relu_fixed_eye():
    model = rn32_cf100_exnull_no_last_relu()
    model = __fixed_eye(model)
    return model
