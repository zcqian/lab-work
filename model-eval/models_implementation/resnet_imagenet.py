import math
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, mobilenet_v2, mnasnet0_5, mnasnet1_0
from typing import Callable, List


from .clsf_utils import generate_cube_random, generate_orthoplex


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ClassifierBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_categories, stride=1):
        super(ClassifierBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, num_categories)
        self.bn2 = nn.BatchNorm2d(num_categories)

        # Generate the skip connection
        w = generate_cube_random(inplanes, num_categories)
        w = w.view(num_categories, inplanes, 1, 1)
        conv = nn.Conv2d(inplanes, num_categories, kernel_size=1, stride=stride, bias=False)
        conv.weight.data = w.float()
        conv.weight.requires_grad_(False)
        self.downsample = nn.Sequential(
            conv,
            nn.BatchNorm2d(num_categories)
        )
        del w, conv
        self.stride = stride

        # initialize the module in the same way
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "downsample" not in name:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if name == "bn2":
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Bias(nn.Module):
    def __init__(self, in_features, out_features: int):
        # in_features is only used for initializing the bias
        super(Bias, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = x + self.bias
        return out

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


def rn18_imgnet_fixed_eye():
    model = resnet18()
    model.fc = Bias(512, 1000)
    model.layer4[1] = ClassifierBlock(512, 512, 1000, 1)
    return model


def rn18_nc(num_categories: int = 1000):
    model = resnet18(pretrained=True)
    model.fc = nn.Sequential()
    for p in model.parameters():
        p.requires_grad_(False)
    model.layer4[1] = ClassifierBlock(512, 512, num_categories, 1)
    return model


def rn18_nc_1k():
    return rn18_nc(1000)


def rn18_nc_100():
    return rn18_nc(100)


def rn18_nc_365():
    m = rn18_365()
    m.load_state_dict(
        torch.load(os.path.expanduser('~/Pretrained/rn18_places365.pt'))
    )
    m.fc = nn.Sequential()
    for p in m.parameters():
        p.requires_grad_(False)
    m.layer4[1] = ClassifierBlock(512, 512, 365, 1)
    return m


def rn18_365():
    m = resnet18()
    m.fc = nn.Linear(512, 365)
    return m


def rn18_500():
    m = resnet18()
    m.fc = nn.Linear(512, 500)
    return m


def rn18_pt_fixed_(num_categories: int, fixed_generator: Callable[[int, int], torch.Tensor],
                   layers_to_train: List[str]) -> nn.Module:
    model = resnet18(pretrained=True)
    classifier = nn.Linear(model.fc.in_features, num_categories)
    classifier.weight.data = fixed_generator(classifier.in_features, classifier.out_features)
    classifier.weight.requires_grad_(False)
    for name, param in model.named_parameters():
        need_train = False
        for train_name in layers_to_train:
            if train_name in name:
                need_train = True
                break
            else:
                pass
        if not need_train:
            param.requires_grad_(False)
        else:
            pass
    model.fc = classifier
    return model


def rn18_fixed_orthoplex_1k():
    return rn18_pt_fixed_(1000, generate_orthoplex, ["layer3", "layer4"])


def rn18_fixed_orthoplex_100():
    return rn18_pt_fixed_(100, generate_orthoplex, ["layer3", "layer4"])


def rn18_fixed_cube_rnd_1k():
    return rn18_pt_fixed_(1000, generate_cube_random, ["layer4.1"])


def mobilenet_v2_nc_1k_scratch():
    """Modified MobileNet v2 with no last fully connected layer (fixed identity matrix output)

    Initializes with a ImageNet pretrained MobileNet v2 model, replaces the classifier with nothing,
    replaces the last ConvBNReLU with a Conv2D, and only makes the last few layers trainable.

    Returns:
        Modified MobileNet v2
    """
    m = mobilenet_v2(pretrained=False, progress=False)
    m.classifier = nn.Sequential()  # ditch the FC layer
    m.features[18] = nn.Conv2d(320, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    return m


def mobilenet_v2_nc_1k():
    """Modified MobileNet v2 with no last fully connected layer (fixed identity matrix output)

    Initializes with a ImageNet pretrained MobileNet v2 model, replaces the classifier with nothing,
    replaces the last ConvBNReLU with a Conv2D, and only makes the last few layers trainable.

    Returns:
        Modified MobileNet v2
    """
    m = mobilenet_v2(pretrained=True, progress=False)
    for p in m.parameters():
        p.requires_grad_(False)
    m.classifier = nn.Sequential()  # ditch the FC layer
    m.features[18] = nn.Conv2d(320, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    for p in m.features[17].parameters():
        p.requires_grad_(True)
    return m


def mobilenet_v2_nc_365():
    m = mobilenet_v2(pretrained=True, progress=False)
    m.classifier = nn.Sequential()  # ditch the FC layer
    m.features[18] = nn.Conv2d(320, 365, kernel_size=(1, 1), stride=(1, 1), bias=True)
    return m


def mobilenet_v2_500():
    m = mobilenet_v2(pretrained=True, progress=False)
    m.classifier = nn.Linear(1280, 500)
    return m


def mobilenet_v2_365():
    m = mobilenet_v2(pretrained=True, progress=False)
    m.classifier = nn.Linear(1280, 365)
    return m


def mnasnet_0_5_nc_1k():
    m = mnasnet0_5(pretrained=True, progress=False)
    m.classifier = nn.Sequential()  # ditch the FC layer
    del m.layers[14:]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.layers[13].parameters():
        p.requires_grad_(True)
    m.layers.add_module('14', nn.Conv2d(160, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True))
    return m


def mnasnet1_0_nc(num_classes: int):
    m = mnasnet1_0(pretrained=True, progress=False)
    m.classifier = nn.Identity()  # ditch the FC layer
    del m.layers[14:]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.layers[13].parameters():
        p.requires_grad_(True)
    m.layers.add_module('14', nn.Conv2d(320, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True))
    return m

def mnasnet1_0_nc_1k():
    return mnasnet1_0_nc(1000)

def mnasnet1_0_nc_365():
    return mnasnet1_0_nc(365)


def mnasnet1_0_365():
    m = mnasnet1_0(pretrained=False, progress=False)
    m.classifier = nn.Linear(1280, 365)
    return m


def rn50_nc_1k():
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()  # not exposed properly?
    model.layer4[2].bn3 = nn.Identity()
    model.layer4[2].relu = nn.Identity()
    model.layer4[2].conv3 = nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.layer4.parameters():
        p.requires_grad_(True)
    model.layer4[2].downsample = LambdaLayer(lambda x: x[:, :1000, :, :])
    return model
    

def rn50_nc_1k_scratch():
    model = resnet50(pretrained=False)
    model.fc = nn.Identity()  # not exposed properly?
    model.layer4[2].bn3 = nn.Identity()
    model.layer4[2].relu = nn.Identity()
    model.layer4[2].conv3 = nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    model.layer4[2].downsample = LambdaLayer(lambda x: x[:, :1000, :, :])
    return model
		


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def mobilenet_v2_nc_1k_dropout(p: float):
    m = mobilenet_v2_nc_1k()
    m.features[18] = nn.Sequential(
                         nn.Dropout2d(p=p, inplace=True),
                         m.features[18],
                     )
    return m


def mobilenet_v2_nc_1k_dropout_01():
    return mobilenet_v2_nc_1k_dropout(0.1)


def mobilenet_v2_nc_1k_dropout_02():
    return mobilenet_v2_nc_1k_dropout(0.2)

def mobilenet_v2_nc_1k_dropout_04():
    return mobilenet_v2_nc_1k_dropout(0.4)

def mobilenet_v2_nc_1k_dropout_005():
    return mobilenet_v2_nc_1k_dropout(0.05)

def mobilenet_v2_pt():
    m = mobilenet_v2(pretrained=True)
    m = m.requires_grad_(False)
    return m

def mobilenet_v2_1k():
    m = mobilenet_v2(pretrained=False)
    return m

