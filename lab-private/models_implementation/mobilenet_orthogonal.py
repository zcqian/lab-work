import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNetV2
from math import sqrt


def mobilenetv2_nc_1k():
    model: MobileNetV2 = mobilenet_v2(pretrained=True)
    model.classifier = nn.Identity()
    model.features[18] = nn.Conv2d(320, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    return model


def mobilenetv2_orthogonal_nc_1k():
    model: MobileNetV2 = mobilenet_v2(pretrained=True)
    model.classifier = nn.Identity()
    convolution_classifier = nn.Conv2d(320, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    # here we try orthogonal initialization for the convolution filters
    # given that we want 320x1x1 x1000, we are unable to get a matrix that is really orthogonal
    # doing the below, according to the documentation, will initialize it semi-orthogonal,
    # flattening the trailing dimensions
    nn.init.orthogonal_(convolution_classifier.weight.data)
    model.features[18] = convolution_classifier
    return model


def mobilenetv2_nc_1k_relu_before_pool():
    model: MobileNetV2 = mobilenet_v2(pretrained=True)
    model.classifier = nn.Identity()
    model.features[18] = nn.Sequential(
        nn.Conv2d(320, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True),
        nn.ReLU(inplace=True)
    )
    return model


def _mobilenetv2_train_top_layers_only(model: MobileNetV2):
    assert type(model) is MobileNetV2
    model.requires_grad_(False)
    model.features[18].requires_grad_(True)
    model.features[17].requires_grad_(True)
    return model


def mobilenet_orthogonal_nc_1k_top_layers():
    return _mobilenetv2_train_top_layers_only(mobilenetv2_orthogonal_nc_1k())


def mobilenet_nc_1k_top_layers():
    return _mobilenetv2_train_top_layers_only(mobilenetv2_nc_1k())


def mobilenet_nc_1k_relu_before_pool_top_layers():
    return _mobilenetv2_train_top_layers_only(mobilenetv2_nc_1k_relu_before_pool())


def mobilenet_v2_orthogonal_all_layers_nc_1k():
    model = mobilenetv2_nc_1k()
    for m in model.modules():
        if type(m) is nn.Conv2d:
            nn.init.orthogonal_(m.weight)
    return model
