import torch.nn as nn

from .resnet_orthogonal import RepackagedResNet18


def rn18_clsf_fc_scratch():
    model = RepackagedResNet18(pretrained=False)
    return model


def rn18_clsf_conv_scratch():
    model = rn18_clsf_fc_scratch()
    conv = nn.Conv2d(512, 1000, 1)
    conv.weight.data = model.classifier[2].weight.data.view(1000, 512, 1, 1)
    conv.bias.data = model.classifier[2].bias.data
    
    model.classifier = nn.Sequential(
        conv,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return model

