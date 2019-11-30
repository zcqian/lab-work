import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0


def shufflenet_v2_x0_5_nc_1k():
    m = shufflenet_v2_x0_5(pretrained=True)
    m.fc = nn.Identity()
    del m.conv5[2], m.conv5[1]  # remove ReLU and BN
    m.conv5[0] = nn.Conv2d(192, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    return m


def shufflenet_v2_x1_0_nc_1k():
    m = shufflenet_v2_x1_0(pretrained=True)
    m.fc = nn.Identity()
    del m.conv5[2], m.conv5[1]  # remove ReLU and BN
    m.conv5[0] = nn.Conv2d(464, 1000, kernel_size=(1, 1), stride=(1, 1), bias=True)
    return m


def shufflenet_v2_x0_5_nc_1k_scratch():
    model = shufflenet_v2_x0_5_nc_1k()
    for m in model.modules():
        try:
            m.reset_parameters()
        except:
            pass
    return model


def shufflenet_v2_x0_5_pretrained():
    return shufflenet_v2_x0_5(pretrained=True, progress=True)
