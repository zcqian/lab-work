from collections import OrderedDict
import inspect

import torch.nn as nn
from torchvision.models import resnet18, ResNet

from .resnet_imagenet import Bias
from .hadamard_3rdpty import HadamardProj


class LambdaLayer(nn.Module):
    """Module/Layer that encapsulates a single function for PyTorch

    This is to make it easier to a lambda in an nn.Sequential() container.
    """

    def __init__(self, lm):
        """

        Args:
            lm (Callable): function to use/call when the module is called.
        """
        super().__init__()
        self.lm = lm
        # this is because I want to see whatever the anonymous function is
        # but I do not know how to parse python syntax or want to learn to write a parser now
        self.src = inspect.getsourcelines(self.lm)
        if len(self.src[0]) == 1:
            module_code_str: str = self.src[0][0]
            lam_start_pos = module_code_str.find("lambda")
            # the case where def f(x): ... is a one liner
            if lam_start_pos == -1 and module_code_str[:4] == 'def ':
                xtr_repr = module_code_str.strip('\r\n')
            else:
                xtr_repr = module_code_str[lam_start_pos:]  # finds the start of "lambda..."
                xtr_repr = xtr_repr.strip(')\r\n')  # removes trailing parenthesis and newlines
            self.xtr_repr = xtr_repr
        else:
            self.xtr_repr = '(not lambda)'

    def forward(self, *input):
        return self.lm(*input)

    def extra_repr(self) -> str:
        return self.xtr_repr


class SoftAttentionPooling(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels

        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, middle_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(middle_channels, 1, kernel_size=1)
        )
    
    def forward(self, x):
        n, c = x.size(0), x.size(1)
        x = x.view(n, c, -1)
        summarized = self.attention(x).view(n, -1)
        att = nn.functional.softmax(summarized, 1)
        x = (x * att.unsqueeze(1)).sum(2)
        return x


class RepackagedResNet18(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        orig_resnet: ResNet = resnet18(pretrained=pretrained)
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', orig_resnet.conv1),
                ('bn1', orig_resnet.bn1),
                ('relu1', orig_resnet.relu),
                ('maxpool1', orig_resnet.maxpool),
                ('layer1', orig_resnet.layer1),
                ('layer2', orig_resnet.layer2),
                ('layer3', orig_resnet.layer3),
                ('layer4', orig_resnet.layer4)
            ]))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            orig_resnet.fc
        )

    def forward(self, x):
        x = self.features(x)
        y = self.classifier(x)
        return y


def rn18_3x3clsf():
    model = RepackagedResNet18(pretrained=False)
    model.classifier = nn.Sequential(
        OrderedDict([
        ('conv', nn.Conv2d(512, 1000, kernel_size=3)),
        ('pool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flatten', nn.Flatten())
    ]))
    return model

def rn18_512_scratch():
    model = RepackagedResNet18(pretrained=False)
    model.classifier[2] = nn.Linear(512, 512)
    return model

def rn18_512_id_scratch():
    model = RepackagedResNet18(pretrained=False)
    model.classifier[2] = nn.Identity()
    return model

def rn18_512_id_bias_scratch():
    model = rn18_512_id_scratch()
    model.classifier.add_module('bias', Bias(512, 512))
    return model

def rn18_512_hadamard_scratch():
    model = RepackagedResNet18(pretrained=False)
    model.classifier[2] = HadamardProj(512, 512)
    return model


def rn18_l4_1a_nc_1k():
    model = RepackagedResNet18(pretrained=True)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    model.features.layer4[1].bn2 = nn.Sequential()
    model.features.layer4[1].conv2 = nn.Conv2d(512, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    pad_size = int((1000 - 512) / 2)
    pad_param = (0, 0, 0, 0, pad_size, pad_size)
    model.features.layer4[1].downsample = LambdaLayer(lambda x: nn.functional.pad(x, pad_param, 'constant', 0))
    return model


def rn18_l4_1a_orthogonal_nc_1k():
    model = rn18_l4_1a_nc_1k()
    nn.init.orthogonal_(model.features.layer4[1].conv2.weight)
    return model


def rn18_l4_1a_all_conv_orthogonal_nc_1k():
    model = rn18_l4_1a_nc_1k()
    for m in model.modules():
        if type(m) is nn.Conv2d:
            nn.init.orthogonal_(m.weight)
    return model


def rn18_l4_1a_relu_before_pool_nc_1k():
    model = rn18_l4_1a_nc_1k()
    model.classifier = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return model

def rn18_l4_1a_nc_1k_scratch():
    model = RepackagedResNet18(pretrained=False)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    model.features.layer4[1].bn2 = nn.Sequential()
    model.features.layer4[1].conv2 = nn.Conv2d(512, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    pad_size = int((1000 - 512) / 2)
    pad_param = (0, 0, 0, 0, pad_size, pad_size)
    model.features.layer4[1].downsample = LambdaLayer(lambda x: nn.functional.pad(x, pad_param, 'constant', 0))
    return model

def rn18_l4_1a_maxpool_nc_1k():
    model = rn18_l4_1a_nc_1k_scratch()
    model.classifier = nn.Sequential(
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten()
    )
    return model


def rn18_l4_1a_lppool_p2():
    model = rn18_l4_1a_nc_1k_scratch()
    model.classifier = nn.Sequential(
        nn.LPPool2d(2, kernel_size=7),
        LambdaLayer(lambda x: x / 7),
        nn.Flatten()
    )
    return model

def rn18_l4_1a_lppool_p1_5():
    model = rn18_l4_1a_nc_1k_scratch()
    model.classifier = nn.Sequential(
        nn.LPPool2d(1.5, kernel_size=7),
        LambdaLayer(lambda x: x / 13.390518),
        nn.Flatten()
    )
    return model

def rn18_l4_1a_lppool(p: float):
    n = 49 ** (1/p)
    model = rn18_l4_1a_nc_1k_scratch()
    model.classifier = nn.Sequential(
        nn.LPPool2d(p, kernel_size=7),
        LambdaLayer(lambda x: x / n),
        nn.Flatten()
    )
    return model

def rn18_l4_1a_lppool_p0_5():
    return rn18_l4_1a_lppool(0.5)

def rn18_l4_1a_lppool_p1_0():
    return rn18_l4_1a_lppool(1.0)

def rn18_l4_1a_lppool_p4_0():
    return rn18_l4_1a_lppool(4.0)


def rn18_l4_1a_soft_attention_pool(units: int):
    model = rn18_l4_1a_nc_1k_scratch()
    model.classifier = nn.Sequential(
        SoftAttentionPooling(1000, units)
    )
    return model

def rn18_l4_1a_soft_attention_pool_32():
    return rn18_l4_1a_soft_attention_pool(32)

def rn18_l4_1a_soft_attention_pool_64():
    return rn18_l4_1a_soft_attention_pool(64)

def rn18_l4_1a_soft_attention_pool_128():
    return rn18_l4_1a_soft_attention_pool(128)

def rn18_l4_1a_soft_attention_pool_256():
    return rn18_l4_1a_soft_attention_pool(256)

def rn18_l4_1a_soft_attention_pool_512():
    return rn18_l4_1a_soft_attention_pool(512)

def rn18_l4_1a_soft_attention_pool_1024():
    return rn18_l4_1a_soft_attention_pool(1024)
