from .resnet_orthogonal import RepackagedResNet18, LambdaLayer
import torch.nn as nn



def rn18_removetwolayers():
    model = RepackagedResNet18(pretrained=False)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    pad_size = int((1000 - 512) / 2)
    pad_param = (0, 0, 0, 0, pad_size, pad_size)
    model.features.layer4[1].downsample = LambdaLayer(lambda x: nn.functional.pad(x, pad_param, 'constant', 0))
    model.features.layer4[1].bn1 = nn.Sequential()
    model.features.layer4[1].relu = nn.Sequential()
    model.features.layer4[1].conv2 = nn.Sequential()
    model.features.layer4[1].bn2 = nn.Sequential()
    model.features.layer4[1].conv1 = nn.Conv2d(512, 1000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    return model
