import torch.nn
import torchvision.models.resnet


def resnet18_tinyimagenet():
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,
                                             [2, 2, 2, 2], num_classes=200)
    return model


def resnet18_tinyimagenet_fixed_eye():
    model = resnet18_tinyimagenet()
    torch.nn.init.eye_(model.fc.weight.data)
    model.fc.weight.requires_grad_(False)
    return model
