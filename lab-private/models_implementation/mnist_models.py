from collections import OrderedDict

import torch
import torch.nn as nn


__all__ = ['lenet5']


class _LambdaLayer(nn.Module):
    def __init__(self, l):
        super(_LambdaLayer, self).__init__()
        self.lbd = l

    def forward(self, x):
        return self.lbd(x)


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 50, 500),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.classifier(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # different b/c padding
            nn.Tanh(),  # seems like this
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # different b/c connection
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 120)
        return self.classifier(x)


def lenet5():
    return LeNet5()