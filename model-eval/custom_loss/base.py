import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class CustomLoss(nn.Module):
    def __init__(self, criterion: _Loss):
        assert isinstance(criterion, _Loss)
        super().__init__()
        self.criterion = criterion
    
    def forward(self, *input):
        raise NotImplementedError


class VanillaLoss(CustomLoss):
    def __init__(self, criterion: _Loss):
        super().__init__(criterion)
    
    def forward(self, *input, **kwargs):
        return self.criterion(*input, **kwargs)
