import inspect

import torch

from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
import torchvision.models.resnet

__all__ = ['ResNetAltWidth', 'rn50_less_wide']

# obtain source code for ResNet implementation
src = inspect.getsource(ResNet)

# monkey patch the source code
# TODO: add more checks
# TODO: add more sizes(?)
# maybe using AST would have been better
if src.count('self.inplanes = 64') != 1:
    raise NotImplementedError("Can't monkey patch ResNet")

# 60 gives comparable total parameters as RN50 w/o FC
sizes = [60 * (2 ** i) for i in range(4)]

src = src.replace('self.inplanes = 64', f'self.inplanes = {sizes[0]}')
src = src.replace('self.layer1 = self._make_layer(block, 64', f'self.layer1 = self._make_layer(block, {sizes[0]}')
src = src.replace('self.layer2 = self._make_layer(block, 128', f'self.layer2 = self._make_layer(block, {sizes[1]}')
src = src.replace('self.layer3 = self._make_layer(block, 256', f'self.layer3 = self._make_layer(block, {sizes[2]}')
src = src.replace('self.layer4 = self._make_layer(block, 512', f'self.layer4 = self._make_layer(block, {sizes[3]}')
src = src.replace('self.fc = nn.Linear(512', f'self.fc = nn.Linear({sizes[3]}')


# re-evaluate the code and obtain the modified ResNet class
ns = {'torch': torch,
      'nn': torch.nn, 
      'conv1x1': torchvision.models.resnet.conv1x1,
      'conv3x3': torchvision.models.resnet.conv3x3
     }
co = compile(src, inspect.getfile(ResNet), mode='exec')
exec(co, ns)
ResNetAltWidth = ns['ResNet']

def rn50_less_wide():
    model = ResNetAltWidth(Bottleneck,  [3, 4, 6, 3])
    return model
