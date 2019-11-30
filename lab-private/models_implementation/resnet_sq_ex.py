from models_implementation.resnet_cifar import ResNet, BasicBlock
import torch.nn as nn
from textwrap import dedent


__all__ = []

for num_categories in [10, 100, 200]:
    code = f"""
    def rn32_{num_categories}_fc_sq_ex():
        model = ResNet(BasicBlock, [5, 5, 5])
        model.fc = nn.Sequential(
            nn.Linear(64, 16, bias=False),
            nn.Linear(16, {num_categories}, bias=True),
        )
        return model
    """
    exec(dedent(code))
    __all__ += [f'rn32_{num_categories}_fc_sq_ex']
