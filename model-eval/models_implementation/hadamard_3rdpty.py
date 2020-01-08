import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from scipy.linalg import hadamard

class HadamardProj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_scale=None):
        super(HadamardProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = torch.from_numpy(hadamard(sz))
        if fixed_weights:
            self.proj = Variable(mat, requires_grad=False)
        else:
            self.proj = nn.Parameter(mat)

        init_scale = 1. / math.sqrt(self.output_size)

        if fixed_scale is not None:
            self.scale = Variable(torch.Tensor(
                [fixed_scale]), requires_grad=False)
        else:
            self.scale = nn.Parameter(torch.Tensor([init_scale]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                output_size).uniform_(-init_scale, init_scale))
        else:
            self.register_parameter('bias', None)

        self.eps = 1e-8

    def forward(self, x):
        # TODO: fix this, although it no longer crashes on Windows + multi GPU
        # I don't think doing assignments here in multi-process/thread setups
        # make much sense
        # I don't know why the old code works on single GPU (single process)
        # but not multi GPU, on Windows.
        #
        # I have tested that it works on Windows/Linux with single GPU, 
        # but I need to figure out if it works on Linux with multiple GPU.
        # 
        # If Linux + Multi-GPU works, then it could be an issue with spawn/fork different
        # Anyways, doing the assignment here does not look good, and is not necessary
        # as the type does not change very often, and is usually just single precision float
        if not isinstance(self.scale, nn.Parameter):
            self.scale = nn.Parameter(self.scale.type_as(x))
        x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)

        out = -self.scale * \
            nn.functional.linear(x, w[:self.output_size, :self.input_size])
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class Proj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, init_scale=10):
        super(Proj, self).__init__()
        if init_scale is not None:
            self.weight = nn.Parameter(torch.Tensor(1).fill_(init_scale))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size).fill_(0))
        self.proj = Variable(torch.Tensor(
            output_size, input_size), requires_grad=False)
        torch.manual_seed(123)
        nn.init.orthogonal(self.proj)

    def forward(self, x):
        w = self.proj.type_as(x)
        x = x / x.norm(2, -1, keepdim=True)
        out = nn.functional.linear(x, w)
        if hasattr(self, 'weight'):
            out = out * self.weight
        if hasattr(self, 'bias'):
            out = out + self.bias.view(1, -1)
        return out

