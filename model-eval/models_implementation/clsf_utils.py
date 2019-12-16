import math
import random

import torch


def generate_hadamard(in_features, out_features):
    from scipy.linalg import hadamard
    n = math.ceil(math.log2(max(in_features, out_features)))
    h = hadamard(2**n)
    return torch.tensor(h[:out_features, :in_features])


def generate_orthoplex(in_features, out_features):
    t = torch.zeros(out_features, in_features)
    for row in range(out_features):
        col = row // 2
        t[row, col] = (-1) ** row
    return t


def generate_cube_ordered(in_features, out_features):
    t = torch.ones(out_features, in_features)
    for row in range(out_features):
        binary_coded = f'{{0:0{in_features}b}}'
        binary_coded = binary_coded.format(row)
        for col, val in enumerate(binary_coded):
            t[row, col] = (-1)**int(val)
    return t / math.sqrt(in_features)


def generate_cube_random(in_features, out_features):
    t = torch.ones(out_features, in_features)
    # FIXME: This causes ValueError: Maximum allowed size exceeded
    rnd_vector_numbers = set()
    while len(rnd_vector_numbers) < out_features:
        rnd_vector_numbers.add(random.randint(0, 2**in_features - 1))
    rnd_vector_numbers = list(rnd_vector_numbers)
    for row in range(out_features):
        binary_coded = f'{{0:0{in_features}b}}'
        binary_coded = binary_coded.format(rnd_vector_numbers[row])
        for col, val in enumerate(binary_coded):
            t[row, col] = (-1)**int(val)
    return t / math.sqrt(in_features)


def __fixed_eye(model):
    torch.nn.init.eye_(model.fc.weight.data)
    model.fc.weight.requires_grad_(False)
    return model


def __no_bias(model):
    model.fc = torch.nn.Linear(model.fc.in_features, model.fc.out_features, bias=False)
    return model
