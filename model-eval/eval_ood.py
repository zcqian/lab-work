#!/usr/bin/env python3

import argparse
import models
import datasets
import torch
import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import OrderedDict


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__dict__
                       if name.islower() and not name.startswith('__')
                       and callable(datasets.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-a', '--arch', metavar='ARCH', required=True,
                    choices=model_names,
                    help=f"model architecture: {'/'.join(model_names)}")
parser.add_argument('-d', '--dataset', metavar='DATASET', required=True,
                    help=f"dataset to use: {'/'.join(dataset_names)}")
parser.add_argument('-c', '--checkpoint', type=str, metavar='CHECKPOINT', required=True,
                    help="use specified checkpoint archive file")
parser.add_argument('-o', '--output', type=str, metavar='OUTPUT FILENAME', required=True,
                    help="file to write output")

args = parser.parse_args()

# reconstruct model
model = models.__dict__[args.arch]()
state_dict = torch.load(args.checkpoint, map_location='cpu')['state_dict']

if list(state_dict.keys())[0].startswith("module"):
    old_state_dict = state_dict
    state_dict = OrderedDict()
    for k in old_state_dict:
        state_dict[k[7:]] = old_state_dict[k]


model.load_state_dict(state_dict)
model.eval()

# dataset
_, dataset = datasets.__dict__[args.dataset]()
dl = DataLoader(dataset, batch_size=256, num_workers=4)
out = []

with torch.no_grad():
    for idx, (data, _) in tqdm.tqdm(enumerate(dl), desc='MiniBatch', total=len(dl), unit='batch'):
        r = model(data)
        r = F.softmax(r, dim=1)
        r = r.max(dim=1)[0]
        out.append(r.detach().clone())
        if __debug__ and idx > 2:
            break

out = torch.cat(out)
torch.save(out, args.output)
