import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from main import accuracy
from main import AverageMeter, ProgressMeter


def validate_10crop(val_loader: DataLoader, model: nn.Module):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # we use only this criterion at the moment 
    criterion = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # this assumes input is ncrop
            bs, ncrops, c, h, w = input.size()
            # compute output
            temp_output = model(input.view(-1, c, h, w))
            output = temp_output.view(bs, ncrops, -1).mean(1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


def dist_state_dict_to_conventional(state_dict: OrderedDict) -> OrderedDict:
    if list(state_dict.keys())[0].startswith("module"):
        old_state_dict = state_dict
        state_dict = OrderedDict()
        for k in old_state_dict:
            state_dict[k[7:]] = old_state_dict[k]  # 7 is len('module.')
    return state_dict
