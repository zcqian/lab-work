from torch.optim.lr_scheduler import _LRScheduler


class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, slope, last_epoch=-1):
        self.slope = slope
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr - self.slope * self.last_epoch for base_lr in self.base_lrs]
