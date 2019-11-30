import os
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data import Subset
import math


__all__ = []


def cifar100():
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    dataset_train = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    dataset_val = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_train, dataset_val


def cifar10():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_train = CIFAR10(root=os.path.expanduser('~/Datasets/cifar10'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    dataset_val = CIFAR10(root=os.path.expanduser('~/Datasets/cifar10'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_train, dataset_val


def cifar_imbalanced(num_of_cats, imbalance):
    if num_of_cats == 10:
        dataset_train, dataset_val = cifar10()
    elif num_of_cats == 100:
        dataset_train, dataset_val = cifar100()
    else:
        raise NotImplementedError
    imbalance_exp = math.exp(math.log(1/imbalance)/(num_of_cats - 1))
    dataset = dataset_train
    base_count = len(dataset_train) // num_of_cats
    counts = [int(round(base_count * imbalance_exp ** idx)) for idx in range(num_of_cats)]
    list_of_selected_idx = []
    for cat_number in range(num_of_cats):
        select_idx = [idx for idx in range(len(dataset)) if dataset.targets[idx] == cat_number]
        select_idx = select_idx[:counts[cat_number]]
        list_of_selected_idx += select_idx
    dataset_train = Subset(dataset, list_of_selected_idx)
    return dataset_train, dataset_val


for imb in [10, 20, 50, 100, 200]:
    exec('cifar10_imbalance_{0} = lambda: cifar_imbalanced(10, {0})'.format(imb))
    exec('cifar100_imbalance_{0} = lambda: cifar_imbalanced(100, {0})'.format(imb))
    __all__ += [f'cifar10_imbalance_{imb}', f'cifar100_imbalance_{imb}']
