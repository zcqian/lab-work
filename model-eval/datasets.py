import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Subset
from datasets_implementation import *


def imagenet1k():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_dir = os.path.expanduser('~/Datasets/imagenet')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    dataset_train = ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_val = ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_train, dataset_val


def tinyimagenet():
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2770, 0.2691, 0.2821])
    dataset_dir = os.path.expanduser('~/Datasets/tinyimagenet')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    dataset_train = ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_val = ImageFolder(val_dir, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    return dataset_train, dataset_val


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


def cifar64_rand():
    import random
    rnd = random.Random()
    rnd.seed()
    all_labels = list(range(100))
    rnd.shuffle(all_labels)
    select_labels = all_labels[:64]
    remap_dict = {all_labels[orig_label]: orig_label for orig_label in range(100)}
    remap_label_transform = transforms.Lambda(lambda x: remap_dict[x])
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    dataset = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), target_transform=remap_label_transform, download=True)
    select_idx = [idx for idx in range(len(dataset)) if dataset.targets[idx] in select_labels]
    dataset_train = Subset(dataset, select_idx)

    dataset = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), target_transform=remap_label_transform)
    select_idx = [idx for idx in range(len(dataset)) if dataset.targets[idx] in select_labels]
    dataset_test = Subset(dataset, select_idx)
    return dataset_train, dataset_test


def cifar64_ordered():
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    dataset = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    select_idx = [idx for idx in range(len(dataset)) if dataset.targets[idx] < 64]
    dataset_train = Subset(dataset, select_idx)

    dataset = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    select_idx = [idx for idx in range(len(dataset)) if dataset.targets[idx] < 64]
    dataset_test = Subset(dataset, select_idx)
    return dataset_train, dataset_test


def svhn_normalize_as_cf100():
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    dataset_train = SVHN(root=os.path.expanduser('~/Datasets/svhn'), split='train',
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, 4),
                             transforms.ToTensor(),
                             normalize,
                         ]), download=True)
    dataset_val = SVHN(root=os.path.expanduser('~/Datasets/svhn'), split='test',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize,
                       ]), download=True)
    return dataset_train, dataset_val
