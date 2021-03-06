import os
from textwrap import dedent
from typing import Callable, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, SVHN, MNIST, ImageNet


def mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset_dir = os.path.expanduser('~/Datasets/mnist')
    dataset_train = MNIST(root=dataset_dir, train=True, download=True, transform=transform)
    dataset_val = MNIST(root=dataset_dir, train=False, transform=transform)
    return dataset_train, dataset_val


def imagenet1k():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_dir = os.path.expanduser('~/Datasets/imagenet')
    dataset_train = ImageNet(
        dataset_dir, split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_val = ImageNet(
        dataset_dir, split='val',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_train, dataset_val


def imagenet1k_10cropvalonly():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_dir = os.path.expanduser('~/Datasets/imagenet')
    val_dir = os.path.join(dataset_dir, 'val')
    dataset_train = None
    dataset_val = ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
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
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    dataset_val = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    return dataset_train, dataset_val


def cifar100_jitter():
    jitter_param = 0.4
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    dataset_train = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=True, transform=transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    dataset_val = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'), train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    return dataset_train, dataset_val


def cifar100_altaug(transform_t, transform_v):
    dataset_train = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'),
                             train=True, transform=transform_t, download=True)
    dataset_val = CIFAR100(root=os.path.expanduser('~/Datasets/cifar100'),
                           train=False, transform=transform_v)
    return dataset_train, dataset_val


def cifar100_aug2():
    """ CIFAR-100, ImageNet statistics normalization, Random Crop then Mirror"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_t = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_v = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return cifar100_altaug(transform_t, transform_v)


def cifar100_aug3():
    """ CIFAR-100, CIFAR-100 statistics normalization, Random Crop then Mirror"""
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    transform_t = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_v = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return cifar100_altaug(transform_t, transform_v)


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


def _ordered_subset(orig_dataset: Callable[[None], Tuple[Dataset, Dataset]], num_categories: int):
    ds_trn, ds_val = orig_dataset()
    accept_categories = list(range(num_categories))
    subset_idx = [i for i in range(len(ds_trn)) if ds_trn.targets[i] in accept_categories]
    ds_trn = Subset(ds_trn, subset_idx)
    subset_idx = [i for i in range(len(ds_val)) if ds_val.targets[i] in accept_categories]
    ds_val = Subset(ds_val, subset_idx)
    return ds_trn, ds_val


def cifar50():
    return _ordered_subset(cifar100, 50)


def cifar64_ordered():
    return _ordered_subset(cifar100, 64)


def svhn_normalize_as_cf100_no_rnd():
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset_train = SVHN(root=os.path.expanduser('~/Datasets/svhn'), split='train',
                         transform=transform, download=True)
    dataset_val = SVHN(root=os.path.expanduser('~/Datasets/svhn'), split='test',
                       transform=transform, download=True)
    return dataset_train, dataset_val


# The dataset from ODIN code
for ds_name in ["TinyImageNet_resize", "TinyImageNet_crop", "iSUN"]:
    code = f"""\
    def odin_{ds_name}():
        normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = ImageFolder(os.path.expanduser("~/Datasets/ODIN/{ds_name}"), transform)
        return None, dataset
    """
    exec(dedent(code))


def rnd_gaussian_32x32_rgb():
    data = torch.randn((10000, 3, 32, 32)) + 0.5
    data = torch.clamp(data, 0, 1)
    data[:, 0] = (data[:, 0] - 0.5071) / 0.2673
    data[:, 1] = (data[:, 1] - 0.4865) / 0.2564
    data[:, 2] = (data[:, 2] - 0.4409) / 0.2762
    return None, TensorDataset(data, torch.zeros(10000).long())


def rnd_uniform_32x32_rgb():
    data = torch.rand((10000, 3, 32, 32))
    data[:, 0] = (data[:, 0] - 0.5071) / 0.2673
    data[:, 1] = (data[:, 1] - 0.4865) / 0.2564
    data[:, 2] = (data[:, 2] - 0.4409) / 0.2762
    return None, TensorDataset(data, torch.zeros(10000).long())


def places365():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # the above normalization seems strange, why use ImageNet values, or are they the same
    dir_train = os.path.expanduser('~/Datasets/places365_standard/train')
    dir_val = os.path.expanduser('~/Datasets/places365_standard/val')
    dataset_train = ImageFolder(root=dir_train,
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
    dataset_val = ImageFolder(root=dir_val,
                              transform=transforms.Compose([
                                  transforms.CenterCrop(224),  # they said it was already 256x256
                                  transforms.ToTensor(),
                                  normalize,
                              ]))
    return dataset_train, dataset_val

for imgnet_subset_count in [50, 100, 8, 16, 32, 64, 128, 256, 512]:
    code = f"""\
        def imagenet_{imgnet_subset_count}():
            return _ordered_subset(imagenet1k, {imgnet_subset_count})
    """
    exec(dedent(code))


def imagenet_100_scale_020_100():
    ds_trn, ds_val = imagenet_100()
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.20, 1.00)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds_trn.dataset.transform = transform
    return ds_trn, ds_val
