import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100


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
