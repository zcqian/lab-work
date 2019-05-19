import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


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