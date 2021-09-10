import numpy as np
import seqtools
import torch
from torchvision import datasets as torch_datasets
from torchvision import transforms
from copy import deepcopy
import gzip
import pickle
import matplotlib.pyplot as plt
import utils
import random

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
dtype_default = np.float32
np.random.seed(1)


def load_snelson():
    x_train, y_train, x_test, noise_std = snelson(
        n_test=1000, x_test_lim=10, standardize_x=True, standardize_y=False
    )
    y_train = y_train.reshape([-1, 1])
    input_dim = x_train.shape[-1]
    output_dim = 1
    return x_train, y_train, x_test, noise_std


def snelson(n_test=500, x_test_lim=2.5, standardize_x=False, standardize_y=False):
    def _load_toydata(filename):
        try:
            with open(f"data/snelson/{filename}", "r") as f:
                return np.array(
                    [dtype_default(i) for i in f.read().strip().split("\n")]
                )
        except Exception as e:
            print(
                f"Error: {e.args[0]}\n\nWorking directory needs to be set to repository root."
            )

    x_train = _load_toydata("train_inputs")
    y_train = _load_toydata("train_outputs")

    mask = ((x_train < 1.5) | (x_train > 3)).flatten()
    x_train = x_train[mask]
    y_train = y_train[mask]

    idx = np.argsort(x_train)
    x_train = x_train[idx]
    y_train = y_train[idx]

    if standardize_x:
        x_train = (x_train - x_train.mean(0)) / x_train.std(0)
    if standardize_y:
        y_train = (y_train - y_train.mean(0)) / y_train.std(0)

    x_test = np.linspace(-x_test_lim, x_test_lim, n_test)[:, None]
    noise_std = 0.286

    return x_train[:, None], y_train[:, None], x_test, noise_std


def get_MNIST(batch_size, train_size, root="./data/"):
    image_size = [1, 32, 32, 1]
    num_classes = 10
    train_size_all = 60000
    train_batch = 1000

    transform_list = [transforms.Resize([32, 32]), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    transform = transforms.Compose(transform_list)

    train_dataset = torch_datasets.MNIST(
        root, train=True, download=True, transform=transform
    )
    test_dataset = torch_datasets.MNIST(
        root, train=False, download=True, transform=transform
    )

    idxs = np.arange(train_size_all)  # shuffle examples first
    rnd = np.random.RandomState(42)
    rnd.shuffle(idxs)
    train_idxs = idxs[:train_size]

    if train_size == train_size_all:
        train_sampler = None
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_batch, shuffle=False, num_workers=2,
    )

    return image_size, num_classes, train_loader, test_loader


def get_CIFAR10(batch_size, train_size, data_augmentation=True, root="./data/", crop_size=32):
    image_dim = crop_size
    image_size = [1, image_dim, image_dim, 3]
    num_classes = 10
    train_size_all = 50000
    test_batch = 1000

    if data_augmentation:
        if image_dim == 32:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(image_dim, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    if image_dim == 32:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    idxs = np.arange(train_size_all)  # shuffle examples first
    rnd = np.random.RandomState(42)
    rnd.shuffle(idxs)
    train_idxs = idxs[:train_size]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)

    train_dataset = torch_datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform
    )
    test_dataset = torch_datasets.CIFAR10(
        root, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=4, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch,
                                              shuffle=False, num_workers=4)

    return image_size, num_classes, train_loader, test_loader


def get_CIFAR100(batch_size, train_size, data_augmentation=True, root="./data/", crop_size=32):
    image_dim = crop_size
    image_size = [1, image_dim, image_dim, 3]
    num_classes = 100
    train_size_all = 50000
    test_batch = 1000

    if image_dim == 32:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    idxs = np.arange(train_size_all)  # shuffle examples first
    rnd = np.random.RandomState(42)
    rnd.shuffle(idxs)
    train_idxs = idxs[:train_size]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)

    train_dataset = torch_datasets.CIFAR100(
        root, train=True, download=True, transform=train_transform
    )
    test_dataset = torch_datasets.CIFAR100(
        root, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=4, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch,
                                              shuffle=False, num_workers=4)

    return image_size, num_classes, train_loader, test_loader


def collate_fn(batch):
    inputs = np.stack([x for x, _ in batch])
    targets = np.stack([y for _, y in batch])
    return inputs, targets


