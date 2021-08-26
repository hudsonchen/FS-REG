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


def get_CIFAR10(batch_size, train_size, data_augmentation=True, root="./data/"):
    image_dim = 32
    image_size = [1, 32, 32, 3]
    num_classes = 10
    train_size_all = 50000
    test_batch = 1000

    if data_augmentation:
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
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    test_transform = transforms.Compose(
        [
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


def get_CIFAR100(batch_size, train_size, data_augmentation=True, root="./data/"):
    image_dim = 32
    image_size = [1, 32, 32, 3]
    num_classes = 10
    train_size_all = 50000.

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

    train_dataset = torch_datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_dataset = torch_datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    trainloader_array = []
    for i, data in enumerate(trainloader, 0):
        x_batch = np.moveaxis(np.array(data[0], dtype=dtype_default), 1, 3)
        y_batch = utils.one_hot(np.array(data[1]), 10)
        trainloader_array.append([x_batch, y_batch])

    testloader_array = []
    for i, data in enumerate(testloader, 0):
        x_batch = np.moveaxis(np.array(data[0], dtype=dtype_default), 1, 3)
        y_batch = utils.one_hot(np.array(data[1]), 10)
        testloader_array.append([x_batch, y_batch])

    index = np.arange(len(trainloader_array))
    ratio = train_size / train_size_all
    partial_num = int(len(trainloader_array) * ratio)
    np.random.shuffle(index)
    partial_trainloader_array = []
    for idx in index[:partial_num]:
        partial_trainloader_array.append(trainloader_array[idx])
    return image_size, num_classes, partial_trainloader_array, trainloader_array, testloader_array


def collate_fn(batch):
    inputs = np.stack([x for x, _ in batch])
    targets = np.stack([y for _, y in batch])
    return inputs, targets


class PermutedMnistGenerator:
    def __init__(self, max_iter=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[0], self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = np.array(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:, perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
