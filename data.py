import numpy as np
import numpy.random as npr
import torch as th
import torch.utils as utils
from torchvision.datasets import MNIST, CIFAR10, CIFAR100


def load_adult():
    x, y = np.load('adult/x.npy'), np.load('adult/y.npy')
    y[y == 0] = -1
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_mnist():
    a = MNIST(root='MNIST', train=True)
    ax, ay = a.train_data, a.train_labels
    b = MNIST(root='MNIST', train=False)
    bx, by = b.test_data, b.test_labels
    x, y = th.cat([ax, bx]), th.cat([ay, by])
    x = x.reshape([len(x), -1]).float()
    return x, y


def load_binary_mnist():
    x, y = load_mnist()
    y[y != 1] = -1
    return x, y


def load_cifar10():
    a = CIFAR10(root='CIFAR10', train=True)
    ax, ay = a.train_data, a.train_labels
    b = CIFAR10(root='CIFAR10', train=False)
    bx, by = b.test_data, b.test_labels
    x, y = np.concatenate([ax, bx]), np.concatenate([ay, by])
    x = x.transpose([0, 3, 1, 2]).reshape([len(x), -1])
    x, y = th.from_numpy(x).float(), th.from_numpy(y)
    return x, y
    

def load_binary_cifar10():
    x, y = load_cifar10()
    y[y != 1] = -1
    return x, y


def load_cifar100():
    a = CIFAR100(root='CIFAR100', train=True)
    ax, ay = a.train_data, a.train_labels
    b = CIFAR100(root='CIFAR100', train=False)
    bx, by = b.test_data, b.test_labels
    x, y = np.concatenate([ax, bx]), np.concatenate([ay, by])
    x = x.transpose([0, 3, 1, 2]).reshape([len(x), -1])
    y[y != 1] = -1
    x, y = th.from_numpy(x).float(), th.from_numpy(y)
    return x, y
    

def load_binary_cifar100():
    x, y = load_cifar100()
    y[y != 1] = -1
    return x, y


def load_covtype():
    x, y = np.load('covtype/x.npy'), np.load('covtype/y.npy')
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_binary_covtype():
    x, y = load_covtype()
    y[y != 4] = -1
    y[y == 4] = 1
    return x, y


def load_kddcup08():
    x, y = np.load('kddcup08/x.npy'), np.load('kddcup08/y.npy')
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_letter():
    x, y = np.load('letter/x.npy'), np.load('letter/y.npy')
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_binary_letter():
    x, y = load_letter()
    y[y != 13] = -1
    y[y == 13] = 1
    return x, y


def normalize(xx, eps=1e-5):
    """
    Parameters
    ----------
    xx : list of th.Tensor
    """
    x = xx[0]
    mean = th.mean(x, 0, keepdim=True)
    x = x - mean
    std = th.sqrt(th.mean(x * x, 0, keepdim=True)) + eps
    x = x / std
    xx = [x] + [(x - mean) / std for x in xx[1:]]
    return xx


def partition(x, y, pp):
    """
    Parameters
    ----------
    pp :
    """
    sum_pp = sum(pp)
    pp = [p / sum_pp for p in pp]
    mskk = [(y == i) for i in th.sort(th.unique(y))[0]]
    xx = list(map(x.__getitem__, mskk))
    yy = list(map(y.__getitem__, mskk))
    nnn = [[int(p * len(x)) for p in pp[:-1]] for x in xx]
    nnn = [nn + [len(x) - sum(nn)] for nn, x in zip(nnn, xx)]
    xxx = [th.split(x, nn) for x, nn in zip(xx, nnn)]
    yyy = [th.split(y, nn) for y, nn in zip(yy, nnn)]
    return zip(zip(*xxx), zip(*yyy))


def shuffle(x, y):
    idx = th.randperm(len(y))
    return x[idx], y[idx]
