import numpy as np
import numpy.random as npr
import torch as th
import torch.utils as utils
import torchvision.datasets as datasets


def load_adult():
    x, y = np.load('adult/x.npy'), np.load('adult/y.npy')
    y[y == 0] = -1
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_cifar10():
    a = datasets.CIFAR10('CIFAR10')
    ax, ay = a.train_data, a.train_labels
    b = datasets.CIFAR10('CIFAR10', train=False)
    bx, by = b.test_data, b.test_labels
    x, y = np.concatenate([ax, bx]), np.concatenate([ay, by])
    x = x.transpose([0, 3, 1, 2]).reshape([len(x), -1])
    y[y != 1] = -1
    x, y = th.from_numpy(x).float(), th.from_numpy(y)
    return x, y
    

def load_covtype():
    x, y = np.load('covtype/x.npy'), np.load('covtype/y.npy')
    y[y != 5] = -1
    y[y == 5] = 1
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_kddcup08():
    x, y = np.load('kddcup08/x.npy'), np.load('kddcup08/y.npy')
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def load_letter():
    x, y = np.load('letter/x.npy'), np.load('letter/y.npy')
    y[y != 13] = -1
    y[y == 13] = 1
    x, y = th.from_numpy(x), th.from_numpy(y)
    return x, y


def normalize(xx, eps=1e-5):
    """
    Parameters
    ----------
    xx : list of th.Tensor
    """
    mean = th.mean(xx[0], 0, keepdim=True)
    xx[0] = xx[0] - mean
    std = th.sqrt(th.mean(xx[0] * xx[0], 0, keepdim=True)) + eps
    xx[0] = xx[0] / std
    xx = [xx[0]] + [(x - mean) / std for x in xx[1:]]
    return xx


def partition(x, y, pp):
    """
    Parameters
    ----------
    pp :
    """
    sum_pp = sum(pp)
    pp = [p / sum_pp for p in pp]
    mskk = [(y == i) for i in th.unique(y)]
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
