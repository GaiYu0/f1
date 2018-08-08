import torch as th
import torch.nn as nn
import sklearn.metrics as metrics


def init(x):
    if isinstance(x, (nn.Linear, nn.Conv2d)):
        x.bias.data.fill_(0)


def div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return float('Inf')

def tp(y, y_bar):
    """
    Parameters
    ----------
    y : (N,)
    y_bar : (N,)
    """
    return th.sum((y > 0) * (y_bar > 0)).item()


def fp(y, y_bar):
    """
    Parameters
    ----------
    y : (N,)
    y_bar : (N,)
    """
    return th.sum((y < 0) * (y_bar > 0)).item()


def fn(y, y_bar):
    """
    Parameters
    ----------
    y : (N,)
    y_bar : (N,)
    """
    return th.sum((y > 0) * (y_bar < 0)).item()


def tn(y, y_bar):
    """
    Parameters
    ----------
    y : (N,)
    y_bar : (N,)
    """
    return th.sum((y < 0) * (y_bar < 0)).item()


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fn + fp)


if __name__ == '__main__':
    N = 10000
    y = th.randint(0, 2, [N])
    y[y == 0] = -1
    y_bar = th.randint(0, 2, [N])
    y_bar[y_bar == 0] = -1

    tp_x = tp(y, y_bar)
    fp_x = fp(y, y_bar)
    fn_x = fn(y, y_bar)
    tn_x = tn(y, y_bar)
    f1_x = f1(tp_x, fp_x, fn_x)
    print(tp_x, fp_x, fn_x, tn_x, f1_x)

    y, y_bar = y.numpy(), y_bar.numpy()
    print(metrics.confusion_matrix(y, y_bar))
    print(metrics.f1_score(y, y_bar))
