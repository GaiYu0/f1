import torch as th
import sklearn.metrics as metrics


def cycle(loader):
    while True:
        for x in loader:
            yield x


def div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return float('Inf')


"""
Parameters
----------
y : (N,)
y_bar : (N,)
"""


def tp(y, y_bar):
    return th.sum((y > 0) * (y_bar >= 0)).item()


def fp(y, y_bar):
    return th.sum((y < 0) * (y_bar >= 0)).item()


def fn(y, y_bar):
    return th.sum((y > 0) * (y_bar < 0)).item()


def tn(y, y_bar):
    return th.sum((y < 0) * (y_bar < 0)).item()


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fn + fp)


def train(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def eval(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


if __name__ == '__main__':
    N = 10000
    y = 2 * th.randint(0, 2, [N]) - 1
    y_bar = 2 * th.randint(0, 2, [N]) - 1

    tp_x = tp(y, y_bar)
    fp_x = fp(y, y_bar)
    fn_x = fn(y, y_bar)
    tn_x = tn(y, y_bar)
    f1_x = f1(tp_x, fp_x, fn_x)

    y, y_bar = y.numpy(), y_bar.numpy()
    f1_z = metrics.f1_score(y, y_bar)
    c = metrics.confusion_matrix(y, y_bar)

    assert f1_x == f1_z
    assert [tp_x, fp_x, fn_x, tn_x] == [c[1, 1], c[0, 1], c[1, 0], c[0, 0]]
