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


def f1(p1, fn, fp):
    return 2 * (p1 - fn) / (2 * p1 - fn + fp)


def g1(p1, fn, fp):
    ret = (p1 - fn) / (p1 * (p1 - fn + fp)) ** 0.5
    if type(ret) is complex:
        ret = float('Inf')
    return ret


def tp_01(y, y_bar):
    return th.sum((y == 1) * (y_bar == 1)).item() / len(y)


def fp_01(y, y_bar):
    return th.sum((y == 0) * (y_bar == 1)).item() / len(y)


def fn_01(y, y_bar):
    return th.sum((y == 1) * (y_bar == 0)).item() / len(y)


def tn_01(y, y_bar):
    return th.sum((y == 0) * (y_bar == 0)).item() / len(y)


def tp_mc(y, y_bar, n_classes):
    return [tp_01((y == i), (y_bar == i)) for i in range(n_classes)]


def fp_mc(y, y_bar, n_classes):
    return [fp_01((y == i), (y_bar == i)) for i in range(n_classes)]


def fn_mc(y, y_bar, n_classes):
    return [fn_01((y == i), (y_bar == i)) for i in range(n_classes)]


def fp_mc(y, y_bar, n_classes):
    return [fp_01((y == i), (y_bar == i)) for i in range(n_classes)]


def precision_macro():
    pass


def f1_macro(pp, fnfn, fpfp):
    return 2 / len(pp) * sum((p - fn) / (2 * p - fn + fp) \
                             for p, fn, fp in zip(pp, fnfn, fpfp))


def f1_micro(pp, fnfn, fpfp):
    sum_fn = sum(fnfn)
    sum_fp = sum(fpfp)
    return 2 * (1 - sum_fn) / (2 - sum_fn + sum_fp)


def g1_micro(p1, fnfn, fpfp):
    sum_fn = sum(fnfn)
    sum_fp = sum(fpfp)
    ret = (1 - sum_fn) / (1 - sum_fn + sum_fp) ** 0.5
    if type(ret) is complex:
        ret = float('Inf')
    return ret


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
