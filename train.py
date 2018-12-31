import argparse
import tensorboardX as tb
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import data
import mlp
import resnet
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bst', nargs='+', type=int, help='Batch Size for Training')
parser.add_argument('--bsi', type=int, help='Batch Size for Inference')
parser.add_argument('--ds', type=str, help='DataSet')
parser.add_argument('--gpu', type=int, help='GPU')
parser.add_argument('--id', type=str, help='IDentifier')
parser.add_argument('--log-every', type=int, help='LOG statistics EVERY _ iterations')
parser.add_argument('--loss', type=str, help='LOSS')
parser.add_argument('--lr', type=float, help='Learning Rate')
parser.add_argument('--model', type=str, help='MODEL')
parser.add_argument('--ni', type=int, help='Number of Iterations')
parser.add_argument('--opt', type=str, help='OPTimizer')
parser.add_argument('--ptt', nargs='+', type=int, help='ParTiTion')
parser.add_argument('--tb', action='store_true', help='TensorBoard')
parser.add_argument('--w', type=float, help='Weight')
parser.add_argument('--wd', type=float, help='Weight Decay')
args = parser.parse_args()

dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)

x, y = {'adult'    : data.load_adult,
        'cifar10'  : data.load_cifar10,
        'cifar100' : data.load_cifar100,
        'covtype'  : data.load_covtype,
        'kddcup08' : data.load_kddcup08,
        'letter'   : data.load_letter}[args.ds]()
x, y = data.shuffle(x, y)
[[train_xx, train_yy],
 [val_xx,   val_yy],
 [test_xx,  test_yy]] = data.partition(x, y, args.ptt)
train_x, val_x, test_x = th.cat(train_xx), th.cat(val_xx), th.cat(test_xx)
train_y, val_y, test_y = th.cat(train_yy), th.cat(val_yy), th.cat(test_yy)
train_x, val_x, test_x = data.normalize([train_x, val_x, test_x])
train_xx = th.split(train_x, [len(x) for x in train_xx])
train_datasets = [D.TensorDataset(x) for x in train_xx]
train_loaders = [utils.cycle(D.DataLoader(ds, bs, shuffle=True)) \
                 for ds, bs in zip(train_datasets, args.bst)]

train_loader = D.DataLoader(D.TensorDataset(train_x, train_y), args.bsi)
val_loader = D.DataLoader(D.TensorDataset(val_x, val_y), args.bsi)
test_loader = D.DataLoader(D.TensorDataset(test_x, test_y), args.bsi)

n_classes = len(train_yy)
pclass_list = [len(y) / len(train_y) for y in train_yy]  # estimate on training set

model = {'linear' : th.nn.Linear(train_x.size(1), n_classes),
         'mlp'    : mlp.MLP([train_x.size(1), 64, 64, 64, n_classes], th.relu, bn=True),
         'resnet' : resnet.ResNet(18, n_classes)}[args.model].to(dev)

params = list(model.parameters())
kwargs = {'params' : params, 'lr' : args.lr, 'weight_decay' : args.wd}
opt = {'sgd'  : optim.SGD(**kwargs),
       'adam' : optim.Adam(amsgrad=True, **kwargs)}[args.opt]

if args.tb:
    path = 'tb/%s' % args.id
    writer = tb.SummaryWriter(path)
    train_writer = tb.SummaryWriter(path + '/train')
    val_writer = tb.SummaryWriter(path + '/val')
    test_writer = tb.SummaryWriter(path + '/test')

def log(model, i):
    # TODO
    mmm = []
    for loader in train_loader, val_loader, test_loader:
        y, y_bar = infer(loader, model)

        tp = utils.tp(y, y_bar)
        tpr = tp / len(y)
        fpr = utils.fp(y, y_bar) / len(y)
        fnr = utils.fn(y, y_bar) / len(y)
        tnr = utils.tn(y, y_bar) / len(y)

        a = th.sum(y == y_bar).item() / len(y)
        p = utils.div(tp, th.sum(y_bar > 0).item())
        r = utils.div(tp, th.sum(y > 0).item())
        f1 = utils.f1(tpr, fpr, fnr)

        mmm.append([tpr, fpr, fnr, tnr, a, p, r, f1])

    tagg = ['tp', 'fp', 'fn', 'tn', 'a', 'p', 'r', 'f1']

    placeholder = '0' * (len(str(args.ni)) - len(str(i)))
    xx = ['/'.join(['%0.2f' % m for m in mm]) for mm in zip(*mmm)]
    x = ' | '.join('%s %s' % (tag, mm) for tag, mm in zip(tagg, xx))
    print('[iteration %s%d]%s' % ((placeholder, i, x)))

    if args.tb:
        for writer, mm in zip([a_writer, b_writer, c_writer], mmm):
            for tag, m in zip(tagg, mm):
                writer.add_scalar(tag, m, i)

'''
utils.eval(model)
log(model, 0)
'''

for i in range(args.ni):
    xx = [next(loader)[0].to(dev) for loader in train_loaders]
    x = th.cat(xx)
    z = F.softmax(model(x), 1)
    zz = th.split(z, [len(x) for x in xx])
    pneg_list = [1 - th.mean(z[:, i]) for i, z in enumerate(zz)]
    fnfn = [p_class * p_neg for p_class, p_neg in zip(pclass_list, pneg_list)]
    fpfp = [(1 - p_class) * p_neg for p_class, p_neg in zip(pclass_list, pneg_list)]

    if args.w > 0:
        pass
    else:
        loss = -getattr(utils, args.loss)(pclass_list, fnfn, fpfp)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(-loss.item())

    '''
    utils.eval(model)
    if (i + 1) % args.log_every == 0:
        log(model, i + 1)
    '''
