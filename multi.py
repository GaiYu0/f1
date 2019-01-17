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
parser.add_argument('--metric', type=str, help='METRIC')
parser.add_argument('--model', type=str, help='MODEL')
parser.add_argument('--ni', type=int, help='Number of Iterations')
parser.add_argument('--opt', type=str, help='OPTimizer')
parser.add_argument('--ptt', nargs='+', type=int, help='ParTiTion')
parser.add_argument('--tb', action='store_true', help='TensorBoard')
parser.add_argument('--w', type=float, help='Weight')
parser.add_argument('--wd', type=float, help='Weight Decay')
args = parser.parse_args()

x, y = {'adult'    : data.load_adult,
        'cifar10'  : data.load_cifar10,
        'cifar100' : data.load_cifar100,
        'covtype'  : data.load_covtype,
        'kddcup08' : data.load_kddcup08,
        'letter'   : data.load_letter,
        'mnist'    : data.load_mnist}[args.ds]()
x, y = data.shuffle(x, y)
[[train_xx, train_yy],
 [val_xx,   val_yy],
 [test_xx,  test_yy]] = data.partition(x, y, args.ptt)
train_x, val_x, test_x = th.cat(train_xx), th.cat(val_xx), th.cat(test_xx)
train_y, val_y, test_y = th.cat(train_yy), th.cat(val_yy), th.cat(test_yy)
train_x, val_x, test_x = data.normalize([train_x, val_x, test_x])
train_xx = th.split(train_x, [len(x) for x in train_xx])
train_datasets = [D.TensorDataset(x) for x in train_xx]
train_loader = D.DataLoader(D.TensorDataset(train_x, train_y), args.bsi)
val_loader = D.DataLoader(D.TensorDataset(val_x, val_y), args.bsi)
test_loader = D.DataLoader(D.TensorDataset(test_x, test_y), args.bsi)
pclass_list = [len(y) / len(train_y) for y in train_yy]

n_classes = len(train_yy)
if len(args.bst) == n_classes:
    bs_list = args.bst
elif len(args.bst) == 1:
    bs_list = [args.bst[0]] * n_classes
else:
    raise RuntimeError()
train_loaders = [utils.cycle(D.DataLoader(ds, bs, shuffle=True)) \
                 for ds, bs in zip(train_datasets, bs_list)]

if args.model == 'linear':
    model = th.nn.Linear(train_x.size(1), n_classes)
elif args.model == 'mlp':
    model = mlp.MLP([train_x.size(1), 64, 64, 64, n_classes], th.relu, bn=True)
elif args.model == 'resnet':
    model = resnet.ResNet(18, n_classes)[args.model]
else:
    raise RuntimeError()
dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
model = model.to(dev)
params = list(model.parameters())
kwargs = {'params' : params, 'lr' : args.lr, 'weight_decay' : args.wd}
opt = {'sgd'  : optim.SGD(**kwargs),
       'adam' : optim.Adam(amsgrad=True, **kwargs)}[args.opt]
metric = getattr(utils, args.metric)

if args.tb:
    raise NotImplementedError()
    path = 'tb/%s' % args.id
    writer = tb.SummaryWriter(path)
    train_writer = tb.SummaryWriter(path + '/train')
    val_writer = tb.SummaryWriter(path + '/val')
    test_writer = tb.SummaryWriter(path + '/test')

def log(model, i):
    mmm = []
    for loader in train_loader, val_loader, test_loader:
        y, y_bar = infer(loader, model)

        a = th.sum(y == y_bar).item() / len(y)
        fnfn = utils.fn_mc(y, y_bar, n_classes)
        fpfp = utils.fp_mc(y, y_bar, n_classes)
        m = metric(pclass_list, fnfn, fpfp)

        mmm.append([a, m])

    tagg = ['a', args.metric]

    placeholder = '0' * (len(str(args.ni)) - len(str(i)))
    xx = ['/'.join(['%0.2f' % m for m in mm]) for mm in zip(*mmm)]
    x = ' | '.join('%s %s' % (tag, mm) for tag, mm in zip(tagg, xx))
    print('[iteration %s%d]%s' % ((placeholder, i, x)))

    if args.tb:
        for writer, mm in zip([a_writer, b_writer, c_writer], mmm):
            for tag, m in zip(tagg, mm):
                writer.add_scalar(tag, m, i)

utils.eval(model)
log(model, 0)

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
        loss = -metric(pclass_list, fnfn, fpfp)

    opt.zero_grad()
    loss.backward()
    opt.step()

    utils.eval(model)
    if (i + 1) % args.log_every == 0:
        log(model, i + 1)
