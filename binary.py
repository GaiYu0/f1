import argparse
import tensorboardX as tb
import torch as th
import torch.nn.functional as F
import torch.utils.data as D
import data
import mlp
import resnet
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bsi', type=int, help='Batch Size for Inference')
parser.add_argument('--bs-pos', type=int, help='Batch Size for POSitive class')
parser.add_argument('--bs-neg', type=int, help='Batch Size for NEGative class')
parser.add_argument('--ds', type=str, help='DataSet')
parser.add_argument('--gpu', type=int, help='GPU')
parser.add_argument('--id', type=str, help='IDentifier')
parser.add_argument('--log-every', type=int, help='LOG statistics EVERY _ iterations')
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

dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
metric = getattr(utils, args.metric)

x, y = {'adult'    : data.load_adult,
        'cifar10'  : data.load_binary_cifar10,
        'cifar100' : data.load_binary_cifar100,
        'covtype'  : data.load_binary_covtype,
        'kddcup08' : data.load_kddcup08,
        'letter'   : data.load_binary_letter,
        'mnist'    : data.load_binary_mnist}[args.ds]()
x, y = data.shuffle(x, y)
[[[ax_pos, ax_neg], [ay_pos, ay_neg]],
 [[bx_pos, bx_neg], [by_pos, by_neg]],
 [[cx_pos, cx_neg], [cy_pos, cy_neg]]] = data.partition(x, y, args.ptt)
ax, bx, cx = th.cat([ax_pos, ax_neg]), th.cat([bx_pos, bx_neg]), th.cat([cx_pos, cx_neg])
ax, bx, cx = data.normalize([ax, bx, cx])
ay, by, cy = th.cat([ay_pos, ay_neg]), th.cat([by_pos, by_neg]), th.cat([cy_pos, cy_neg])

ax_pos, ax_neg = ax[:len(ax_pos)], ax[len(ax_pos):]
pos, neg = D.TensorDataset(ax_pos), D.TensorDataset(ax_neg)
pos_loader = utils.cycle(D.DataLoader(pos, args.bs_pos, shuffle=True))
neg_loader = utils.cycle(D.DataLoader(neg, args.bs_neg, shuffle=True))

a_loader = D.DataLoader(D.TensorDataset(ax, ay), args.bsi)
b_loader = D.DataLoader(D.TensorDataset(bx, by), args.bsi)
c_loader = D.DataLoader(D.TensorDataset(cx, cy), args.bsi)

p0 = len(ax_neg) / len(ax)
p1 = 1 - p0

if args.model == 'linear':
    model = th.nn.Linear(ax_pos.size(1), 2)
elif args.model == 'mlp':
    model = mlp.MLP([ax_pos.size(1), 64, 64, 64, 2], th.relu, bn=True)
elif args.model == 'resnet':
    model = resnet.ResNet(18, 2)
else:
    raise RuntimeError()
model = model.to(dev)

kwargs = {'weight_decay' : args.wd}
if args.opt == 'sgd':
    opt = th.optim.SGD(model.parameters(), args.lr, **kwargs)
elif args.opt == 'adam':
    opt = th.optim.Adam(model.parameters(), args.lr, amsgrad=True, **kwargs)
else:
    raise RuntimeError()

if args.tb:
    path = 'tb/%s' % args.id
    writer = tb.SummaryWriter(path)
    a_writer = tb.SummaryWriter(path + '/a')
    b_writer = tb.SummaryWriter(path + '/b')
    c_writer = tb.SummaryWriter(path + '/c')

def infer(loader, model):
    yy = []
    y_barr = []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        y_bar = 2 * th.max(model(x), 1)[1] - 1
        yy.append(y)
        y_barr.append(y_bar)
    y = th.cat(yy)
    y_bar = th.cat(y_barr)
    return y, y_bar

def log(model, i):
    mmm = []
    for loader in a_loader, b_loader, c_loader:
        y, y_bar = infer(loader, model)

        tp = utils.tp(y, y_bar) / len(y)
        fp = utils.fp(y, y_bar) / len(y)
        fn = utils.fn(y, y_bar) / len(y)
        tn = utils.tn(y, y_bar) / len(y)

        a = tp + tn
        p = utils.div(tp, tp + fp)
        r = utils.div(tp, p1)
        m = metric(p1, fn, fp)
        mmm.append([tp, fp, fn, tn, a, p, r, m])

    tagg = ['tp', 'fp', 'fn', 'tn', 'a', 'p', 'r', args.metric]

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
    [x_pos], [x_neg] = next(pos_loader), next(neg_loader)
    x_pos, x_neg = x_pos.to(dev), x_neg.to(dev)

    utils.train(model)

    x = th.cat([x_pos, x_neg])
    z = F.softmax(model(x), 1)
    z_pos, z_neg = th.split(z, [x_pos.size(0), x_neg.size(0)])
    fn = p1 * th.mean(z_pos[:, 0])
    fp = p0 * th.mean(z_neg[:, 1])

    if args.w > 0:
        loss = args.w * fn + (1 - args.w) * fp
    else:
        loss = -metric(p1, fn, fp)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if args.tb:
        writer.add_scalar('fnx', fn, i + 1)
        writer.add_scalar('fpx', fp, i + 1)
        writer.add_scalar('loss', loss, i + 1)

        for k, v in model.state_dict().items():
            if k.endswith('weight'):
                writer.add_scalar(k, th.sqrt(th.sum(v * v)), i + 1)

    utils.eval(model)

    if (i + 1) % args.log_every == 0:
        log(model, i + 1)
