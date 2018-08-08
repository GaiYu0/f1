import argparse as ap
import itertools as it
import tensorboardX as tb
import torch as th
import torch.utils.data as thdata
import data
import mlp
import resnet
import utils

parser = ap.ArgumentParser()
parser.add_argument('--bs', type=int, help='Batch Size')
parser.add_argument('--bsi', type=int, help='Batch Size for Inference')
parser.add_argument('--ds', type=str, help='DataSet')
parser.add_argument('--gpu', type=int, help='GPU')
parser.add_argument('--log-every', type=int, help='LOG statistics EVERY _ iterations')
parser.add_argument('--lr', type=float, help='Learning Rate')
parser.add_argument('--model', type=str, help='MODEL')
parser.add_argument('--ni', type=int, help='Number of Iterations')
parser.add_argument('--ptt', nargs='+', type=int, help='ParTiTion')
parser.add_argument('--srgt', type=str, help='SurRoGaTe')
parser.add_argument('--tb', action='store_true', help='TensorBoard')
parser.add_argument('--update-every', type=int, help='UPDATE weight EVERY _ iterations')
parser.add_argument('--w-pos', type=str, help='Weight for POSitive class')
parser.add_argument('--w-neg', type=str, help='Weight for NEGative class')
args = parser.parse_args()

x, y = {'cifar10'   : data.load_cifar10,
        'covertype' : data.load_covertype}[args.ds]()
# x, y = data.shuffle(x, y)
[[[ax_pos, ax_neg], [ay_pos, ay_neg]],
 [[bx_pos, bx_neg], [by_pos, by_neg]],
 [[cx_pos, cx_neg], [cy_pos, cy_neg]]] = data.partition(x, y, args.ptt)
ax, bx, cx = th.cat([ax_pos, ax_neg]), th.cat([bx_pos, bx_neg]), th.cat([cx_pos, cx_neg])
ax, bx, cx = data.normalize([ax, bx, cx])

ax_pos, ax_neg = ax[:len(ax_pos)], ax[len(ax_pos):]
pos, neg = thdata.TensorDataset(ax_pos), thdata.TensorDataset(ax_neg)
pos_loader = it.cycle(thdata.DataLoader(pos, args.bs, shuffle=True, drop_last=True))
neg_loader = it.cycle(thdata.DataLoader(neg, args.bs, shuffle=True, drop_last=True))

a_loader = thdata.DataLoader(thdata.TensorDataset(ax, th.cat([ay_pos, ay_neg])), args.bsi)
b_loader = thdata.DataLoader(thdata.TensorDataset(bx, th.cat([by_pos, by_neg])), args.bsi)
c_loader = thdata.DataLoader(thdata.TensorDataset(cx, th.cat([cy_pos, cy_neg])), args.bsi)

model = {'linear' : th.nn.Linear(ax_pos.size(1), 1),
         'mlp'    : mlp.MLP([ax_pos.size(1), 64, 64, 64, 1], th.tanh),
         'resnet' : resnet.ResNet(18, 1)}[args.model]
model.apply(utils.init)
opt = th.optim.Adam(model.parameters(), args.lr, amsgrad=True)

if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)
    model.cuda()

def infer(loader, model):
    yy = []
    y_barr = []
    for x, y in loader:
        if cuda:
            x, y = x.cuda(), y.cuda()
        yy.append(y)
        y_barr.append(th.sign(model(x)).long())
    y = th.cat(yy)
    y_bar = th.cat(y_barr)
    y_bar = y_bar.squeeze(1)
    return y, y_bar

loaderr = [a_loader, b_loader, c_loader]
tagg = ['tp', 'fp', 'fn', 'tn', 'a', 'p', 'r', 'f1']

if args.tb:
    keys = sorted(vars(args).keys())
    excluded = ['bsi', 'gpu', 'log_every', 'ni', 'tb', 'update_every']
    path = 'runs/' + '#'.join('%s:%s' % (k, str(getattr(args, k))) \
                              for k in keys if k not in excluded)
    writer = tb.SummaryWriter(path)
    a_writer = tb.SummaryWriter(path + '/a')
    b_writer = tb.SummaryWriter(path + '/b')
    c_writer = tb.SummaryWriter(path + '/c')
    writerr = [a_writer, b_writer, c_writer]

def log(model, i):
    mmm = []
    for loader in loaderr:
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

    if args.tb:
        for writer, mm in zip(writerr, mmm):
            for tag, m in zip(tagg, mm):
                writer.add_scalar(tag, m, i)

    placeholder = '0' * (len(str(args.ni)) - len(str(i)))
    xx = ['/'.join(['%0.2f' % m for m in mm]) for mm in zip(*mmm)]
    x = ' | '.join('%s %s' % (tag, mm) for tag, mm in zip(tagg, xx))
    print('[iteration %s%d]%s' % ((placeholder, i, x)))

log(model, 0)

if not args.update_every:
#   w_pos, w_neg = len(ax_pos) / len(ax_neg), 1
    w_pos, w_neg = eval(args.w_pos), eval(args.w_neg)

for i in range(args.ni):
    if args.update_every > 0 and i % args.update_every == 0:
        for p in model.parameters():
            p.requires_grad = False
        y, y_bar = infer(b_loader, model)
        p1 = th.sum(y > 0).float()
        fn = utils.fn(y, y_bar)
        fp = utils.fp(y, y_bar)

        '''
        w_pos, w_neg = fn, fp
        w_pos, w_neg = 1, (p1 - fn) / (p1 + fp)
        '''

        w_pos, w_neg = eval(args.w_pos), eval(args.w_neg)

    [x_pos], [x_neg] = next(pos_loader), next(neg_loader)

    if cuda:
        x_pos, x_neg = x_pos.cuda(), x_neg.cuda()

    for p in model.parameters():
        p.requires_grad = True

    if args.srgt == 'max':
        zeros = th.zeros(args.bs, 1)
        if cuda:
            zeros = zeros.cuda()
        z_pos = th.mean(th.max(zeros, 1 - model(x_pos)))
        z_neg = th.mean(th.max(zeros, 1 + model(x_neg)))
    elif args.srgt == 'exp':
        z_pos = th.mean(th.exp(-model(x_pos)))
        z_neg = th.mean(th.exp(model(x_neg)))
    else:
        raise RuntimeError()

    z = w_pos * z_pos + w_neg * z_neg

    opt.zero_grad()
    z.backward()
    opt.step()

    if args.tb:
        writer.add_scalar('z_pos', z_pos, i + 1)
        writer.add_scalar('z_neg', z_neg, i + 1)
        writer.add_scalar('z', z, i + 1)

    if (i + 1) % args.log_every == 0:
        for p in model.parameters():
            p.requires_grad = False
        log(model, i + 1)
