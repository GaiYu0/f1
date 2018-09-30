import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, feats, nonlinear, bn=False):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(i, j) \
                                          for i, j in zip(feats[:-1], feats[1:])])
        self.nonlinear = nonlinear
        # TODO style
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(i, affine=False) for i in feats[1 : -1]]) if bn else []

    def forward(self, x, training=False):
        if self.bn_list:
            for linear, bn in zip(self.linear_list[:-1], self.bn_list):
                x = bn(self.nonlinear(linear(x)))
        else:
            for linear in self.linear_list[:-1]:
                x = self.nonlinear(linear(x))
        return self.linear_list[-1](x)
