import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, featt, nonlinear, p=0):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(i, j) \
                                          for i, j in zip(featt[:-1], featt[1:])])
        self.nonlinear = nonlinear
        self.dropout = nn.Dropout(p) if p > 0 else None

    def forward(self, x, training=False):
        for linear in self.linear_list[:-1]:
            x = self.nonlinear(linear(x))
            x = x if self.dropout is None else self.dropout(x)
        return self.linear_list[-1](x)
