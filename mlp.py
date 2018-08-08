import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, featt, nonlinear):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(i, j) \
                                          for i, j in zip(featt[:-1], featt[1:])])
        self.nonlinear = nonlinear

    def forward(self, x):
        for linear in self.linear_list[:-1]:
            x = self.nonlinear(linear(x))
        return self.linear_list[-1](x)
