import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class F1(nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
        self.fn = nn.Parameter(th.zeros(1))
        self.fp = nn.Parameter(th.zeros(1))

    def forward(self):
        fn = self.p1 * F.sigmoid(self.fn)
        fp = (1 - self.p1) * F.sigmoid(self.fp)
        f1 = 2 * (self.p1 - fn) / (2 * self.p1 - fn + fp)
        return f1

f1 = F1(0.0163)
opt = optim.Adam(f1.parameters(), lr=0.1)
for i in range(10):
    opt.zero_grad()
    (-f1()).backward()
    opt.step()

    print(f1().item())
