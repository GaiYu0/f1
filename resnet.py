# https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv2d(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class Basic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Basic, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        expansion = self.expansion * planes
        if stride != 1 or in_planes != expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion)
            )

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        z = F.relu(z + self.shortcut(x))
        return z

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        expansion = self.expansion * planes
        if stride != 1 or in_planes != expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion)
            )

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = self.bn3(self.conv3(z))
        z = F.relu(z + self.shortcut(x))
        return z

class ResNet(nn.Module):
    def __init__(self, depth, n_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, n_blocks = {18  : [Basic, [2, 2, 2, 2]],
                           34  : [Basic, [3, 4, 6, 3]],
                           50  : [Bottleneck, [3, 4, 6, 3]],
                           101 : [Bottleneck, [3, 4, 23, 3]],
                           152 : [Bottleneck, [3, 8, 36, 3]]}[depth]

        self.conv1 = Conv2d(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.mklayer(block, 16, n_blocks[0], stride=1)
        self.layer2 = self.mklayer(block, 32, n_blocks[1], stride=2)
        self.layer3 = self.mklayer(block, 64, n_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, n_classes)

    def mklayer(self, block, planes, n_blocks, stride):
        stridee = [stride] + [1] * (n_blocks - 1)
        layerr = []

        for stride in stridee:
            layerr.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layerr)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        z = F.relu(self.bn(self.conv1(x)))
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = F.avg_pool2d(z, 8).view(z.size(0), -1)
        return self.linear(z)
