# -*-coding:utf-8-*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['densenet100bc', 'densenet190bc']


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn_1 = nn.BatchNorm2d(in_planes)
        self.conv_1 = nn.Conv2d(in_planes, 4 * growth_rate,
                                kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv_2 = nn.Conv2d(4 * growth_rate, growth_rate,
                                kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv_1(F.relu(self.bn_1(x)))
        out = self.conv_2(F.relu(self.bn_2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(
            self, block, depth, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        nblocks = (depth - 4) // 6
        num_planes = 2 * growth_rate
        self.conv_1 = nn.Conv2d(
            3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense_3 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.fc = nn.Linear(num_planes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.trans_1(self.dense1(out))
        out = self.trans_2(self.dense2(out))
        out = self.dense_3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def densenet100bc(num_classes):
    return DenseNet(
        Bottleneck,
        depth=100,
        growth_rate=12,
        num_classes=num_classes
    )


def densenet190bc(num_classes):
    return DenseNet(
        Bottleneck,
        depth=190,
        growth_rate=40,
        num_classes=num_classes
    )
