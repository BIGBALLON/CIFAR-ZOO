# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['sk_resnext29_8x64d', 'sk_resnext29_16x64d']


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=1+i*2,
                          stride=stride, padding=i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat(
                    [attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Bottleneck(nn.Module):
    def __init__(self, in_features, out_features, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features),
            SKConv(out_features, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(out_features),
            nn.Conv2d(out_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return self.relu(fea + self.shortcut(x))


class SKNet(nn.Module):
    def __init__(self, depth, num_classes,  M=2, G=8, r=1):
        super(SKNet, self).__init__()
        self.M = M
        self.G = G
        self.r = r
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.num_classes = num_classes
        self.stages = [64, 64 * 4, 128 * 4, 256 * 4]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block(
            'stage_1', self.stages[0], self.stages[1], self.M, self.G, self.r, stride=1)
        self.stage_2 = self.block(
            'stage_2', self.stages[1], self.stages[2], self.M, self.G, self.r, stride=2)
        self.stage_3 = self.block(
            'stage_3', self.stages[2], self.stages[3], self.M, self.G, self.r, stride=2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, M, G, r, stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, Bottleneck(
                    in_channels, out_channels, M, G, r, stride=stride))
            else:
                block.add_module(name_, Bottleneck(
                    out_channels, out_channels, M, G, r, stride=1))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.fc(x)


def sk_resnext29_8x64d(num_classes):
    return SKNet(depth=29, num_classes=num_classes, M=2, G=8, r=2)


def sk_resnext29_16x64d(num_classes):
    return SKNet(depth=29, num_classes=num_classes, M=2, G=16, r=2)
