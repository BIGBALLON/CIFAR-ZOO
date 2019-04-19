# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['sk_resnext29_16x32d', 'sk_resnext29_16x64d']


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
        d = max(int(features / r), L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=1 + i * 2,
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

    def __init__(
        self, in_channels, out_channels, stride, cardinality,
        base_width, expansion, M, r, L
    ):
        super(Bottleneck, self).__init__()
        width_ratio = out_channels / (expansion * 64.)
        D = cardinality * int(base_width * width_ratio)

        self.relu = nn.ReLU(inplace=True)

        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)

        self.conv_sk = SKConv(D, M, cardinality, r, stride=stride, L=L)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'shortcut_conv',
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride,
                    padding=0, bias=False
                )
            )
            self.shortcut.add_module(
                'shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv_reduce.forward(x)
        out = self.relu(self.bn_reduce.forward(out))
        out = self.conv_sk.forward(out)
        out = self.relu(self.bn.forward(out))
        out = self.conv_expand.forward(out)
        out = self.bn_expand.forward(out)
        residual = self.shortcut.forward(x)
        return self.relu(residual + out)


class SkResNeXt(nn.Module):
    def __init__(
            self, cardinality, depth, num_classes, base_width, expansion=4,
            M=2, r=32, L=32
    ):
        super(SkResNeXt, self).__init__()
        self.M = M
        self.r = r
        self.L = L
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.expansion = expansion
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.expansion, 128 *
                       self.expansion, 256 * self.expansion]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.fc = nn.Linear(self.stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(
                    name_,
                    Bottleneck(
                        in_channels,
                        out_channels,
                        pool_stride,
                        self.cardinality,
                        self.base_width,
                        self.expansion,
                        self.M,
                        self.r,
                        self.L)
                )
            else:
                block.add_module(
                    name_,
                    Bottleneck(
                        out_channels,
                        out_channels,
                        1,
                        self.cardinality,
                        self.base_width,
                        self.expansion,
                        self.M, self.r, self.L
                    )
                )
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


def sk_resnext29_16x32d(num_classes):
    return SkResNeXt(
        cardinality=16,
        depth=29,
        num_classes=num_classes,
        base_width=32
    )


def sk_resnext29_16x64d(num_classes):
    return SkResNeXt(
        cardinality=16,
        depth=29,
        num_classes=num_classes,
        base_width=64
    )
