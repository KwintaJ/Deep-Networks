# resnet for cifar10 (sec 4.2)
# y = F(x, {Wi}) + x, F = conv-bn-relu-conv-bn, relu after the add (eq 1)
# 3x3 conv, then 3 stages of 2n layers on 32x32, 16x16, 8x8 maps with 16, 32, 64 filters,
# global avg pool, fc 10. depth = 6n+2
# plain=True drops the shortcuts, same net otherwise (the plain baseline of fig 6)

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, plain=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.plain = plain
        self.pad = cout - cin

    def shortcut(self, x):
        # option a: identity, when dims grow subsample by 2 and zero pad the new channels
        # so the shortcuts add no params
        if self.pad:
            x = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, self.pad))
        return x

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.plain:
            out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, depth=20, plain=False, classes=10):
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError(f"depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        blocks, cin = [], 16
        for i, cout in enumerate((16, 32, 64)):
            for j in range(n):
                # first block of stage 2 and 3 halves the map
                stride = 2 if i > 0 and j == 0 else 1
                blocks.append(Block(cin, cout, stride, plain))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return self.fc(x.mean((2, 3)))


if __name__ == "__main__":
    for d in (20, 32, 44, 56, 110):
        net = ResNet(d)
        print(d, net(torch.randn(2, 3, 32, 32)).shape, sum(p.numel() for p in net.parameters()), "params")
