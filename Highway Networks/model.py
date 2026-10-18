# highway network
# y = H(x)*T(x) + x*(1-T(x)), T = sigmoid(Wt x + bt), carry gate C = 1-T (eq 3)

import torch
from torch import nn

ACT = {"relu": nn.ReLU, "tanh": nn.Tanh}


def init(linear, act="linear"):
    # he init, gain picked by activation; sigmoid gates use gain 1
    nn.init.kaiming_normal_(linear.weight, nonlinearity=act if act in ("relu", "tanh") else "linear")
    nn.init.zeros_(linear.bias)


class HighwayLayer(nn.Module):
    def __init__(self, dim, act="tanh", gate_bias=-2.0):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.act = ACT[act]()
        init(self.H, act)
        init(self.T)
        # negative bias so the net starts close to identity
        nn.init.constant_(self.T.bias, gate_bias)

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        return self.act(self.H(x)) * t + x * (1 - t)


class HighwayNet(nn.Module):
    # first layer is plain and changes dim (3072 -> width)
    # then depth-1 highway layers, then linear to classes
    # depth = total hidden layers
    def __init__(self, depth=50, width=50, act="tanh", gate_bias=-4.0, in_dim=3 * 32 * 32, classes=10):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.act = ACT[act]()
        init(self.inp, act)
        self.layers = nn.ModuleList(HighwayLayer(width, act, gate_bias) for _ in range(depth - 1))
        self.out = nn.Linear(width, classes)

    def forward(self, x):
        x = self.act(self.inp(x.flatten(1)))
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


if __name__ == "__main__":
    net = HighwayNet(depth=10)
    print(net(torch.randn(2, 3, 32, 32)).shape, sum(p.numel() for p in net.parameters()), "params")
