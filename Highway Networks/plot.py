# pull run history from wandb and draw it like the paper
# usage: python plot.py highway_d50_w50_b-4_tanh            -> figures/<name>.png (train ce log y + accs)
#        python plot.py compare fig1 name1 name2 ...         -> figures/fig1.png, all runs on one plot
#        python plot.py gates checkpoints/<name>.pt          -> figures/gates_<name>.png, fig 2 style heatmaps
# picks the longest run with a given name if there are several

import math
import sys
from pathlib import Path

import matplotlib
import torch
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import HighwayNet
from train import CONFIG, get_data

FIG = Path("figures")


def fetch(api, name):
    runs = list(api.runs(f"{CONFIG['entity']}/{CONFIG['project']}", {"display_name": name}))
    if not runs:
        sys.exit(f"no run named {name}")
    run = max(runs, key=lambda r: r.summary.get("epoch", 0))
    hist = run.history(samples=100000, pandas=False)
    hist = sorted((h for h in hist if h.get("train/cross_entropy") is not None), key=lambda h: h["epoch"])
    return run, hist


LABEL = dict(depth="Depth", width="Width", gate_bias="Gate Bias", act="", lr="LR", lr_decay="LR Decay", grad_clip="Grad Clip")


def fmt(key, value):
    if value is None:
        return f"No {LABEL[key]}"
    if isinstance(value, float):
        value = f"{value:g}"
    return f"{LABEL[key]} {value}".strip()


def header(cfg):
    keys = ["depth", "width", "gate_bias", "act", "lr", "lr_decay"]
    if cfg.get("grad_clip") is not None:
        keys.append("grad_clip")
    return ", ".join(fmt(k, cfg.get(k)) for k in keys)


def log_decades(ax):
    # snap log axis to whole decades, no labels on minor ticks
    lo, hi = ax.dataLim.intervaly
    ax.set_ylim(10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def single(api, name):
    run, hist = fetch(api, name)
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.semilogy(ep, [h["train/cross_entropy"] for h in hist], color="green", lw=1.2)
    log_decades(a)
    a.set_xlabel("Epoch")
    a.set_ylabel("Mean Cross Entropy")
    a.set_title("Training Cross Entropy")
    b.plot(ep, [h["train/acc"] for h in hist], label="Training")
    b.plot(ep, [h["val/acc"] for h in hist], label="Validation")
    b.set_xlabel("Epoch")
    b.set_ylabel("Accuracy")
    b.set_ylim(0, 1)
    b.set_title(f"Accuracy (Test: {run.summary.get('test/acc', float('nan')):.3f})")
    b.legend()
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.suptitle(f"{header(run.config)}, {run.summary.get('params', '?')} Parameters", fontsize=9)
    save(fig, FIG / f"{name}.png", f"{len(hist)} epochs, url {run.url}")


def compare(api, out, names):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    runs = [fetch(api, n) for n in names]
    # legend shows only the config keys that differ between the runs
    keys = ["depth", "width", "gate_bias", "act", "lr", "lr_decay", "grad_clip"]
    keys = [k for k in keys if len({str(r.config.get(k)) for r, _ in runs}) > 1] or ["depth"]
    for run, hist in runs:
        ep = [h["epoch"] for h in hist]
        label = ", ".join(fmt(k, run.config.get(k)) for k in keys)
        a.semilogy(ep, [h["train/cross_entropy"] for h in hist], lw=1.2, label=label)
        b.plot(ep, [h["val/acc"] for h in hist], lw=1.2, label=label)
    log_decades(a)
    a.set_xlabel("Epoch")
    a.set_ylabel("Mean Cross Entropy")
    a.set_title("Training Cross Entropy")
    b.set_xlabel("Epoch")
    b.set_ylabel("Accuracy")
    b.set_title("Validation Accuracy")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    # one legend under both panels so it never covers the curves
    handles, labels = a.get_legend_handles_labels()
    ncol = 4 if len(names) <= 4 else 3
    rows = (len(names) + ncol - 1) // ncol
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8, frameon=False)
    save(fig, FIG / f"{out}.png", f"{len(names)} runs", rect=(0, 0.06 + 0.05 * rows, 1, 1))


@torch.no_grad()
def gates(ckpt_path, samples=10000):
    # fig 2 of the paper: gate biases, mean gate output over 10k train samples,
    # gate output and block output for one random sample. rows = layers, cols = blocks
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    net = HighwayNet(cfg["depth"], cfg["width"], cfg["act"], cfg["gate_bias"])
    net.load_state_dict(state["net"])
    net.eval()
    loader = get_data(cfg)["train_eval"]
    layers = net.layers
    mean_t = torch.zeros(len(layers), cfg["width"])
    seen = 0
    for x, _ in loader:
        x = x[: samples - seen]
        h = net.act(net.inp(x.flatten(1)))
        for i, layer in enumerate(layers):
            t = torch.sigmoid(layer.T(h))
            mean_t[i] += t.sum(0)
            h = layer(h)
        if seen == 0:
            # first sample of the first batch is the "one random sample"
            x0 = x[:1]
        seen += len(x)
        if seen >= samples:
            break
    mean_t /= seen
    h = net.act(net.inp(x0.flatten(1)))
    one_t, one_y = [], []
    for layer in layers:
        one_t.append(torch.sigmoid(layer.T(h))[0])
        h = layer(h)
        one_y.append(h[0])
    panels = [
        ("Transform Gate Biases", torch.stack([l.T.bias for l in layers]), None, None),
        ("Mean Transform Gate Outputs", mean_t, 0, 0.4),
        ("Transform Gate Outputs", torch.stack(one_t), 0, 1),
        ("Block Outputs", torch.stack(one_y), -1, 1),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (title, data, lo, hi) in zip(axes, panels):
        im = ax.imshow(data.numpy(), aspect="auto", cmap="coolwarm", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Block")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("Depth")
    name = Path(ckpt_path).stem
    fig.suptitle(f"{header(cfg)}, Epoch {state['epoch']}, {seen} Training Samples", fontsize=9)
    save(fig, FIG / f"gates_{name}.png", f"mean gate {mean_t.mean():.3f}")


def save(fig, out, info, rect=None):
    FIG.mkdir(exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=150)
    print(f"saved {out} ({info})")


def main():
    args = sys.argv[1:]
    if args and args[0] == "gates":
        for ckpt in args[1:]:
            gates(ckpt)
    elif args and args[0] == "compare":
        compare(wandb.Api(), args[1], args[2:])
    else:
        api = wandb.Api()
        for name in args or [f"highway_d{CONFIG['depth']}_w{CONFIG['width']}_b{CONFIG['gate_bias']:g}_{CONFIG['act']}"]:
            single(api, name)


if __name__ == "__main__":
    main()
