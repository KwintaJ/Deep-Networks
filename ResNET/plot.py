# pull run history from wandb and draw it like the paper
# usage: python plot.py resnet20                          -> figures/<name>.png (loss + train/test error)
#        python plot.py compare fig6_plain plain20 plain32  -> figures/fig6_plain.png, fig 6 style
#        python plot.py responses fig7 checkpoints/a.pt ... -> figures/fig7.png, std of layer responses
# picks the longest run with a given name if there are several

import sys
from pathlib import Path

import matplotlib
import torch
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ResNet
from train import CONFIG, get_data, run_name

FIG = Path("figures")


def fetch(api, name):
    runs = list(api.runs(f"{CONFIG['entity']}/{CONFIG['project']}", {"display_name": name}))
    if not runs:
        sys.exit(f"no run named {name}")
    run = max(runs, key=lambda r: r.summary.get("iter", 0))
    hist = run.history(samples=100000, pandas=False)
    hist = sorted((h for h in hist if h.get("test/err") is not None), key=lambda h: h["iter"])
    return run, hist


def label(cfg):
    return f"{'plain' if cfg.get('plain') else 'ResNet'}-{cfg['depth']}"


def pct(hist, key):
    return [100 * h[key] for h in hist]


def single(api, name):
    run, hist = fetch(api, name)
    it = [h["iter"] / 1e4 for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(it, [h["train/loss"] for h in hist], label="Training")
    a.plot(it, [h["test/loss"] for h in hist], label="Test")
    a.set_ylabel("Cross Entropy")
    a.set_title("Loss")
    b.plot(it, pct(hist, "train/err"), label="Training")
    b.plot(it, pct(hist, "val/err"), label="Validation")
    b.plot(it, pct(hist, "test/err"), label="Test")
    b.set_ylim(0, 20)
    b.set_ylabel("Error (%)")
    b.set_title(f"Error (Final Test: {100 * hist[-1]['test/err']:.2f}%)")
    for ax in (a, b):
        ax.set_xlabel("Iter (1e4)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{label(run.config)}, {run.summary.get('params', '?')} Parameters", fontsize=9)
    save(fig, FIG / f"{name}.png", f"{len(hist)} epochs, url {run.url}")


def compare(api, out, names):
    # fig 6: dashed = training error, bold = test error
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for i, name in enumerate(names):
        run, hist = fetch(api, name)
        it = [h["iter"] / 1e4 for h in hist]
        color = f"C{i}"
        ax.plot(it, pct(hist, "train/err"), "--", color=color, lw=0.8)
        ax.plot(it, pct(hist, "test/err"), color=color, lw=1.6,
                label=f"{label(run.config)} ({100 * hist[-1]['test/err']:.2f}%)")
    ax.set_ylim(0, 20)
    ax.set_xlabel("Iter (1e4)")
    ax.set_ylabel("Error (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, FIG / f"{out}.png", f"{len(names)} runs")


@torch.no_grad()
def responses(out, paths, samples=1000):
    # fig 7: std of each 3x3 layer response, taken after bn and before relu / the add,
    # over test images. top = original layer order, bottom = sorted by magnitude
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 5.5))
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["cfg"]
        net = ResNet(cfg["depth"], cfg["plain"])
        net.load_state_dict(state["net"])
        net.eval()
        stds = []
        bns = [m for blk in net.blocks for m in (blk.bn1, blk.bn2)]
        hooks = [bn.register_forward_hook(lambda m, i, o: stds.append(o.std(0).mean().item())) for bn in bns]
        x, _ = get_data(cfg, torch.device("cpu"))["test"]
        net(x[:samples])
        for h in hooks:
            h.remove()
        layers = range(1, len(stds) + 1)
        a.plot(layers, stds, lw=1, label=label(cfg))
        b.plot(layers, sorted(stds, reverse=True), lw=1, label=label(cfg))
    a.set_xlabel("Layer Index (Original)")
    b.set_xlabel("Layer Index (Sorted by Magnitude)")
    for ax in (a, b):
        ax.set_ylabel("Std")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, FIG / f"{out}.png", f"{len(paths)} checkpoints, {samples} test samples")


def save(fig, out, info):
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out} ({info})")


def main():
    args = sys.argv[1:]
    if args and args[0] == "responses":
        responses(args[1], args[2:])
    elif args and args[0] == "compare":
        compare(wandb.Api(), args[1], args[2:])
    else:
        api = wandb.Api()
        for name in args or [run_name(CONFIG)]:
            single(api, name)


if __name__ == "__main__":
    main()
