# train resnet / plain net on cifar10 (sec 4.2), log to wandb
# usage: python train.py
#        python train.py depth=56 plain=true device=cpu wandb=false
#        python train.py depth=56 resume=true   (continue from checkpoints/<name>.pt)

import json
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torchvision import datasets

from model import ResNet

CONFIG = dict(
    depth=20,            # 6n+2: 20 32 44 56 110
    plain=False,         # true = no shortcuts
    iters=64000,         # paper stops at 64k
    milestones=[32000, 48000],  # lr / 10 here
    warmup=0,            # iters at lr/10 first, paper uses ~400 for resnet110
    batch_size=128,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    val_size=5000,       # 45k/5k split like the paper
    subset=None,         # limit train set
    seed=42,
    device="auto",       # auto = mps > cuda > cpu
    resume=False,
    suffix="",           # appended to run name
    wandb=True,
    entity="jankwinta-ju",
    project="resnet-cifar10",
)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
MIRROR = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"


def parse_args(cfg):
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        if k not in cfg:
            sys.exit(f"unknown option {k}")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_name(cfg):
    return f"{'plain' if cfg['plain'] else 'resnet'}{cfg['depth']}{cfg['suffix']}"


def to_tensor(ds, device):
    # whole set fits in memory, so keep it on the device and skip the dataloader
    x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div(255)
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    return x.to(device), torch.tensor(ds.targets).to(device)


def get_data(cfg, device, root="data"):
    root = Path(root)
    root.mkdir(exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.exists() and not (root / "cifar-10-batches-py").exists():
        print("downloading cifar10 from mirror")
        urllib.request.urlretrieve(MIRROR, archive)
    x, y = to_tensor(datasets.CIFAR10(root, train=True, download=True), device)
    tx, ty = to_tensor(datasets.CIFAR10(root, train=False, download=True), device)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(cfg["seed"])).to(device)
    val_idx, train_idx = perm[: cfg["val_size"]], perm[cfg["val_size"] :]
    if cfg["subset"]:
        train_idx = train_idx[: cfg["subset"]]
    return dict(train=(x[train_idx], y[train_idx]), val=(x[val_idx], y[val_idx]), test=(tx, ty))


def augment(x):
    # pad 4, random 32x32 crop, random horizontal flip (same as the paper)
    n = len(x)
    x = F.pad(x, (4, 4, 4, 4))
    r = torch.arange(32, device=x.device)
    i = torch.randint(0, 9, (n, 1, 1, 1), device=x.device) + r.view(1, 1, 32, 1)
    j = torch.randint(0, 9, (n, 1, 1, 1), device=x.device) + r.view(1, 1, 1, 32)
    x = x[torch.arange(n, device=x.device).view(n, 1, 1, 1), torch.arange(3, device=x.device).view(1, 3, 1, 1), i, j]
    flip = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    return torch.where(flip, x.flip(3), x)


@torch.no_grad()
def evaluate(net, data, bs=1000):
    net.eval()
    x, y = data
    loss, wrong = 0.0, 0
    for k in range(0, len(x), bs):
        out = net(x[k : k + bs])
        loss += F.cross_entropy(out, y[k : k + bs], reduction="sum").item()
        wrong += (out.argmax(1) != y[k : k + bs]).sum().item()
    return loss / len(x), wrong / len(x)


def main():
    cfg = parse_args(dict(CONFIG))
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    name = run_name(cfg)
    ckpt = Path("checkpoints") / f"{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)

    data = get_data(cfg, device)
    net = ResNet(cfg["depth"], cfg["plain"]).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    # stepped every iteration, not every epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (0.1 if it < cfg["warmup"] else 1.0) * 0.1 ** sum(it >= m for m in cfg["milestones"]))
    start, it, run_id = 1, 0, None
    if cfg["resume"] and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, it, run_id = state["epoch"] + 1, state["iter"], state["run_id"]
        print(f"resumed {ckpt} at epoch {start}, iter {it}")

    run = wandb.init(
        entity=cfg["entity"], project=cfg["project"], name=name, config=cfg, tags=["fig6"],
        id=run_id, resume="allow" if run_id else None, mode="online" if cfg["wandb"] else "disabled",
    )
    wandb.define_metric("test/err", summary="min")
    params = sum(p.numel() for p in net.parameters())
    run.summary["params"] = params
    x_train, y_train = data["train"]
    n, bs = len(x_train), cfg["batch_size"]
    epochs = -(-cfg["iters"] * bs // n)
    print(f"{name}: {params} params on {device}, {cfg['iters']} iters = {epochs} epochs")

    epoch = start
    while it < cfg["iters"]:
        t0 = time.time()
        net.train()
        loss_sum = torch.zeros((), device=device)
        wrong = torch.zeros((), device=device)
        seen = 0
        perm = torch.randperm(n, device=device)
        # drop the last partial batch so every iter is a full 128
        for k in range(0, n - bs + 1, bs):
            if it >= cfg["iters"]:
                break
            idx = perm[k : k + bs]
            x, y = augment(x_train[idx]), y_train[idx]
            out = net(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            # summed on the device, .item() every step would sync mps and slow things down
            loss_sum += loss.detach() * len(y)
            wrong += (out.argmax(1) != y).sum()
            seen += len(y)
        lr = opt.param_groups[0]["lr"]
        train_loss, train_err = loss_sum.item() / seen, wrong.item() / seen
        val_loss, val_err = evaluate(net, data["val"]) if cfg["val_size"] else (float("nan"),) * 2
        test_loss, test_err = evaluate(net, data["test"])
        run.log(dict(epoch=epoch, iter=it, lr=lr, epoch_time=time.time() - t0,
                     **{"train/loss": train_loss, "train/err": train_err, "val/loss": val_loss, "val/err": val_err,
                        "test/loss": test_loss, "test/err": test_err}))
        print(f"[{epoch}/{epochs}] iter {it} train loss {train_loss:.3f} err {train_err:.4f} | "
              f"val err {val_err:.4f} | test err {test_err:.4f} | lr {lr:.0e} | {time.time() - t0:.0f}s")
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, iter=it, run_id=run.id, cfg=cfg), ckpt)
        epoch += 1

    run.summary.update({"final/test_loss": test_loss, "final/test_err": test_err})
    print(f"final test loss {test_loss:.3f} err {test_err:.4f}")
    run.finish()


if __name__ == "__main__":
    main()
