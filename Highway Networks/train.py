# train highway net on cifar100, log to wandb
# usage: python train.py
#        python train.py depth=100 epochs=50 device=cpu wandb=false
#        python train.py depth=100 resume=true   (continue from checkpoints/<name>.pt)

import json
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import HighwayNet

CONFIG = dict(
    depth=50,            # total hidden layers
    width=50,            # blocks per layer
    act="tanh",          # relu | tanh
    gate_bias=-4.0,
    epochs=400,
    batch_size=128,
    lr=0.05,
    momentum=0.9,
    lr_decay=0.99,       # per epoch, exponential like fig 1
    grad_clip=None,      # max grad norm, relu nets blow up without it
    val_size=5000,       # taken from the 50k train set
    subset=None,         # limit train set
    seed=42,
    device="auto",       # auto = mps > cuda > cpu
    resume=False,
    suffix="",           # appended to run name, eg _lr0.1
    wandb=True,
    entity="jankwinta-ju",
    project="highway-networks",
)

MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
MIRROR = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"


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


def get_data(cfg, root="data"):
    root = Path(root)
    root.mkdir(exist_ok=True)
    archive = root / "cifar-100-python.tar.gz"
    if not archive.exists() and not (root / "cifar-100-python").exists():
        print("downloading cifar100 from mirror")
        urllib.request.urlretrieve(MIRROR, archive)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.CIFAR100(root, train=True, download=True, transform=tf)
    test = datasets.CIFAR100(root, train=False, download=True, transform=tf)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    val_idx, train_idx = perm[: cfg["val_size"]], perm[cfg["val_size"] :]
    if cfg["subset"]:
        train_idx = train_idx[: cfg["subset"]]
    bs = cfg["batch_size"]
    return dict(
        train=DataLoader(Subset(train, train_idx), bs, shuffle=True),
        train_eval=DataLoader(Subset(train, train_idx), 1024),
        val=DataLoader(Subset(train, val_idx), 1024),
        test=DataLoader(test, 1024),
    )


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss / n, correct / n


def main():
    cfg = parse_args(dict(CONFIG))
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    name = f"highway_d{cfg['depth']}_w{cfg['width']}_b{cfg['gate_bias']:g}_{cfg['act']}{cfg['suffix']}"
    ckpt = Path("checkpoints") / f"{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)

    data = get_data(cfg)
    net = HighwayNet(cfg["depth"], cfg["width"], cfg["act"], cfg["gate_bias"]).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, cfg["lr_decay"])
    start, run_id = 1, None
    if cfg["resume"] and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, run_id = state["epoch"] + 1, state["run_id"]
        print(f"resumed {ckpt} at epoch {start}")

    run = wandb.init(
        entity=cfg["entity"], project=cfg["project"], name=name, config=cfg, tags=["fig1"],
        id=run_id, resume="allow" if run_id else None, mode="online" if cfg["wandb"] else "disabled",
    )
    wandb.define_metric("train/cross_entropy", summary="min")  # fig 1 y axis, set log scale in the ui
    params = sum(p.numel() for p in net.parameters())
    run.summary["params"] = params
    print(f"{name}: {params} params on {device}")

    for epoch in range(start, cfg["epochs"] + 1):
        t0 = time.time()
        net.train()
        for x, y in tqdm(data["train"], desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
        lr = opt.param_groups[0]["lr"]
        sched.step()
        # mean cross entropy over the whole train set
        train_ce, train_acc = evaluate(net, data["train_eval"], device)
        val_loss, val_acc = evaluate(net, data["val"], device)
        run.log(dict(epoch=epoch, lr=lr, epoch_time=time.time() - t0,
                     **{"train/cross_entropy": train_ce, "train/acc": train_acc,
                        "val/loss": val_loss, "val/acc": val_acc}))
        print(f"[{epoch}/{cfg['epochs']}] train ce {train_ce:.4e} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} | lr {lr:.2e} | {time.time() - t0:.0f}s")
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, run_id=run.id, cfg=cfg), ckpt)

    test_loss, test_acc = evaluate(net, data["test"], device)
    run.summary.update({"test/loss": test_loss, "test/acc": test_acc})
    print(f"test loss {test_loss:.3f} acc {test_acc:.3f}")
    run.finish()


if __name__ == "__main__":
    main()
