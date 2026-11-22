# ResNet on CIFAR-10

PyTorch implementation of Deep Residual Learning (He, Zhang, Ren, Sun, 2015) trained on CIFAR-10, the sec 4.2 setup

## Math

`y = F(x, {Wi}) + x, F = conv-bn-relu-conv-bn, relu after the add`

- 3x3 conv with 16 filters, then 3 stages of n blocks (2n layers) on 32x32, 16x16, 8x8 maps with 16, 32, 64 filters
- global avg pool, fc 10, depth = 6n+2 (20, 32, 44, 56, 110)
- shortcut option A: identity, when dims grow subsample by 2 and zero pad the channels, no extra params
- plain net = same net without the shortcuts
- sgd lr 0.1, momentum 0.9, weight decay 1e-4, batch 128, lr / 10 at 32k and 48k iters, stop at 64k
- 45k/5k train/val split, pad 4 + random crop + horizontal flip

## Run

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                                  # defaults, resnet20
python train.py depth=56 plain=true device=cpu   # any CONFIG key can be overridden
python train.py depth=110 warmup=400             # paper warms up resnet110 at lr 0.01
```

First run downloads cifar10 dataset (from a mirror)
