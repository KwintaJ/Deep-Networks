# Highway Networks on CIFAR-10

PyTorch implementation of Highway Networks (Srivastava, Greff, Schmidhuber, 2015) trained on CIFAR-10

## Math

`y = H(x)*T(x) + x*(1-T(x)), T = sigmoid(Wt x + bt), carry gate = 1-T`

- gate bias starts negative so the net starts as identity (paper: -4)
- other weights he init
- first layer is plain, changes dim 3072 -> 50, then depth-1 highway layers of 50 blocks, softmax 10

## Run

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                                   # defaults, depth 50
python train.py depth=100 epochs=400 device=cpu   # any CONFIG key can be overridden
```

First run downloads cifar10 dataset (from a mirror)
