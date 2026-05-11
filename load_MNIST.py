import os
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

__all__ = ["load_mnist_numpy_flat01", "make_timeseries_loaders"]

# ---------------- MNIST load ----------------
def load_mnist_numpy_flat01():
    """
    Returns:
        X_train: (60000, 784) float32 in [0,1]
        y_train: (60000,) int64
        X_test:  (10000, 784) float32 in [0,1]
        y_test:  (10000,) int64
    """
    X_train = X_test = y_train = y_test = None
    from torchvision import datasets  # type: ignore
    ds_tr = datasets.MNIST(root="./mnistdata", train=True,  download=True, transform=None)
    ds_te = datasets.MNIST(root="./mnistdata", train=False, download=True, transform=None)
    X_train = ds_tr.data.numpy()
    y_train = ds_tr.targets.numpy()
    X_test  = ds_te.data.numpy()
    y_test  = ds_te.targets.numpy()

    # Flatten to sequences and scale to [0,1]
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)  # (60000, 784)
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32)   # (10000, 784)
    X_train /= 255.0
    X_test  /= 255.0

    # Ensure label dtypes
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    # Basic stats
    print("[INFO] Label dist (train):", dict(Counter(y_train)))
    print("[INFO] Label dist (test) :", dict(Counter(y_test)))

    return X_train, y_train, X_test, y_test

def make_timeseries_loaders(
    batch_size: int = 128,
    val_size: int = 5_000,
    seed: int = 123,
    num_workers: int | None = None,
    print_stats: bool = False,
):
    """
    Build PyTorch DataLoaders from flattened MNIST time series (T=784).
    
    Args:
        batch_size: per-batch size
        val_size:   number of samples carved out of train for validation
        seed:       RNG seed for a fixed train/val split
        num_workers: DataLoader workers (default: os.cpu_count()-1, min 0)
        print_stats: print split sizes and example shapes

    Returns:
        train_loader, val_loader, test_loader
    """
    X_tr_np, y_tr_np, X_te_np, y_te_np = load_mnist_numpy_flat01()  # (N,784), (N,)
    N = X_tr_np.shape[0]
    T = X_tr_np.shape[1]
    
    # Fixed split for fairness
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    val_idx = idx[:val_size]
    tr_idx  = idx[val_size:]
    #tr_idx  = idx[val_size:val_size*2]

    # Make training, validation, and testing data.
    X_tr = torch.from_numpy(X_tr_np[tr_idx]) 
    y_tr = torch.from_numpy(y_tr_np[tr_idx]).long()
    X_va = torch.from_numpy(X_tr_np[val_idx])
    y_va = torch.from_numpy(y_tr_np[val_idx]).long()
    X_te = torch.from_numpy(X_te_np)
    y_te = torch.from_numpy(y_te_np).long()

    ds_tr = TensorDataset(X_tr, y_tr)
    ds_va = TensorDataset(X_va, y_va)
    ds_te = TensorDataset(X_te, y_te)

    if num_workers is None:
        # Reasonable default; never negative
        num_workers = max(0, (os.cpu_count() or 2) - 1)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(ds_tr, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(ds_va, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(ds_te, shuffle=False, **loader_kwargs)

    if print_stats:
        xb, yb = next(iter(train_loader))
        print(f"[INFO] Train/Val/Test sizes: {len(ds_tr)}/{len(ds_va)}/{len(ds_te)}")
        print(f"[INFO] Example batch: x {tuple(xb.shape)} {xb.dtype}, y {tuple(yb.shape)} {yb.dtype}")
        print(f"[INFO] Sequence length T={X_tr_np.shape[1]}")

    return train_loader, val_loader, test_loader
