import random, numpy as np, torch
import csv, os, glob
from collections import defaultdict
import matplotlib.pyplot as plt


# ---------------- Utilities ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    

def run_id(arch, seed, lr, wd, ls, sig, init, rhythm, norec):
    return f"{arch}|{seed}|{lr:.12g}|{wd:.12g}|{ls:.12g}|{sig:.12g}|{init:.12g}|{rhythm}|{int(norec)}"


def timeseries_prepare_batch(batch, device):
    x, y = batch
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.long)

    # Make x -> (B, T, 1)
    if x.dim() == 3 and x.size(1) == 1:      # (B, 1, T) -> (B, T, 1)
        x = x.transpose(1, 2)
    elif x.dim() == 2:                       # (B, T) -> (B, T, 1)
        x = x.unsqueeze(-1)

    return x, y


def append_row(csv_path, cols, row):
    new = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})
        f.flush()
        os.fsync(f.fileno())


def run_one(
    loaders,
    model_fn,
    prepare_batch,
    arch_name: str,
    seed: int,
    lr: float,
    wd: float,
    ls: float,
    *,
    epochs: int,
    run_name: str | None = None,
    extra_row: dict | None = None,
):
    """
    One grid-search run for a single (seed, lr, wd, ls).

    model_fn: callable that returns a NEW model instance each time (important!)
    prepare_batch: function(batch, device) -> (x, y)
    """
    # Local import avoids circular import with train.py
    from train import TrainConfig, Trainer

    set_seed(seed)
    train_loader, val_loader, _test_loader = loaders[seed]

    model = model_fn()

    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        weight_decay=wd,
        label_smoothing=ls,
        seed=seed,
        history_csv=False,
    )
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    trainer = Trainer(cfg, prepare_batch=prepare_batch)

    if run_name is None:
        run_name = f"{arch_name}_seed{seed}_lr{lr:g}_wd{wd:g}_ls{ls:g}"

    results = trainer.fit(model, train_loader, val_loader, None, run_name=run_name)

    row = dict(
        arch_name=arch_name,
        seed=seed,
        lr=lr,
        weight_decay=wd,
        label_smoothing=ls,
        best_epoch=results["best_epoch"],
        best_val_acc=results["best_val_acc"],
    )
    if extra_row:
        row.update(extra_row)
    return row
    
def select_and_save_best_hparams(
    grid_dir: str,
    out_csv: str,
    grid_seeds: list[int],
    pattern: str = "*gridsearch*.csv",
    hp_cols: list[str] | None = None,
    val_col: str = "best_val_acc",
):
    """
    Load gridsearch CSVs, select best hyperparameters per architecture using only rows
    whose 'seed' is in grid_seeds, and save the selections to out_csv.

    Selection criterion:
      - maximize mean(val_col) across grid_seeds for each (arch_name, hp combo)
      - require all grid_seeds present (n_seeds == len(grid_seeds))
      - tie-breakers: smaller SEM, then smaller weight_decay, then smaller lr, then smaller label_smoothing

    Returns: pandas.DataFrame with one row per arch_name.
    """
    import os, glob
    import numpy as np
    import pandas as pd

    if hp_cols is None:
        hp_cols = ["lr", "weight_decay", "label_smoothing"]

    paths = sorted(glob.glob(os.path.join(grid_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No gridsearch CSVs found in {grid_dir} matching pattern '{pattern}'")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    required = {"arch_name", "seed", val_col, *hp_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in grid CSVs: {sorted(missing)}")

    df = df[df["seed"].isin(grid_seeds)].copy()
    if df.empty:
        raise ValueError(f"No rows remain after filtering to grid_seeds={grid_seeds}")

    summary = (
        df.groupby(["arch_name"] + hp_cols, dropna=False)
          .agg(
              n_seeds=("seed", "nunique"),
              mean_val=(val_col, "mean"),
              std_val=(val_col, "std"),
          )
          .reset_index()
    )

    summary = summary[summary["n_seeds"] == len(grid_seeds)].copy()
    if summary.empty:
        raise ValueError(
            "No (arch, hp) combos include all grid_seeds. "
            "Some runs likely failed or seed list mismatches the CSVs."
        )

    summary["sem_val"] = summary["std_val"].fillna(0.0) / np.sqrt(summary["n_seeds"])

    # Tie-breakers assume these exist; if you change hp_cols, adjust sort order accordingly
    sort_cols = ["arch_name", "mean_val", "sem_val"]
    sort_asc  = [True, False, True]

    for t in ["weight_decay", "lr", "label_smoothing"]:
        if t in summary.columns:
            sort_cols.append(t)
            sort_asc.append(True)

    best = (
        summary.sort_values(sort_cols, ascending=sort_asc)
               .groupby("arch_name", as_index=False)
               .head(1)
               .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best.to_csv(out_csv, index=False)
    return best


def read_best_params(best_csv):
    """
    Expects columns at least:
      arch_name, lr, weight_decay, label_smoothing
    """
    best = {}
    with open(best_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            arch = row["arch_name"]
            best[arch] = dict(
                lr=float(row["lr"]),
                weight_decay=float(row["weight_decay"]),
                label_smoothing=float(row["label_smoothing"]),
            )
    return best

def compute_confusion_matrix(model, loader, prepare_batch_fn, n_classes=10):
    """
    Returns an (n_classes x n_classes) confusion matrix with counts:
      cm[true_class, pred_class] += 1
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = prepare_batch_fn(batch, "cpu")

            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]  # e.g., (logits, aux)

            # logits -> predicted class
            y_pred = torch.argmax(out, dim=1)

            # handle one-hot labels if applicable
            if y.ndim > 1:
                y_true = torch.argmax(y, dim=1)
            else:
                y_true = y

            yt = y_true.view(-1).detach().cpu().numpy()
            yp = y_pred.view(-1).detach().cpu().numpy()

            # fast bincount trick
            idx = yt * n_classes + yp
            binc = np.bincount(idx, minlength=n_classes * n_classes)
            cm += binc.reshape(n_classes, n_classes)

    return cm

def row_normalize_confmat(cm: np.ndarray) -> np.ndarray:
    """
    Convert a count confusion matrix to row-normalized proportions.
    Output has same shape; rows sum to 1 when row sum > 0.
    """
    cm = cm.astype(np.float64, copy=False)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=(row_sums != 0))


def load_confmats_for_arch(cm_dir: str, arch_name: str, pattern: str = "confmat_{arch}_seed*.npy"):
    """
    Load confusion matrices for a given architecture.

    Returns
    -------
    cms : np.ndarray
        Array of shape (n_seeds, n_classes, n_classes)
    paths : list[str]
        Matching file paths, sorted.
    """
    glob_pat = os.path.join(cm_dir, pattern.format(arch=arch_name))
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No confusion matrices found for '{arch_name}' using: {glob_pat}")
    cms = [np.load(p) for p in paths]
    return np.stack(cms, axis=0), paths

def draw_confmat_on_ax(
    ax,
    avg_cm: np.ndarray,
    arch_name: str,
    n_classes: int = 10,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "Blues",
    fontsize: int = 8,
    show_axis_labels: bool = True,
):
    """
    Draw a (row-normalized) confusion matrix on an existing matplotlib Axes,
    annotated with values, and titled with the architecture name.
    """
    ax.imshow(avg_cm, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    if show_axis_labels:
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Replace generic title with architecture name
    ax.set_title(arch_name)

    # annotate values
    for i in range(n_classes):
        for j in range(n_classes):
            val = float(avg_cm[i, j])
            txt_color = "white" if val >= 0.5 else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=txt_color, fontsize=fontsize)

    # keep (0,0) at top-left
    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(n_classes - 0.5, -0.5)

