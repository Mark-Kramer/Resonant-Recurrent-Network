from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Dict, Any
import os, time, csv
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Utilities import set_seed


# ---------------- Utils ----------------
def get_logits(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    raise ValueError("Model forward must return logits Tensor, (logits, ...), or {'logits': ...}.")


def per_class_tp(conf: np.ndarray) -> Sequence[int]:
    return np.diag(conf)


def format_tp(tp: Sequence[int]) -> str:
    return ", ".join([f"{i}={int(v)}" for i, v in enumerate(tp)])


# ---------------- Config ----------------
@dataclass
class TrainConfig:
    num_classes:         int = 10
    epochs:              int = 30
    lr:                  float = 1e-3
    weight_decay:        float = 1e-2
    label_smoothing:     float = 0.0
    grad_clip_norm:      float = 1.0
    scheduler:           str = "onecycle"
    seed:                int = 123
    log_dir:             str = "out/logs"
    ckpt_dir:            str = "out/checkpoints"
    history_csv:         bool = True


# ---------------- Trainer ----------------
class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        prepare_batch: Callable[[Tuple[torch.Tensor, torch.Tensor], torch.device], Tuple[torch.Tensor, torch.Tensor]],
        class_names: Optional[Sequence[str]] = None,
    ):
        self.cfg = cfg
        self.prepare_batch = prepare_batch
        self.class_names = class_names if class_names is not None else [str(i) for i in range(cfg.num_classes)]
        set_seed(cfg.seed)
        self.device = torch.device("cpu")

    def _make_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _make_scheduler(self, optimizer: optim.Optimizer, steps_per_epoch: int):
        if self.cfg.scheduler == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.lr,
                epochs=self.cfg.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.15,
                anneal_strategy="cos",
                div_factor=25.0,
            )
        return None

    def _evaluate(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, Any]:
        model.eval()
        y_true, y_pred = [], []
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                x, y = self.prepare_batch(batch, self.device)
                logits = get_logits(model(x))
                loss = loss_fn(logits, y)

                bs = y.size(0)
                total_loss += float(loss.detach().cpu()) * bs
                n_samples += bs
                
                preds = torch.argmax(logits, dim=1)
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = accuracy_score(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred, labels=list(range(self.cfg.num_classes)))

        return {
            "loss": total_loss / max(1, n_samples),
            "acc": acc,
            "conf": conf,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        run_name: str = "model",
    ) -> Dict[str, Any]:

        model.to(self.device)

        loss_fn   = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        optimizer = self._make_optimizer(model)
        steps_per_epoch = len(train_loader)
        scheduler = self._make_scheduler(optimizer, steps_per_epoch)

        best_val_acc = -np.inf
        best_epoch = -1
        epochs_no_improve = 0

        ckpt_path = os.path.join(self.cfg.ckpt_dir, f"{run_name}_best.pt")
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)

        hist_csv_path = os.path.join(self.cfg.log_dir, f"{run_name}_history.csv")
        if self.cfg.history_csv:
            with open(hist_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            n_samples = 0
            y_true_tr, y_pred_tr = [], []

            for batch in train_loader:
                x, y = self.prepare_batch(batch, self.device)
                optimizer.zero_grad(set_to_none=True)

                logits = get_logits(model(x))
                loss = loss_fn(logits, y)

                loss.backward()
                if self.cfg.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                optimizer.step()

                proj = getattr(model, "project_W_no_diag", None)
                if callable(proj):
                    proj()

                if scheduler is not None and self.cfg.scheduler == "onecycle":
                    scheduler.step()

                bs = y.size(0)
                total_loss += float(loss.detach().cpu()) * bs
                n_samples += bs
                preds = torch.argmax(logits, dim=1)
                y_true_tr.append(y.detach().cpu().numpy())
                y_pred_tr.append(preds.detach().cpu().numpy())

            y_true_tr = np.concatenate(y_true_tr)
            y_pred_tr = np.concatenate(y_pred_tr)
            train_acc = accuracy_score(y_true_tr, y_pred_tr)
            train_conf = confusion_matrix(y_true_tr, y_pred_tr, labels=list(range(self.cfg.num_classes)))
            train_tp = per_class_tp(train_conf)
            train_loss = total_loss / max(1, n_samples)

            if val_loader is not None:
                eval_val = self._evaluate(model, val_loader, loss_fn)
                val_loss, val_acc, val_conf = eval_val["loss"], eval_val["acc"], eval_val["conf"]
                val_tp = per_class_tp(val_conf)
            else:
                val_loss, val_acc, val_tp = np.nan, np.nan, []

            if scheduler is not None and self.cfg.scheduler != "onecycle":
                scheduler.step()

            improved = val_loader is None or (val_acc > best_val_acc + 1e-6)
            if improved:
                best_val_acc = val_acc if val_loader is not None else train_acc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save({"model_state": model.state_dict(),
                            "epoch": epoch,
                            "val_acc": best_val_acc}, ckpt_path)
            else:
                epochs_no_improve += 1

            ep_time = time.time() - t0
            train_tp_str = format_tp(train_tp)
            val_tp_str = format_tp(val_tp) if len(val_tp) else ""
            print(
                f"Epoch {epoch:02d} | {ep_time:5.1f}s | "
                f"loss {train_loss:.4f} | train_acc {train_acc:.3f} "
                f"| val_acc {val_acc:.3f} | TP(train): {train_tp_str}"
                + (f" | TP(val): {val_tp_str}" if val_tp_str else "")
            )

            if self.cfg.history_csv:
                with open(hist_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state["model_state"])

        results = {"best_epoch": best_epoch, "best_val_acc": best_val_acc}

        if test_loader is not None:
            eval_te = self._evaluate(model, test_loader, loss_fn)
            conf = eval_te["conf"]
            print("\n=== TEST RESULTS ===")
            print(f"test_loss: {eval_te['loss']:.4f} | test_acc: {eval_te['acc']:.3f}")
            print("Confusion matrix (rows=true, cols=pred):")
            print(conf)
            print("\nClassification report:")
            target_names = self.class_names if len(self.class_names) == self.cfg.num_classes else None
            print(classification_report(eval_te["y_true"], eval_te["y_pred"], digits=3, target_names=target_names))
            results.update({"test": eval_te})

        return results
