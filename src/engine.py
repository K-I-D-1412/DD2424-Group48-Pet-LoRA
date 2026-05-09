"""
Training and evaluation loops.

Pure functions that act on a model + dataloader + loss + optimizer.
LoRA, gradual unfreezing and imbalance experiments all reuse the same engine
so behaviour stays consistent across strategies.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import compute_macro_f1, compute_top1_accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: Optional[int] = None,
    epoch_index: Optional[int] = None,
) -> dict:
    """
    Run one epoch of training. Only parameters with `requires_grad=True`
    will actually update because the optimizer was constructed from those
    parameters and the others receive no gradient.

    Returns: {"loss", "top1", "num_samples", "time_seconds"}.
    """
    model.train()

    running_loss_sum = 0.0
    running_correct = 0
    running_count = 0

    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss_sum += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            running_count += batch_size

        if log_every is not None and (batch_idx + 1) % log_every == 0:
            avg_loss = running_loss_sum / running_count
            avg_acc = running_correct / running_count
            prefix = f"  epoch {epoch_index} | " if epoch_index is not None else "  "
            print(
                f"{prefix}batch {batch_idx + 1}/{len(loader)} "
                f"| loss {avg_loss:.4f} | acc {avg_acc:.4f}"
            )

    avg_loss = running_loss_sum / max(running_count, 1)
    avg_acc = running_correct / max(running_count, 1)
    elapsed = time.time() - start

    return {
        "loss": avg_loss,
        "top1": avg_acc,
        "num_samples": running_count,
        "time_seconds": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    return_predictions: bool = True,
) -> dict:
    """
    Evaluate the model on the given dataloader.

    Returns: {"loss", "top1", "macro_f1", "num_samples",
              "predictions", "labels", "logits"}.

    `return_predictions=False` controls whether the raw arrays are returned
    in the result dict, NOT whether they are used to compute metrics. Metrics
    are always computed from the actual predictions. The arrays are tiny
    (a few MB at most for our dataset sizes), so the memory cost of always
    collecting them is negligible.
    """
    model.eval()

    total_loss_sum = 0.0
    total_count = 0

    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss_sum += loss.item() * batch_size
        total_count += batch_size

        # Always collect for metric computation. Storage cost for our scale
        # (~3700 samples * 37 classes * 4 bytes ~= 0.5 MB) is trivial.
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_logits.append(logits.detach().cpu())

    preds_np = torch.cat(all_preds).numpy().astype(np.int64)
    labels_np = torch.cat(all_labels).numpy().astype(np.int64)
    logits_np = torch.cat(all_logits).numpy().astype(np.float32)

    avg_loss = total_loss_sum / max(total_count, 1)
    top1 = compute_top1_accuracy(preds_np, labels_np)
    macro_f1 = compute_macro_f1(preds_np, labels_np, num_classes=num_classes)

    result = {
        "loss": avg_loss,
        "top1": top1,
        "macro_f1": macro_f1,
        "num_samples": total_count,
    }

    if return_predictions:
        result["predictions"] = preds_np
        result["labels"] = labels_np
        result["logits"] = logits_np
    else:
        # Keep these keys for downstream code that always reads them, but
        # use empty placeholders to indicate the caller didn't want them.
        result["predictions"] = np.empty((0,), dtype=np.int64)
        result["labels"] = np.empty((0,), dtype=np.int64)
        result["logits"] = np.empty((0, num_classes), dtype=np.float32)

    return result