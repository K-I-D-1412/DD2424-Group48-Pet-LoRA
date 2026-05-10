"""
Training entry point.

Jingmeng - build the baselines training with 2 strategies, and save the training results.
    train_from_config
    unfrozen_layers (Strategy 1)
    gradual unfreezing scheduler (Strategy 2)


Usage:
    python -m src.train --config configs/linear_probe_100.yaml
    python -m src.train --config configs/finetune_100.yaml

Full output persistence.

For each experiment writes to `experiments/{name}/`:
    config.yaml             - copy of the resolved config
    training_curves.csv     - per-epoch loss / acc / f1 / lr / time
    metrics.json            - best-epoch summary + per-class metrics +
                              confusion matrix (val and, if available, test)
    best_model.pt           - state_dict of the best (val top-1) epoch
    best_predictions.npz    - val_predictions / val_labels / val_logits and
                              the same for test
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from src.data import get_dataloaders_for_split, get_test_dataloader
from src.engine import evaluate, train_one_epoch
from src.metrics import (
    compute_confusion_matrix,
    compute_per_class_accuracy,
    compute_per_class_f1,
)
from src.models import build_resnet18, configure_model_for_strategy
from src.utils import count_parameters, get_device, load_config, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Optimizer / loss factories 
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    optimizer_name = training_cfg["optimizer"].lower()
    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg["weight_decay"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    if optimizer_name == "adam":
        return torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_loss(config: dict, class_weights: torch.Tensor, device: torch.device) -> nn.Module:
    use_weighted_loss = bool(config["data"].get("use_weighted_loss", False))
    if use_weighted_loss:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# I/O helpers 
# ---------------------------------------------------------------------------

def save_training_curves(history: list, path: Path):
    """Write per-epoch records to a CSV. Empty history is a no-op."""
    if not history:
        return
    keys = list(history[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_yaml_copy(config: dict, path: Path):
    """Persist the resolved config so we can reproduce this run later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_from_config(config_path: str):
    config = load_config(config_path)

    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = get_device()

    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    logging_cfg = config.get("logging", {})

    # ---------------- output dir ----------------
    output_dir = Path(
        logging_cfg.get(
            "output_dir",
            f"experiments/{config['experiment']['name']}",
        )
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # snapshot the resolved config first thing.
    save_yaml_copy(config, output_dir / "config.yaml")

    # ---------------- data ----------------
    data = get_dataloaders_for_split(
        train_split_name=data_cfg["train_split"],
        val_split_name=data_cfg["val_split"],
        image_size=int(data_cfg["image_size"]),
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg.get("num_workers", 2)),
        use_augmentation=bool(data_cfg["use_augmentation"]),
        use_oversampling=bool(data_cfg["use_oversampling"]),
        download=False,
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    num_classes = data["num_classes"]
    class_weights = data["class_weights"]

    # ---------------- model ----------------
    architecture = model_cfg["architecture"].lower()
    if architecture == "resnet18":
        model = build_resnet18(
            num_classes=int(model_cfg["num_classes"]),
            pretrained=bool(model_cfg["pretrained"]),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

# Adding unfrozen_layers
    strategy = model_cfg["strategy"].lower()
    unfrozen_layers = model_cfg.get("unfrozen_layers", None)
    lora_cfg = config.get("lora", None)
    model = configure_model_for_strategy(
        model=model,
        strategy=strategy,
        unfrozen_layers=unfrozen_layers,
        lora_cfg=lora_cfg,
    )
    model = model.to(device)

    # ---------------- loss + optimizer ----------------
    loss_fn = build_loss(config=config, class_weights=class_weights, device=device)
    optimizer = build_optimizer(model=model, config=config)

    total_params, trainable_params = count_parameters(model)

    # ---------------- header ----------------
    print("=" * 80)
    print(f"Experiment: {config['experiment']['name']}")
    print("=" * 80)
    print(f"Device           : {device}")
    print(f"Strategy         : {strategy}")
    print(f"Train samples    : {len(data['train_dataset'])}")
    print(f"Val samples      : {len(data['val_dataset'])}")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Epochs           : {training_cfg['epochs']}")
    print(
        f"Optimizer        : {training_cfg['optimizer']}, "
        f"lr={training_cfg['learning_rate']}, wd={training_cfg['weight_decay']}"
    )
    print(f"Output dir       : {output_dir}")
    print("=" * 80)

# ---------------- gradual unfreezing scheduler (Strategy 2) ----------------
    unfreezing_scheduler = None
    if strategy == "gradual_unfreezing":
        from src.schedulers import GradualUnfreezingScheduler
        schedule = model_cfg.get("gradual_unfreezing_schedule")
        if schedule is None:
            raise ValueError(
                "strategy='gradual_unfreezing' needs to set in config"
                "model.gradual_unfreezing_schedule"
            )
        unfreezing_scheduler = GradualUnfreezingScheduler(model, schedule)
        print("[Gradual Unfreezing] schedule:")
        print(unfreezing_scheduler.describe())
        # Apply the schedule of epoch 0 and immediately rebuild the optimizer
        changed = unfreezing_scheduler.step(0)
        if changed:
            optimizer = build_optimizer(model=model, config=config)
            _, n_trainable = count_parameters(model)
            print(f"[Unfreeze] epoch 0 init: trainable_params = {n_trainable:,}")

    # ---------------- main training loop ----------------
    epochs = int(training_cfg["epochs"])
    history: list = []         # per-epoch records
    best_val_top1 = -1.0
    best_epoch = -1
    best_ckpt_path = output_dir / "best_model.pt"
    save_best = bool(logging_cfg.get("save_best_checkpoint", True))

    train_start = time.time()

    for epoch in range(epochs):
        # ── Strategy 2: Check if it is the right time to unfreeze the new block ──────────────────
        if unfreezing_scheduler is not None and epoch > 0:
            changed = unfreezing_scheduler.step(epoch)
            if changed:
                optimizer = build_optimizer(model=model, config=config)
                _, n_trainable = count_parameters(model)
                print(
                    f"[Unfreeze] epoch {epoch}: "
                    f"trainable_params = {n_trainable:,}, optimizer rebuilt."
                )
        # ────────────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model, loader=train_loader, loss_fn=loss_fn,
            optimizer=optimizer, device=device,
        )
        val_metrics = evaluate(
            model=model, loader=val_loader, loss_fn=loss_fn,
            device=device, num_classes=num_classes,
            return_predictions=False,
        )

        marker = ""
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            if save_best:
                torch.save(model.state_dict(), best_ckpt_path)
            marker = "  *new best*"

        # accumulate every epoch into history for the CSV.
        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_top1": round(train_metrics["top1"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_top1": round(val_metrics["top1"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "epoch_time_seconds": round(train_metrics["time_seconds"], 3),
        }
        history.append(record)

        print(
            f"epoch {epoch:3d} | "
            f"train_loss {record['train_loss']:.4f} | train_top1 {record['train_top1']:.4f} | "
            f"val_loss {record['val_loss']:.4f} | val_top1 {record['val_top1']:.4f} | "
            f"val_macro_f1 {record['val_macro_f1']:.4f} | "
            f"epoch_time {record['epoch_time_seconds']:.1f}s"
            f"{marker}"
        )

    total_train_time = time.time() - train_start

    # persist curves first thing, even before the test eval might fail.
    save_training_curves(history, output_dir / "training_curves.csv")

    print("=" * 80)
    print(
        f"Training done. Best epoch={best_epoch}, "
        f"best_val_top1={best_val_top1:.4f}, "
        f"total_time={total_train_time:.1f}s"
    )
    print("=" * 80)

    # ---------------- final evaluation on the best checkpoint ----------------
    if save_best and best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Loaded best checkpoint from {best_ckpt_path}")

    # re-evaluate on val WITH predictions, to compute per-class metrics
    # confusion matrix for the saved best model.
    val_best = evaluate(
        model=model, loader=val_loader, loss_fn=loss_fn,
        device=device, num_classes=num_classes,
        return_predictions=True,
    )

    final_metrics = {
        "experiment_name": config["experiment"]["name"],
        "strategy": strategy,
        "lora": config.get("lora", None),
        "train_split": data_cfg["train_split"],
        "val_split": data_cfg["val_split"],
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_top1": float(val_best["top1"]),
        "best_val_macro_f1": float(val_best["macro_f1"]),
        "best_val_loss": float(val_best["loss"]),
        "final_val_top1": float(history[-1]["val_top1"]),
        "final_val_macro_f1": float(history[-1]["val_macro_f1"]),
        "total_parameters": total_params,
        "trainable_parameters_final": count_parameters(model)[1],
        "training_time_seconds": round(total_train_time, 2),
        "device": str(device),
        "per_class_val_accuracy": compute_per_class_accuracy(
            val_best["predictions"], val_best["labels"], num_classes
        ),
        "per_class_val_f1": compute_per_class_f1(
            val_best["predictions"], val_best["labels"], num_classes
        ),
        "val_confusion_matrix": compute_confusion_matrix(
            val_best["predictions"], val_best["labels"], num_classes
        ),
        "class_names": data["class_names"],
    }

    # ---------------- official test set evaluation ----------------
    test_results = None
    if data_cfg.get("test_split", None) == "official_test":
        try:
            test_data = get_test_dataloader(
                image_size=int(data_cfg["image_size"]),
                batch_size=int(training_cfg["batch_size"]),
                num_workers=int(training_cfg.get("num_workers", 2)),
                download=False,
            )
            test_results = evaluate(
                model=model, loader=test_data["test_loader"],
                loss_fn=loss_fn, device=device,
                num_classes=num_classes, return_predictions=True,
            )
            final_metrics["test_top1"] = float(test_results["top1"])
            final_metrics["test_macro_f1"] = float(test_results["macro_f1"])
            final_metrics["per_class_test_accuracy"] = compute_per_class_accuracy(
                test_results["predictions"], test_results["labels"], num_classes
            )
            final_metrics["per_class_test_f1"] = compute_per_class_f1(
                test_results["predictions"], test_results["labels"], num_classes
            )
            final_metrics["test_confusion_matrix"] = compute_confusion_matrix(
                test_results["predictions"], test_results["labels"], num_classes
            )
            print(
                f"[TEST] top1 = {test_results['top1']:.4f} | "
                f"macro_f1 = {test_results['macro_f1']:.4f} | "
                f"samples = {test_results['num_samples']}"
            )
        except Exception as exc:
            print(f"[Warning] test-set evaluation failed: {exc!r}")

    # write metrics.json and best_predictions.npz
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    np.savez(
        output_dir / "best_predictions.npz",
        val_predictions=val_best["predictions"],
        val_labels=val_best["labels"],
        val_logits=val_best["logits"],
        test_predictions=(
            test_results["predictions"]
            if test_results is not None
            else np.empty((0,), dtype=np.int64)
        ),
        test_labels=(
            test_results["labels"]
            if test_results is not None
            else np.empty((0,), dtype=np.int64)
        ),
        test_logits=(
            test_results["logits"]
            if test_results is not None
            else np.empty((0, num_classes), dtype=np.float32)
        ),
        class_names=np.array(data["class_names"], dtype=object),
    )

    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()