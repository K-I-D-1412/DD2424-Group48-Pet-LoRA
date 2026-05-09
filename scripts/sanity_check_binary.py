"""
§1.1 Binary sanity check: cat vs dog, linear probing expected test acc ≥ 99%。

Usage:
    python -m scripts.sanity_check_binary
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import torch
from torch import nn

from src.binary_data import get_binary_dataloaders
from src.engine import evaluate, train_one_epoch
from src.models import build_resnet18, freeze_backbone_for_linear_probe
from src.utils import count_parameters, get_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sanity_binary_cat_vs_dog_seed42"


def main():
    set_seed(42)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = get_binary_dataloaders(
        image_size=224, batch_size=64, num_workers=2,
        use_augmentation=False, download=False,
    )

    model = build_resnet18(num_classes=2, pretrained=True)
    freeze_backbone_for_linear_probe(model)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4,
    )

    print("=" * 70)
    print("Binary sanity check: cat vs dog  (linear probing, pretrained ResNet-18)")
    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Train samples    : {len(data['train_dataset'])}")
    print(f"Test  samples    : {len(data['test_dataset'])}")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}  (expect 1,026 = 512x2 + 2)")
    print("=" * 70)

    epochs = 10
    history = []
    start = time.time()

    for epoch in range(epochs):
        m = train_one_epoch(model, data["train_loader"], loss_fn, optimizer, device)
        t = evaluate(
            model, data["test_loader"], loss_fn, device,
            num_classes=2, return_predictions=False,
        )
        record = {
            "epoch": epoch,
            "train_loss": round(m["loss"], 4),
            "train_top1": round(m["top1"], 4),
            "test_top1": round(t["top1"], 4),
            "test_macro_f1": round(t["macro_f1"], 4),
        }
        history.append(record)
        print(
            f"epoch {epoch} | train_loss {m['loss']:.4f} | train_top1 {m['top1']:.4f} | "
            f"test_top1 {t['top1']:.4f} | test_macro_f1 {t['macro_f1']:.4f}"
        )

    total_time = time.time() - start

    # Final test eval
    final = evaluate(
        model, data["test_loader"], loss_fn, device,
        num_classes=2, return_predictions=True,
    )

    passed = final["top1"] >= 0.99
    summary = {
        "experiment": "sanity_binary_cat_vs_dog",
        "epochs": epochs,
        "test_top1": float(final["top1"]),
        "test_macro_f1": float(final["macro_f1"]),
        "trainable_params": trainable_params,
        "training_time_seconds": round(total_time, 1),
    }

    with (OUTPUT_DIR / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Final test_top1   = {final['top1']:.4f}")
    print(f"Final test_macro_f1 = {final['macro_f1']:.4f}")
    print(f"Trainable params  = {trainable_params:,}  (fc only)")
    print(f"Results saved: {OUTPUT_DIR / 'metrics.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()