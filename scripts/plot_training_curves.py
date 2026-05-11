"""Plot training curves from experiments/*/training_curves.csv.

Generates two figures per group: train/val loss and val top-1.
Groups are: baselines @ each label budget, gradual unfreezing,
LoRA layer4 rank ablation, and the imbalance experiments.

Run: python -m scripts.plot_training_curves
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "results" / "figures" / "training_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_curve(name: str) -> dict[str, list[float]]:
    path = EXP_DIR / name / "training_curves.csv"
    cols: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


GROUPS: dict[str, list[tuple[str, str]]] = {
    "baselines_100": [
        ("resnet18_linear_probe_100_seed42", "Linear probe"),
        ("resnet18_finetune_100_seed42", "Full fine-tune"),
        ("resnet18_gradual_unfreeze_100_seed42", "Gradual unfreeze"),
        ("resnet18_lora_layer4_r4_100_seed42", "Layer4-LoRA r=4"),
    ],
    "baselines_10": [
        ("resnet18_linear_probe_10_seed42", "Linear probe"),
        ("resnet18_finetune_10_seed42", "Full fine-tune"),
        ("resnet18_gradual_unfreeze_10_seed42", "Gradual unfreeze"),
        ("resnet18_lora_layer4_r4_10_seed42", "Layer4-LoRA r=4"),
    ],
    "baselines_1": [
        ("resnet18_linear_probe_1_seed42", "Linear probe"),
        ("resnet18_finetune_1_seed42", "Full fine-tune"),
        ("resnet18_gradual_unfreeze_1_seed42", "Gradual unfreeze"),
        ("resnet18_lora_layer4_r4_1_seed42", "Layer4-LoRA r=4"),
    ],
    "lora_layer4_rank_100": [
        ("resnet18_lora_layer4_r4_100_seed42", "r=4"),
        ("resnet18_lora_layer4_r8_100_seed42", "r=8"),
        ("resnet18_lora_layer4_r16_100_seed42", "r=16"),
    ],
    "lora_fc_rank_100": [
        ("resnet18_lora_r4_100_seed42", "r=4"),
        ("resnet18_lora_r8_100_seed42", "r=8"),
        ("resnet18_lora_r16_100_seed42", "r=16"),
    ],
    "imbalance": [
        ("resnet18_finetune_imbalanced_baseline_seed42", "No compensation"),
        ("resnet18_finetune_imbalanced_cat20_weighted_ce_seed42", "Weighted CE"),
        ("resnet18_finetune_imbalanced_cat20_oversampling_seed42", "Oversampling"),
        ("resnet18_finetune_100_seed42", "Balanced reference"),
    ],
    "augmentation_ablation": [
        ("resnet18_finetune_10_seed42", "10% + aug"),
        ("resnet18_finetune_10_noaug_seed42", "10% no aug"),
        ("resnet18_finetune_1_seed42", "1% + aug"),
        ("resnet18_finetune_1_noaug_seed42", "1% no aug"),
    ],
}


PRETTY_TITLE = {
    "baselines_100": "Strategies @ 100% labels",
    "baselines_10": "Strategies @ 10% labels",
    "baselines_1": "Strategies @ 1% labels",
    "lora_layer4_rank_100": "Layer4-LoRA rank ablation (100% labels)",
    "lora_fc_rank_100": "FC-LoRA rank ablation (100% labels)",
    "imbalance": "Class imbalance experiments",
    "augmentation_ablation": "Augmentation ablation",
}


def plot_group(group_name: str, runs: list[tuple[str, str]]) -> None:
    title = PRETTY_TITLE.get(group_name, group_name)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax_loss, ax_top1, ax_f1 = axes

    for run_dir, label in runs:
        path = EXP_DIR / run_dir / "training_curves.csv"
        if not path.exists():
            print(f"  skip (missing): {run_dir}")
            continue
        d = load_curve(run_dir)
        epochs = d["epoch"]
        ax_loss.plot(epochs, d["train_loss"], linestyle="--", alpha=0.55)
        ax_loss.plot(epochs, d["val_loss"], label=label)
        ax_top1.plot(epochs, d["val_top1"], label=label, marker="o", markersize=3)
        ax_f1.plot(epochs, d["val_macro_f1"], label=label, marker="o", markersize=3)

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss  (dashed = train, solid = val)")
    ax_loss.grid(alpha=0.3); ax_loss.legend(fontsize=8)

    ax_top1.set_xlabel("Epoch"); ax_top1.set_ylabel("Val top-1")
    ax_top1.set_title("Validation top-1 accuracy")
    ax_top1.grid(alpha=0.3); ax_top1.legend(fontsize=8)

    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Val macro F1")
    ax_f1.set_title("Validation macro F1")
    ax_f1.grid(alpha=0.3); ax_f1.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / f"{group_name}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def main() -> None:
    print(f"Writing figures to {OUT_DIR.relative_to(ROOT)}/")
    for group_name, runs in GROUPS.items():
        print(f"Plotting {group_name} ...")
        plot_group(group_name, runs)
    print("Done.")


if __name__ == "__main__":
    main()
