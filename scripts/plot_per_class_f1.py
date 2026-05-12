"""Per-class F1 bar charts for the class-imbalance experiments.

Cat breeds (minority in imbalanced splits) are coloured red, dog
breeds blue. Grouped bars compare balanced / no-compensation /
weighted-CE / oversampling. A separate figure shows the
mean-F1-by-species (cat vs dog) summary.

Run: python -m scripts.plot_per_class_f1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "results" / "figures" / "per_class_f1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}

RUNS = [
    ("resnet18_finetune_100_seed42",                          "Balanced ref."),
    ("resnet18_finetune_imbalanced_baseline_seed42",          "Imbalanced (none)"),
    ("resnet18_finetune_imbalanced_cat20_weighted_ce_seed42", "Weighted CE"),
    ("resnet18_finetune_imbalanced_cat20_oversampling_seed42","Oversampling"),
]

BAR_COLORS = ["#4c72b0", "#c44e52", "#dd8452", "#55a868"]


def load(name: str) -> dict:
    return json.loads((EXP_DIR / name / "metrics.json").read_text())


def grouped_bar_chart() -> None:
    base = load(RUNS[0][0])
    class_names = base["class_names"]
    # Sort: cats first (in original order), then dogs.
    order = (
        [i for i, c in enumerate(class_names) if c in CAT_BREEDS]
        + [i for i, c in enumerate(class_names) if c not in CAT_BREEDS]
    )
    sorted_names = [class_names[i] for i in order]
    n_cats = sum(c in CAT_BREEDS for c in class_names)

    f1s = []
    for run, _label in RUNS:
        m = load(run)
        arr = np.asarray(m["per_class_test_f1"])
        f1s.append(arr[order])

    x = np.arange(len(sorted_names))
    width = 0.21

    fig, ax = plt.subplots(figsize=(17, 6))
    for k, ((_run, label), arr, color) in enumerate(zip(RUNS, f1s, BAR_COLORS)):
        ax.bar(x + (k - 1.5) * width, arr, width=width,
               label=label, color=color, edgecolor="black", linewidth=0.3)

    # Visually mark the cat region.
    ax.axvspan(-0.5, n_cats - 0.5, color="crimson", alpha=0.06, zorder=0)
    ax.text(
        (n_cats - 1) / 2, 1.02,
        f"Cat breeds ({n_cats})  ← reduced to 20%",
        color="crimson", ha="center", va="bottom", fontsize=10,
    )
    ax.text(
        n_cats + (len(sorted_names) - n_cats) / 2 - 0.5, 1.02,
        f"Dog breeds ({len(sorted_names) - n_cats})  unchanged",
        color="#1f4e79", ha="center", va="bottom", fontsize=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=75, fontsize=8, ha="right")
    for i, name in enumerate(sorted_names):
        if name in CAT_BREEDS:
            ax.get_xticklabels()[i].set_color("crimson")
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Per-class test F1")
    ax.set_title("Per-class test F1 — class imbalance experiments (cats at 20%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    out_path = OUT_DIR / "imbalance_per_class_f1.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def species_summary() -> None:
    base = load(RUNS[0][0])
    class_names = base["class_names"]
    is_cat = np.array([c in CAT_BREEDS for c in class_names])

    cat_means, dog_means, overall_means = [], [], []
    for run, _ in RUNS:
        m = load(run)
        arr = np.asarray(m["per_class_test_f1"])
        cat_means.append(arr[is_cat].mean())
        dog_means.append(arr[~is_cat].mean())
        overall_means.append(arr.mean())

    labels = [lab for _, lab in RUNS]
    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, cat_means, width, label="Cats (mean F1)",
           color="#c44e52", edgecolor="black", linewidth=0.3)
    ax.bar(x, dog_means, width, label="Dogs (mean F1)",
           color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax.bar(x + width, overall_means, width, label="Macro F1 (all 37)",
           color="#888888", edgecolor="black", linewidth=0.3)

    for xi, (cm_, dm_, om_) in enumerate(zip(cat_means, dog_means, overall_means)):
        for dx, v in zip([-width, 0, width], [cm_, dm_, om_]):
            ax.text(xi + dx, v + 0.005, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Mean F1")
    ax.set_title("Cat vs dog mean test F1 across imbalance strategies")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "imbalance_species_summary.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def delta_chart() -> None:
    """Show per-class F1 *change* from no-compensation baseline."""
    base = load(RUNS[0][0])
    class_names = base["class_names"]
    order = (
        [i for i, c in enumerate(class_names) if c in CAT_BREEDS]
        + [i for i, c in enumerate(class_names) if c not in CAT_BREEDS]
    )
    sorted_names = [class_names[i] for i in order]
    n_cats = sum(c in CAT_BREEDS for c in class_names)

    f1_none = np.asarray(load("resnet18_finetune_imbalanced_baseline_seed42")
                         ["per_class_test_f1"])[order]
    f1_wce  = np.asarray(load("resnet18_finetune_imbalanced_cat20_weighted_ce_seed42")
                         ["per_class_test_f1"])[order]
    f1_os   = np.asarray(load("resnet18_finetune_imbalanced_cat20_oversampling_seed42")
                         ["per_class_test_f1"])[order]

    fig, ax = plt.subplots(figsize=(17, 5))
    x = np.arange(len(sorted_names))
    width = 0.4

    ax.bar(x - width / 2, f1_wce - f1_none, width,
           label="Weighted CE − none", color="#dd8452",
           edgecolor="black", linewidth=0.3)
    ax.bar(x + width / 2, f1_os - f1_none, width,
           label="Oversampling − none", color="#55a868",
           edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(-0.5, n_cats - 0.5, color="crimson", alpha=0.06, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=75, fontsize=8, ha="right")
    for i, name in enumerate(sorted_names):
        if name in CAT_BREEDS:
            ax.get_xticklabels()[i].set_color("crimson")
    ax.set_ylabel("ΔF1 vs imbalanced baseline")
    ax.set_title("Effect of compensation per class  (positive = compensation helped)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "imbalance_per_class_delta.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def main() -> None:
    print(f"Writing figures to {OUT_DIR.relative_to(ROOT)}/")
    grouped_bar_chart()
    species_summary()
    delta_chart()
    print("Done.")


if __name__ == "__main__":
    main()
