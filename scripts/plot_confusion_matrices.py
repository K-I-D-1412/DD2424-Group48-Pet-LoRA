"""Plot confusion matrix heatmaps from experiments/*/metrics.json.

Row-normalised heatmaps for the test split. Cat rows/columns are
highlighted on the imbalance experiments so the minority-class
behaviour is visible at a glance.

Run: python -m scripts.plot_confusion_matrices
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "results" / "figures" / "confusion_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}


def load_metrics(name: str) -> dict:
    return json.loads((EXP_DIR / name / "metrics.json").read_text())


def plot_cm(ax, cm: np.ndarray, class_names: list[str], title: str,
            highlight_cats: bool) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                        where=row_sums > 0)

    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)

    if highlight_cats:
        cat_idx = [i for i, c in enumerate(class_names) if c in CAT_BREEDS]
        for i in cat_idx:
            ax.get_xticklabels()[i].set_color("crimson")
            ax.get_yticklabels()[i].set_color("crimson")
        # Translucent overlays on cat rows.
        for i in cat_idx:
            ax.axhspan(i - 0.5, i + 0.5, color="crimson", alpha=0.08, zorder=0)

    return im


def plot_single(name: str, title: str, out_name: str,
                highlight_cats: bool = False) -> None:
    m = load_metrics(name)
    cm = np.asarray(m["test_confusion_matrix"], dtype=float)
    class_names = m["class_names"]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = plot_cm(ax, cm, class_names, title, highlight_cats)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised count")
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def plot_grid(runs: list[tuple[str, str]], out_name: str, title: str,
              highlight_cats: bool = False) -> None:
    n = len(runs)
    cols = 2 if n <= 2 else 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11 * cols, 9.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for ax, (run_name, sub_title) in zip(axes, runs):
        m = load_metrics(run_name)
        cm = np.asarray(m["test_confusion_matrix"], dtype=float)
        last_im = plot_cm(ax, cm, m["class_names"], sub_title, highlight_cats)

    for ax in axes[len(runs):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    if last_im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Row-normalised count")
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def main() -> None:
    print(f"Writing figures to {OUT_DIR.relative_to(ROOT)}/")

    # Imbalance grid (cats highlighted)
    plot_grid(
        runs=[
            ("resnet18_finetune_100_seed42",
             "Balanced reference  (test top-1 0.868)"),
            ("resnet18_finetune_imbalanced_baseline_seed42",
             "Imbalanced, no compensation  (0.830)"),
            ("resnet18_finetune_imbalanced_cat20_weighted_ce_seed42",
             "Imbalanced + weighted CE  (0.847)"),
            ("resnet18_finetune_imbalanced_cat20_oversampling_seed42",
             "Imbalanced + oversampling  (0.841)"),
        ],
        out_name="imbalance_grid.png",
        title="Confusion matrices: class imbalance (cat rows highlighted in red)",
        highlight_cats=True,
    )

    # Best models per strategy at 100% data
    plot_grid(
        runs=[
            ("resnet18_linear_probe_100_seed42",
             "Linear probe  (0.846)"),
            ("resnet18_finetune_100_seed42",
             "Full fine-tune  (0.868)"),
            ("resnet18_gradual_unfreeze_100_seed42",
             "Gradual unfreeze  (0.874)"),
            ("resnet18_lora_layer4_r4_100_seed42",
             "Layer4-LoRA r=4  (0.850)"),
        ],
        out_name="strategies_100_grid.png",
        title="Confusion matrices: strategies at 100% labels",
    )

    # Single-figure imbalance closeups
    plot_single(
        "resnet18_finetune_imbalanced_baseline_seed42",
        "Imbalanced (no compensation), test top-1 0.830",
        "imbalance_baseline.png",
        highlight_cats=True,
    )
    plot_single(
        "resnet18_finetune_imbalanced_cat20_weighted_ce_seed42",
        "Imbalanced + weighted CE, test top-1 0.847",
        "imbalance_weighted_ce.png",
        highlight_cats=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
