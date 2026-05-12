"""Training cost comparison figures.

Pulls trainable parameters, total training time, time per epoch
and best-epoch from every experiment's metrics.json, then plots:

  * Time vs test accuracy scatter (per label budget).
  * Per-epoch time vs trainable params (the "cost per epoch"
    frontier).
  * Convergence speed bar chart: best_epoch per strategy / budget.

Run: python -m scripts.plot_training_cost
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "results" / "figures" / "training_cost"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Run:
    name: str
    label: str
    strategy: str
    budget: str  # "1%", "10%", "100%" or "imbalanced"
    params: int
    train_time: float
    epochs: int
    best_epoch: int
    test_top1: float
    time_per_epoch: float


METHOD_STYLE = {
    "Linear probe":       dict(color="#888888", marker="^"),
    "Full fine-tune":     dict(color="#222222", marker="D"),
    "Gradual unfreeze":   dict(color="#55a868", marker="P"),
    "FC-LoRA":            dict(color="#4c72b0", marker="o"),
    "Layer4-LoRA":        dict(color="#c44e52", marker="s"),
}


# Selected canonical runs for fair comparison across label budgets.
RUN_CATALOG: list[tuple[str, str, str]] = [
    # (experiment_dir, method_label, budget_label)
    ("resnet18_linear_probe_1_seed42",        "Linear probe",     "1%"),
    ("resnet18_linear_probe_10_seed42",       "Linear probe",     "10%"),
    ("resnet18_linear_probe_100_seed42",      "Linear probe",     "100%"),
    ("resnet18_finetune_1_seed42",            "Full fine-tune",   "1%"),
    ("resnet18_finetune_10_seed42",           "Full fine-tune",   "10%"),
    ("resnet18_finetune_100_seed42",          "Full fine-tune",   "100%"),
    ("resnet18_gradual_unfreeze_1_seed42",    "Gradual unfreeze", "1%"),
    ("resnet18_gradual_unfreeze_10_seed42",   "Gradual unfreeze", "10%"),
    ("resnet18_gradual_unfreeze_100_seed42",  "Gradual unfreeze", "100%"),
    ("resnet18_lora_r4_1_seed42",             "FC-LoRA",          "1%"),
    ("resnet18_lora_r8_1_seed42",             "FC-LoRA",          "1%"),
    ("resnet18_lora_r16_1_seed42",            "FC-LoRA",          "1%"),
    ("resnet18_lora_r4_10_seed42",            "FC-LoRA",          "10%"),
    ("resnet18_lora_r8_10_seed42",            "FC-LoRA",          "10%"),
    ("resnet18_lora_r16_10_seed42",           "FC-LoRA",          "10%"),
    ("resnet18_lora_r4_100_seed42",           "FC-LoRA",          "100%"),
    ("resnet18_lora_r8_100_seed42",           "FC-LoRA",          "100%"),
    ("resnet18_lora_r16_100_seed42",          "FC-LoRA",          "100%"),
    ("resnet18_lora_layer4_r4_1_seed42",      "Layer4-LoRA",      "1%"),
    ("resnet18_lora_layer4_r8_1_seed42",      "Layer4-LoRA",      "1%"),
    ("resnet18_lora_layer4_r16_1_seed42",     "Layer4-LoRA",      "1%"),
    ("resnet18_lora_layer4_r4_10_seed42",     "Layer4-LoRA",      "10%"),
    ("resnet18_lora_layer4_r8_10_seed42",     "Layer4-LoRA",      "10%"),
    ("resnet18_lora_layer4_r16_10_seed42",    "Layer4-LoRA",      "10%"),
    ("resnet18_lora_layer4_r4_100_seed42",    "Layer4-LoRA",      "100%"),
    ("resnet18_lora_layer4_r8_100_seed42",    "Layer4-LoRA",      "100%"),
    ("resnet18_lora_layer4_r16_100_seed42",   "Layer4-LoRA",      "100%"),
]


def _rank_from_name(name: str) -> str:
    for r in ("r16", "r8", "r4"):
        if r in name:
            return r
    return ""


def collect() -> list[Run]:
    runs: list[Run] = []
    for exp_name, method, budget in RUN_CATALOG:
        path = EXP_DIR / exp_name / "metrics.json"
        if not path.exists():
            print(f"  skip (missing): {exp_name}")
            continue
        m = json.loads(path.read_text())
        epochs = int(m["epochs_run"])
        time_s = float(m["training_time_seconds"])
        params = int(m["trainable_parameters_final"])
        rank = _rank_from_name(exp_name)
        label = f"{method} {rank}".strip()
        runs.append(Run(
            name=exp_name,
            label=label,
            strategy=method,
            budget=budget,
            params=params,
            train_time=time_s,
            epochs=epochs,
            best_epoch=int(m["best_epoch"]),
            test_top1=float(m["test_top1"]),
            time_per_epoch=time_s / max(epochs, 1),
        ))
    return runs


def fig_time_vs_acc(runs: list[Run]) -> None:
    budgets = ["1%", "10%", "100%"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    for ax, budget in zip(axes, budgets):
        rs = [r for r in runs if r.budget == budget]
        for r in rs:
            st = METHOD_STYLE[r.strategy]
            ax.scatter(r.train_time, r.test_top1, s=85,
                       edgecolor="black", linewidth=0.5, **st)
            tag = (_rank_from_name(r.name) if r.strategy.endswith("LoRA")
                   else r.strategy)
            ax.annotate(tag, (r.train_time, r.test_top1),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color="#444")
        ax.set_title(f"{budget} label budget")
        ax.set_xlabel("Total training time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test top-1 accuracy")

    handles = [
        plt.scatter([], [], **METHOD_STYLE[m], s=85,
                    edgecolor="black", linewidth=0.5, label=m)
        for m in METHOD_STYLE
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(METHOD_STYLE), bbox_to_anchor=(0.5, -0.02),
               fontsize=9)
    fig.suptitle("Training time vs test accuracy", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = OUT_DIR / "time_vs_acc.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")


def fig_perepoch_vs_params(runs: list[Run]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for r in runs:
        if r.budget != "100%":
            continue
        st = METHOD_STYLE[r.strategy]
        lbl = r.strategy if r.strategy not in seen else None
        seen.add(r.strategy)
        ax.scatter(r.params, r.time_per_epoch, s=90, label=lbl,
                   edgecolor="black", linewidth=0.5, **st)
        tag = (_rank_from_name(r.name) if r.strategy.endswith("LoRA")
               else r.strategy)
        ax.annotate(tag, (r.params, r.time_per_epoch),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color="#444")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Time per epoch (s)")
    ax.set_title("Per-epoch training cost vs trainable parameters (100% labels)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "perepoch_vs_params.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")


def fig_convergence(runs: list[Run]) -> None:
    """Best epoch (lower = converges faster) per strategy per budget."""
    budgets = ["1%", "10%", "100%"]
    methods = list(METHOD_STYLE.keys())
    # For each method/budget, average best_epoch across ranks if multiple.
    data: dict[tuple[str, str], float] = {}
    for r in runs:
        key = (r.strategy, r.budget)
        data.setdefault(key, []).append(r.best_epoch)
    means = {k: float(np.mean(v)) for k, v in data.items()}

    x = np.arange(len(budgets))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        ys = [means.get((method, b), np.nan) for b in budgets]
        st = METHOD_STYLE[method]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, ys, width,
               label=method, color=st["color"],
               edgecolor="black", linewidth=0.4)
        for xi, y in zip(x + (i - (len(methods) - 1) / 2) * width, ys):
            if np.isnan(y):
                continue
            ax.text(xi, y + 0.3, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(budgets)
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Best epoch (mean over ranks)")
    ax.set_title("Convergence speed: best validation epoch by strategy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "convergence_best_epoch.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")


def fig_efficiency_summary(runs: list[Run]) -> None:
    """Compact 1x2: cost-per-accuracy bar + Pareto sketch at 100%."""
    rs100 = [r for r in runs if r.budget == "100%"]
    # Cost-per-accuracy = time_per_epoch / test_top1 (lower is better).
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: time/epoch normalised by test top-1 (lower = more efficient).
    rs_sorted = sorted(rs100, key=lambda r: r.time_per_epoch / max(r.test_top1, 1e-3))
    names = [
        (f"{r.strategy} {_rank_from_name(r.name)}".strip()) for r in rs_sorted
    ]
    vals = [r.time_per_epoch / r.test_top1 for r in rs_sorted]
    colors = [METHOD_STYLE[r.strategy]["color"] for r in rs_sorted]
    axes[0].barh(np.arange(len(names)), vals, color=colors,
                 edgecolor="black", linewidth=0.4)
    axes[0].set_yticks(np.arange(len(names))); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Time per epoch / test top-1  (s / accuracy)")
    axes[0].set_title("Cost-per-accuracy (100% labels, lower = better)")
    axes[0].grid(axis="x", alpha=0.3)

    # Right: Pareto sketch — time vs (1 - accuracy).
    for r in rs100:
        st = METHOD_STYLE[r.strategy]
        axes[1].scatter(r.train_time, 1 - r.test_top1, s=85,
                        edgecolor="black", linewidth=0.5, **st)
        tag = (_rank_from_name(r.name) if r.strategy.endswith("LoRA")
               else r.strategy)
        axes[1].annotate(tag, (r.train_time, 1 - r.test_top1),
                         xytext=(5, 4), textcoords="offset points",
                         fontsize=7, color="#444")
    axes[1].set_xlabel("Total training time (s)")
    axes[1].set_ylabel("Test error  (1 − top-1)")
    axes[1].set_title("Error vs total time (100% labels — lower-left is better)")
    axes[1].grid(alpha=0.3)
    handles = [
        plt.scatter([], [], **METHOD_STYLE[m], s=85,
                    edgecolor="black", linewidth=0.5, label=m)
        for m in METHOD_STYLE
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="upper right")

    fig.suptitle("Training efficiency at 100% labels", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "efficiency_100.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")


def main() -> None:
    print(f"Writing figures to {OUT_DIR.relative_to(ROOT)}/")
    runs = collect()
    print(f"  collected {len(runs)} runs")
    fig_time_vs_acc(runs)
    fig_perepoch_vs_params(runs)
    fig_convergence(runs)
    fig_efficiency_summary(runs)
    print("Done.")


if __name__ == "__main__":
    main()
