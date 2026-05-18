"""Scatter plots: trainable parameters vs test accuracy.

Pulls FC-LoRA and Layer4-LoRA results from
results/lora_ablation_summary.csv and the Part-2 baselines from
results/experiment_summary.csv. Produces one figure per label
budget (1% / 10% / 100%) plus a combined figure.

Run: python -m scripts.plot_lora_params_vs_acc
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "lora_params_vs_acc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_CSV = ROOT / "results" / "lora_ablation_summary.csv"
SUMMARY_CSV = ROOT / "results" / "experiment_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_budget(b: str) -> str:
    return b.strip()


def infer_lora_family(targets: str) -> str:
    normalized = "+".join(t.strip() for t in targets.split("+") if t.strip())
    target_set = set(normalized.split("+"))
    if normalized == "fc":
        return "FC-LoRA"
    if normalized == "layer4":
        return "Layer4-LoRA"
    if normalized == "layer3":
        return "Layer3-LoRA"
    if target_set == {"layer3", "layer4"}:
        return "Layer4+3-LoRA"
    return f"LoRA({normalized})"


def collect_lora() -> dict[str, dict[str, list[tuple[int, float, int]]]]:
    """budget -> family -> list of (params, test_top1, rank)."""
    rows = read_csv(LORA_CSV)
    out: dict[str, dict[str, list[tuple[int, float, int]]]] = {}
    for r in rows:
        budget = parse_budget(r["budget"])
        family = infer_lora_family(r["targets"])
        params = int(r["trainable_params"])
        acc = float(r["test_top1"])
        rank = int(r["rank"])
        out.setdefault(budget, {}).setdefault(family, []).append((params, acc, rank))
    for budget in out:
        for fam in out[budget]:
            out[budget][fam].sort(key=lambda t: t[0])
    return out


CANONICAL_BASELINES = {
    "resnet18_linear_probe_1_seed42",
    "resnet18_linear_probe_10_seed42",
    "resnet18_linear_probe_100_seed42",
    "resnet18_finetune_1_seed42",
    "resnet18_finetune_10_seed42",
    "resnet18_finetune_100_seed42",
    "resnet18_gradual_unfreeze_1_seed42",
    "resnet18_gradual_unfreeze_10_seed42",
    "resnet18_gradual_unfreeze_100_seed42",
}


def collect_baselines() -> dict[str, list[tuple[str, int, float]]]:
    """budget -> [(method, params, test_top1)]."""
    rows = read_csv(SUMMARY_CSV)
    out: dict[str, list[tuple[str, int, float]]] = {}
    for r in rows:
        if r["experiment"] not in CANONICAL_BASELINES:
            continue
        method = r["method"]
        budget = parse_budget(r["budget"])
        params = int(r["trainable_params"])
        acc = float(r["test_accuracy"])
        out.setdefault(budget, []).append((method, params, acc))
    return out


STYLE = {
    "FC-LoRA":       dict(color="#4c72b0", marker="o"),
    "Layer4-LoRA":   dict(color="#c44e52", marker="s"),
    "Layer3-LoRA":   dict(color="#8172b3", marker="X"),
    "Layer4+3-LoRA": dict(color="#dd8452", marker="v"),
}

BASELINE_STYLE = {
    "Linear probe":     dict(color="#888888", marker="^", s=110),
    "Full fine-tuning": dict(color="#222222", marker="D", s=110),
    "Gradual unfreezing": dict(color="#55a868", marker="P", s=130),
}


def _plot_one(ax, budget: str, lora_by_fam: dict[str, list[tuple[int, float, int]]],
              baselines: list[tuple[str, int, float]]) -> None:
    for fam, pts in lora_by_fam.items():
        xs = [p for p, _, _ in pts]
        ys = [a for _, a, _ in pts]
        ranks = [r for _, _, r in pts]
        st = STYLE.get(fam, dict(color="#666666", marker="o"))
        ax.plot(xs, ys, linestyle="--", linewidth=1.2, alpha=0.7, color=st["color"])
        ax.scatter(xs, ys, label=fam, s=80, edgecolor="black",
                   linewidth=0.5, **st)
        for x, y, r in zip(xs, ys, ranks):
            ax.annotate(f"r={r}", (x, y), xytext=(6, 4),
                        textcoords="offset points", fontsize=8)

    label_offsets = {
        "Linear probe":       (6, 6),
        "Full fine-tuning":   (-6, -14),
        "Gradual unfreezing": (-6, 8),
    }
    label_ha = {
        "Linear probe":       "left",
        "Full fine-tuning":   "right",
        "Gradual unfreezing": "right",
    }
    for method, params, acc in baselines:
        st = BASELINE_STYLE[method]
        ax.scatter([params], [acc], label=method,
                   edgecolor="black", linewidth=0.6, **st)
        ax.annotate(method, (params, acc),
                    xytext=label_offsets[method],
                    textcoords="offset points",
                    fontsize=7, color="#444",
                    ha=label_ha[method])

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test top-1 accuracy")
    ax.set_title(f"{budget} label budget")
    ax.grid(alpha=0.3, which="both")
    # Deduplicate the legend.
    handles, labels = ax.get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h); uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, fontsize=8, loc="lower right")


def main() -> None:
    print(f"Writing figures to {OUT_DIR.relative_to(ROOT)}/")
    lora = collect_lora()
    baselines = collect_baselines()

    budgets = ["1%", "10%", "100%"]

    # One figure per budget.
    for budget in budgets:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        _plot_one(ax, budget, lora.get(budget, {}), baselines.get(budget, []))
        fig.suptitle("Trainable parameters vs test accuracy", fontsize=12)
        fig.tight_layout()
        out_path = OUT_DIR / f"params_vs_acc_{budget.replace('%','pct')}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"  saved {out_path.relative_to(ROOT)}")

    # Combined 1x3.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, budget in zip(axes, budgets):
        _plot_one(ax, budget, lora.get(budget, {}), baselines.get(budget, []))
    fig.suptitle("Trainable parameters vs test accuracy — all label budgets",
                 fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / "params_vs_acc_all.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
