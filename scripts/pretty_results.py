from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("experiments")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_budget(name: str) -> str:
    if "_100" in name:
        return "100%"
    if "_10" in name:
        return "10%"
    if "_1" in name:
        return "1%"
    return "-"


def infer_rank(name: str) -> str:
    """
    Correctly infer LoRA rank from experiment names.

    Works for:
    - resnet18_lora_r4_100_seed42
    - resnet18_lora_r8_100_seed42
    - resnet18_lora_r16_100_seed42
    - resnet18_lora_layer4_r4_100_seed42
    - resnet18_lora_layer4_r8_100_seed42
    - resnet18_lora_layer4_r16_100_seed42

    Important:
    We use regex "_r(4|8|16)_" so that "layer4" is not mistaken as rank 4.
    """
    match = re.search(r"_r(4|8|16)(?:_|$)", name)
    if match:
        return match.group(1)
    return "-"


def infer_target(name: str) -> str:
    if "lora_layer4" in name:
        return "layer4 + fc"
    if "lora" in name:
        return "fc only"
    if "linear_probe" in name:
        return "fc only"
    if "gradual" in name:
        return "gradual"
    if "partial" in name:
        return "partial"
    if "finetune" in name:
        return "full"
    return "-"


def infer_method(name: str) -> str:
    if "lora_layer4" in name:
        return "Layer4-LoRA"
    if "lora" in name:
        return "FC-LoRA"
    if "linear_probe" in name:
        return "Linear probe"
    if "gradual" in name:
        return "Gradual unfreezing"
    if "partial_finetune" in name:
        return "Partial fine-tuning"
    if "finetune" in name:
        return "Full fine-tuning"
    if "sanity_binary" in name:
        return "Binary sanity check"
    return "-"


def fmt_float(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for metrics_path in sorted(ROOT.glob("*/metrics.json")):
        exp_dir = metrics_path.parent
        name = exp_dir.name
        metrics = load_json(metrics_path)

        rows.append(
            {
                "experiment": name,
                "method": infer_method(name),
                "target": infer_target(name),
                "budget": infer_budget(name),
                "rank": infer_rank(name),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_accuracy": metrics.get("best_val_top1"),
                "test_accuracy": metrics.get("test_top1"),
                "test_macro_f1": metrics.get("test_macro_f1"),
                "trainable_params": metrics.get("trainable_parameters_final"),
                "time_seconds": metrics.get("training_time_seconds"),
            }
        )

    return rows


def sort_key(row: dict[str, Any]) -> tuple:
    method_order = {
        "FC-LoRA": 0,
        "Layer4-LoRA": 1,
        "Linear probe": 2,
        "Full fine-tuning": 3,
        "Gradual unfreezing": 4,
        "Partial fine-tuning": 5,
        "Binary sanity check": 6,
    }

    budget_order = {
        "1%": 0,
        "10%": 1,
        "100%": 2,
        "-": 3,
    }

    rank = row.get("rank", "-")
    try:
        rank_value = int(rank)
    except Exception:
        rank_value = 999

    return (
        method_order.get(row.get("method", "-"), 99),
        budget_order.get(row.get("budget", "-"), 99),
        rank_value,
        row.get("experiment", ""),
    )


def write_csv(rows: list[dict[str, Any]]) -> None:
    out_path = OUT_DIR / "experiment_summary.csv"
    fieldnames = [
        "experiment",
        "method",
        "target",
        "budget",
        "rank",
        "best_epoch",
        "best_val_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "trainable_params",
        "time_seconds",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {out_path}")


def write_markdown(rows: list[dict[str, Any]]) -> None:
    out_path = OUT_DIR / "experiment_summary.md"

    header = [
        "Experiment",
        "Method",
        "Target",
        "Budget",
        "Rank",
        "Best epoch",
        "Best val accuracy",
        "Test accuracy",
        "Test macro F1",
        "Trainable params",
        "Time",
    ]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r['experiment']}`",
                    str(r["method"]),
                    str(r["target"]),
                    str(r["budget"]),
                    str(r["rank"]),
                    str(r["best_epoch"]),
                    fmt_float(r["best_val_accuracy"]),
                    fmt_float(r["test_accuracy"]),
                    fmt_float(r["test_macro_f1"]),
                    fmt_int(r["trainable_params"]),
                    f"{fmt_float(r['time_seconds'])}s",
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path}")


def print_console_table(rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 140)
    print("Experiment Summary")
    print("=" * 140)
    print(
        f"{'Experiment':45s} "
        f"{'Method':20s} "
        f"{'Target':12s} "
        f"{'Budget':>6s} "
        f"{'Rank':>4s} "
        f"{'Val Acc':>8s} "
        f"{'Test Acc':>8s} "
        f"{'Macro F1':>8s} "
        f"{'Params':>12s} "
        f"{'Time':>10s}"
    )
    print("-" * 140)

    for r in rows:
        print(
            f"{r['experiment'][:45]:45s} "
            f"{str(r['method'])[:20]:20s} "
            f"{str(r['target'])[:12]:12s} "
            f"{str(r['budget']):>6s} "
            f"{str(r['rank']):>4s} "
            f"{fmt_float(r['best_val_accuracy']):>8s} "
            f"{fmt_float(r['test_accuracy']):>8s} "
            f"{fmt_float(r['test_macro_f1']):>8s} "
            f"{fmt_int(r['trainable_params']):>12s} "
            f"{fmt_float(r['time_seconds']):>9s}s"
        )

    print("=" * 140)
    print()


def write_report_ready_tables(rows: list[dict[str, Any]]) -> None:
    """
    Generate smaller report-ready tables:
    1. fc_lora_ablation.md
    2. layer4_lora_ablation.md
    3. best_lora_vs_baselines.md
    """

    fc_rows = [r for r in rows if r["method"] == "FC-LoRA"]
    layer4_rows = [r for r in rows if r["method"] == "Layer4-LoRA"]

    def write_small_table(path: Path, selected_rows: list[dict[str, Any]]) -> None:
        header = [
            "Method",
            "Target",
            "Budget",
            "Rank",
            "Trainable params",
            "Val accuracy",
            "Test accuracy",
            "Test macro F1",
        ]

        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for r in selected_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["method"]),
                        str(r["target"]),
                        str(r["budget"]),
                        str(r["rank"]),
                        fmt_int(r["trainable_params"]),
                        fmt_float(r["best_val_accuracy"]),
                        fmt_float(r["test_accuracy"]),
                        fmt_float(r["test_macro_f1"]),
                    ]
                )
                + " |"
            )

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[saved] {path}")

    write_small_table(OUT_DIR / "fc_lora_ablation.md", fc_rows)
    write_small_table(OUT_DIR / "layer4_lora_ablation.md", layer4_rows)

    # Best Layer4-LoRA r=4 compared with important baselines.
    selected_names = {
        "resnet18_linear_probe_1_seed42",
        "resnet18_linear_probe_10_seed42",
        "resnet18_linear_probe_100_seed42",
        "resnet18_finetune_1_seed42",
        "resnet18_finetune_10_seed42",
        "resnet18_finetune_100_seed42",
        "resnet18_gradual_unfreeze_1_seed42",
        "resnet18_gradual_unfreeze_10_seed42",
        "resnet18_gradual_unfreeze_100_seed42",
        "resnet18_lora_layer4_r4_1_seed42",
        "resnet18_lora_layer4_r4_10_seed42",
        "resnet18_lora_layer4_r4_100_seed42",
    }

    baseline_rows = [r for r in rows if r["experiment"] in selected_names]
    baseline_rows = sorted(
        baseline_rows,
        key=lambda r: (
            {"1%": 0, "10%": 1, "100%": 2}.get(r["budget"], 99),
            {
                "Linear probe": 0,
                "Full fine-tuning": 1,
                "Gradual unfreezing": 2,
                "Layer4-LoRA": 3,
            }.get(r["method"], 99),
        ),
    )

    write_small_table(OUT_DIR / "best_lora_vs_baselines.md", baseline_rows)


def main() -> None:
    rows = collect_rows()
    rows = sorted(rows, key=sort_key)

    print_console_table(rows)
    write_csv(rows)
    write_markdown(rows)
    write_report_ready_tables(rows)


if __name__ == "__main__":
    main()