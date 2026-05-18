from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("experiments")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


PAIRS = [
    ("1%", "resnet18_lora_layer4_r4_1_seed42", "resnet18_lora_layer4_r4_lr1e4_1_seed42"),
    ("10%", "resnet18_lora_layer4_r4_10_seed42", "resnet18_lora_layer4_r4_lr1e4_10_seed42"),
    ("100%", "resnet18_lora_layer4_r4_100_seed42", "resnet18_lora_layer4_r4_lr1e4_100_seed42"),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_pct(x: Any) -> str:
    if x is None or x == "":
        return "-"
    return f"{float(x) * 100:.2f}"


def fmt_int(x: Any) -> str:
    if x is None or x == "":
        return "-"
    return f"{int(x):,}"


def read_exp(exp_name: str) -> dict[str, Any]:
    exp_dir = ROOT / exp_name
    metrics_path = exp_dir / "metrics.json"
    config_path = exp_dir / "config.yaml"

    if not metrics_path.exists():
        return {"missing": True, "experiment": exp_name}

    metrics = load_json(metrics_path)
    cfg = load_yaml(config_path) if config_path.exists() else {}

    return {
        "missing": False,
        "experiment": exp_name,
        "lr": cfg.get("optimizer", {}).get("lr", cfg.get("training", {}).get("lr", "")),
        "rank": cfg.get("lora", {}).get("rank", ""),
        "alpha": cfg.get("lora", {}).get("alpha", ""),
        "trainable_params": metrics.get("trainable_parameters_final", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_val_top1": metrics.get("best_val_top1", ""),
        "test_top1": metrics.get("test_top1", ""),
        "test_macro_f1": metrics.get("test_macro_f1", ""),
        "time": metrics.get("training_time_seconds", ""),
    }


def main() -> None:
    rows = []

    for budget, old_exp, new_exp in PAIRS:
        old = read_exp(old_exp)
        new = read_exp(new_exp)

        for label, exp in [("Layer4-LoRA r=4, lr=1e-3", old), ("Layer4-LoRA r=4, lr=1e-4", new)]:
            if exp["missing"]:
                rows.append([budget, label, "-", "-", "-", "-", "-", "-", "MISSING"])
            else:
                rows.append([
                    budget,
                    label,
                    fmt_int(exp["trainable_params"]),
                    str(exp["best_epoch"]),
                    fmt_pct(exp["best_val_top1"]),
                    fmt_pct(exp["test_top1"]),
                    fmt_pct(exp["test_macro_f1"]),
                    f"{float(exp['time']):.1f}s" if exp["time"] not in ("", None) else "-",
                    exp["experiment"],
                ])

    header = [
        "Budget",
        "Method",
        "Trainable params",
        "Best epoch",
        "Best val acc (%)",
        "Test acc (%)",
        "Test macro F1 (%)",
        "Time",
        "Experiment",
    ]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    out_path = OUT_DIR / "layer4_lora_lr_ablation.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()