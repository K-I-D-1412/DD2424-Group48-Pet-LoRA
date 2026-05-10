"""Collect LoRA experiment metrics into one CSV.

Run from project root after experiments finish:
    python -m scripts.collect_lora_results
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUT_PATH = PROJECT_ROOT / "results" / "lora_ablation_summary.csv"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    rows = []
    for metrics_path in sorted(EXPERIMENTS_DIR.glob("*lora*/metrics.json")):
        exp_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        cfg_path = exp_dir / "config.yaml"
        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        lora = cfg.get("lora", metrics.get("lora", {})) or {}
        data = cfg.get("data", {})
        train_split = data.get("train_split", metrics.get("train_split", ""))
        if "train_100" in train_split:
            budget = "100%"
        elif "train_10" in train_split:
            budget = "10%"
        elif "train_1" in train_split:
            budget = "1%"
        else:
            budget = train_split

        rows.append(
            {
                "experiment": metrics.get("experiment_name", exp_dir.name),
                "budget": budget,
                "targets": "+".join(lora.get("target_modules", [])),
                "rank": lora.get("rank", ""),
                "alpha": lora.get("alpha", ""),
                "dropout": lora.get("dropout", ""),
                "train_classifier": lora.get("train_classifier", False),
                "trainable_params": metrics.get("trainable_parameters_final", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "best_val_top1": metrics.get("best_val_top1", ""),
                "test_top1": metrics.get("test_top1", ""),
                "test_macro_f1": metrics.get("test_macro_f1", ""),
                "training_time_seconds": metrics.get("training_time_seconds", ""),
            }
        )

    if not rows:
        raise FileNotFoundError("No LoRA metrics found under experiments/*lora*/metrics.json")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
