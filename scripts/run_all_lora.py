from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


FC_LORA_CONFIGS = [
    "configs/lora_r4_100.yaml",
    "configs/lora_r8_100.yaml",
    "configs/lora_r16_100.yaml",
    "configs/lora_r4_10.yaml",
    "configs/lora_r8_10.yaml",
    "configs/lora_r16_10.yaml",
    "configs/lora_r4_1.yaml",
    "configs/lora_r8_1.yaml",
    "configs/lora_r16_1.yaml",
]

LAYER4_LORA_CONFIGS = [
    "configs/lora_layer4_r4_100.yaml",
    "configs/lora_layer4_r8_100.yaml",
    "configs/lora_layer4_r16_100.yaml",
    "configs/lora_layer4_r4_10.yaml",
    "configs/lora_layer4_r8_10.yaml",
    "configs/lora_layer4_r16_10.yaml",
    "configs/lora_layer4_r4_1.yaml",
    "configs/lora_layer4_r8_1.yaml",
    "configs/lora_layer4_r16_1.yaml",
]

PLACEMENT_LORA_CONFIGS = [
    # Minimal A-level placement ablation at the best rank r=4.
    # layer3-only answers "is layer4 special?";
    # layer4+layer3 answers "does adding more residual stages help?".
    "configs/lora_layer3_r4_100.yaml",
    "configs/lora_layer3_r4_10.yaml",
    "configs/lora_layer3_r4_1.yaml",
    "configs/lora_layer4_layer3_r4_100.yaml",
    "configs/lora_layer4_layer3_r4_10.yaml",
    "configs/lora_layer4_layer3_r4_1.yaml",
]

LR_SENSITIVITY_CONFIGS = [
    # Sensitivity check for the main confound: LoRA uses lr=1e-3 while
    # full fine-tuning uses lr=1e-4. Default to low-label regimes because
    # this is where Layer4-LoRA is claimed to outperform full fine-tuning.
    "configs/lora_layer4_r4_lr1e4_10.yaml",
    "configs/lora_layer4_r4_lr1e4_1.yaml",
]

LR_SENSITIVITY_ALL_CONFIGS = [
    "configs/lora_layer4_r4_lr1e4_100.yaml",
    *LR_SENSITIVITY_CONFIGS,
]


def infer_experiment_dir_from_config(config_path: str) -> Path:
    """
    Infer output folder from config filename.

    Example:
    configs/lora_r8_100.yaml
    -> experiments/resnet18_lora_r8_100_seed42

    configs/lora_layer4_r8_100.yaml
    -> experiments/resnet18_lora_layer4_r8_100_seed42
    """
    stem = Path(config_path).stem
    return Path("experiments") / f"resnet18_{stem}_seed42"


def is_completed(config_path: str) -> bool:
    exp_dir = infer_experiment_dir_from_config(config_path)
    metrics_path = exp_dir / "metrics.json"

    if not metrics_path.exists():
        return False

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        required_keys = ["test_top1", "test_macro_f1", "best_val_top1"]
        return all(key in metrics for key in required_keys)

    except Exception:
        return False


def run_command(command: list[str], log_file: Path) -> int:
    print("\n" + "=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)

    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("Running: " + " ".join(command) + "\n")
        f.write("=" * 100 + "\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        process.wait()
        return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["fc", "layer4", "placement", "lr_sensitivity", "lr_sensitivity_all", "all"],
        default="all",
        help="Which LoRA experiments to run. This only means LoRA experiments, not baselines.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later configs even if one experiment fails.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun even if metrics.json already exists.",
    )
    parser.add_argument(
        "--no-checks",
        action="store_true",
        help="Skip check_lora and check_configs.",
    )
    args = parser.parse_args()

    if args.suite == "fc":
        configs = FC_LORA_CONFIGS
    elif args.suite == "layer4":
        configs = LAYER4_LORA_CONFIGS
    elif args.suite == "placement":
        configs = PLACEMENT_LORA_CONFIGS
    elif args.suite == "lr_sensitivity":
        configs = LR_SENSITIVITY_CONFIGS
    elif args.suite == "lr_sensitivity_all":
        configs = LR_SENSITIVITY_ALL_CONFIGS
    else:
        configs = (
            FC_LORA_CONFIGS
            + LAYER4_LORA_CONFIGS
            + PLACEMENT_LORA_CONFIGS
            + LR_SENSITIVITY_CONFIGS
        )

    missing = [cfg for cfg in configs if not Path(cfg).exists()]
    if missing:
        print("Missing config files:")
        for cfg in missing:
            print("  -", cfg)
        sys.exit(1)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "run_lora_remaining.log"

    start_time = time.time()

    if not args.no_checks:
        for command in [
            [sys.executable, "-m", "scripts.check_lora"],
            [sys.executable, "-m", "scripts.check_configs"],
        ]:
            code = run_command(command, log_file)
            if code != 0:
                print(f"Check failed with exit code {code}. Stop.")
                sys.exit(code)

    failed: list[str] = []
    skipped: list[str] = []
    completed: list[str] = []

    for cfg in configs:
        exp_dir = infer_experiment_dir_from_config(cfg)

        if is_completed(cfg) and not args.rerun:
            print("\n" + "-" * 100)
            print(f"[SKIP] Already completed: {cfg}")
            print(f"       Existing output: {exp_dir}")
            print("-" * 100)
            skipped.append(cfg)
            continue

        command = [sys.executable, "-m", "src.train", "--config", cfg]
        code = run_command(command, log_file)

        if code == 0:
            completed.append(cfg)
        else:
            failed.append(cfg)
            print(f"\n[FAILED] {cfg}")

            if not args.continue_on_error:
                print("Stop because --continue-on-error was not set.")
                sys.exit(code)

    # Generate summary if scripts exist.
    for module in ["scripts.collect_lora_results", "scripts.pretty_results"]:
        module_file = Path(module.replace(".", "/") + ".py")
        if module_file.exists():
            run_command([sys.executable, "-m", module], log_file)

    total_time = time.time() - start_time

    print("\n" + "=" * 100)
    print("LoRA batch finished.")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Log file: {log_file}")
    print("=" * 100)

    print("\nSkipped existing experiments:")
    for cfg in skipped:
        print("  -", cfg)

    print("\nNewly completed experiments:")
    for cfg in completed:
        print("  -", cfg)

    if failed:
        print("\nFailed experiments:")
        for cfg in failed:
            print("  -", cfg)
        sys.exit(1)


if __name__ == "__main__":
    main()