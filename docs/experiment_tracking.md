# Experiment Tracking

This document defines the experiment naming and logging conventions for Group 48.

## Goals

All experiments should be reproducible and comparable across transfer learning strategies.

Each experiment should specify:

- Dataset split
- Model architecture
- Training strategy
- Label budget
- Data augmentation setting
- Optimizer and learning rate
- Weight decay
- Random seed
- Output directory

## Naming Convention

Use the following naming pattern:

```text
{architecture}_{strategy}_{data_setting}_seed{seed}
```

Examples:

```text
resnet18_linear_probe_100_seed42
resnet18_linear_probe_10_seed42
resnet18_finetune_1_seed42
resnet18_finetune_imbalanced_cat20_weighted_ce_seed42
resnet18_lora_r8_100_seed42
```

## Required Metrics

For all experiments, report:

- Top-1 accuracy
- Macro F1
- Per-class accuracy
- Per-class F1

For LoRA experiments, additionally report:

- Trainable parameter count
- Training time
- LoRA rank
- LoRA alpha

For imbalance experiments, additionally report:

- Confusion matrix
- Per-class F1, especially for underrepresented cat breeds

## Output Directory

Each experiment should save outputs under:

```text
experiments/{experiment_name}/
```

Recommended files:

```text
config.yaml
metrics.json
training_curves.csv
best_model.pt
```

Large checkpoint files should not be committed to GitHub.

## Reproducibility

Use:

- Random seed: 42 by default
- Saved split files from `data/splits/`
- Shared YAML configuration files under `configs/`

## Current Planned Experiments

### Linear Probing

- `resnet18_linear_probe_100_seed42`
- `resnet18_linear_probe_10_seed42`
- `resnet18_linear_probe_1_seed42`

### Full Fine-Tuning

- `resnet18_finetune_100_seed42`
- `resnet18_finetune_10_seed42`
- `resnet18_finetune_1_seed42`

### Class Imbalance

- `resnet18_finetune_imbalanced_cat20_weighted_ce_seed42`
- `resnet18_finetune_imbalanced_cat20_oversampling_seed42`

### LoRA

- `resnet18_lora_r4_100_seed42`
- `resnet18_lora_r8_100_seed42`
- `resnet18_lora_r16_100_seed42`