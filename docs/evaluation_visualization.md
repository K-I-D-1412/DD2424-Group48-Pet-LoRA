# Evaluation and Visualization

Jiachen Shi is responsible for evaluation, visualization, and report writing.

## Overview

This document covers the evaluation and visualization layer of the project.
The training infrastructure (Part 1 + Part 2) already produces a full set of
metrics and raw predictions for every experiment, so no retraining is required:

- `experiments/{run}/metrics.json` — top-1, macro F1, per-class accuracy,
  per-class F1, confusion matrices (both validation and test), trainable
  parameter counts, and per-run training time.
- `experiments/{run}/training_curves.csv` — per-epoch loss, top-1, macro F1,
  and epoch time.
- `experiments/{run}/best_predictions.npz` — raw test/val predictions, labels,
  and 37-dim logits used for representation analysis.

All figures below are produced by stand-alone scripts under `scripts/` and
written to `results/figures/`. Every script is reproducible: re-running it
overwrites the previous PNGs from the same JSON / CSV / NPZ inputs.

## Scripts

Each script is invoked as `python -m scripts.<name>`.

| Script | Purpose | Output directory |
|---|---|---|
| `scripts/plot_training_curves.py` | Train/val loss, val top-1, val macro F1 per group of runs | `results/figures/training_curves/` |
| `scripts/plot_confusion_matrices.py` | Row-normalised confusion matrices for the test split | `results/figures/confusion_matrices/` |
| `scripts/plot_per_class_f1.py` | Per-class test F1 bar charts for the imbalance experiments | `results/figures/per_class_f1/` |
| `scripts/plot_lora_params_vs_acc.py` | Trainable parameters vs test accuracy (LoRA vs baselines) | `results/figures/lora_params_vs_acc/` |
| `scripts/plot_representation_analysis.py` | t-SNE and PCA of test logits, per strategy | `results/figures/representation/` |
| `scripts/plot_training_cost.py` | Training time, per-epoch cost, convergence speed, efficiency | `results/figures/training_cost/` |

The scripts depend only on `numpy`, `pandas`, `matplotlib`, and (for the
representation analysis) `scikit-learn`. They do not require PyTorch or the
Oxford-IIIT Pet image data.

## Figures Produced

A total of 26 figures across six categories.

### Training curves (`results/figures/training_curves/`)

Each figure is a 1x3 panel of train/val loss, validation top-1, and
validation macro F1 across epochs.

- `baselines_1.png`, `baselines_10.png`, `baselines_100.png` — linear probe,
  full fine-tuning, gradual unfreezing, and Layer4-LoRA r=4 at 1% / 10% / 100%
  label budgets.
- `lora_layer4_rank_100.png` and `lora_fc_rank_100.png` — rank ablations
  (r = 4, 8, 16) at 100% labels.
- `imbalance.png` — balanced reference vs the three imbalance compensation
  strategies.
- `augmentation_ablation.png` — augmentation on vs off at 10% and 1% labels.

### Confusion matrices (`results/figures/confusion_matrices/`)

Row-normalised so the diagonal reads as per-class recall. Cat breeds are
highlighted in red on imbalance figures to make minority-class behaviour
immediately visible.

- `imbalance_grid.png` — 2x2 grid of balanced reference, no compensation,
  weighted CE, and oversampling.
- `strategies_100_grid.png` — 2x2 grid of linear probe, full fine-tuning,
  gradual unfreezing, and Layer4-LoRA r=4 at 100% labels.
- `imbalance_baseline.png` and `imbalance_weighted_ce.png` — single-figure
  closeups suitable for the report.

### Per-class F1 bar charts (`results/figures/per_class_f1/`)

Cats are placed on the left and shown in red so the imbalance story is
visually grouped.

- `imbalance_per_class_f1.png` — grouped bars: balanced reference, no
  compensation, weighted CE, oversampling.
- `imbalance_species_summary.png` — mean F1 over cats vs dogs plus the full
  macro F1, one bar per compensation strategy.
- `imbalance_per_class_delta.png` — per-class delta vs the imbalanced
  baseline, separating weighted CE and oversampling.

### LoRA parameters vs accuracy (`results/figures/lora_params_vs_acc/`)

Log-scale x axis. Pareto frontier comparisons of trainable parameter count
against test top-1.

- `params_vs_acc_1pct.png`, `params_vs_acc_10pct.png`,
  `params_vs_acc_100pct.png` — one figure per label budget.
- `params_vs_acc_all.png` — combined 1x3 panel for direct comparison.

### Representation analysis (`results/figures/representation/`)

Two-dimensional embeddings of the test logits for four strategies at 100%
labels. Logits are used because penultimate-feature extraction would require
re-running the backbone; the geometry of the logit space still distinguishes
the four strategies.

- `tsne_by_class.png` — t-SNE coloured by ground-truth class (37 classes).
- `tsne_by_species.png` — t-SNE coloured by species (cat vs dog).
- `pca_by_class.png`, `pca_by_species.png` — PCA equivalents as a
  reproducible cross-check.

### Training cost (`results/figures/training_cost/`)

- `time_vs_acc.png` — 1x3 scatter of total training time vs test top-1, one
  panel per label budget.
- `perepoch_vs_params.png` — per-epoch time vs trainable parameters at 100%
  labels, log-scale x axis.
- `convergence_best_epoch.png` — grouped bars of best validation epoch by
  strategy and label budget; lower is faster.
- `efficiency_100.png` — cost-per-accuracy ranking plus a Pareto sketch of
  test error vs total training time at 100% labels.

## Coverage Against the Proposal

| Proposal requirement | Source | Figure / table |
|---|---|---|
| Top-1 accuracy, macro F1, per-class accuracy for all experiments | `metrics.json` | All figures below; numeric summaries in `results/experiment_summary.{csv,md}` |
| Track trainable parameter counts | `metrics.json` | `lora_params_vs_acc/`, `training_cost/perepoch_vs_params.png` |
| Plot training curves to analyse convergence | `training_curves.csv` | `training_curves/`, `training_cost/convergence_best_epoch.png` |
| Confusion matrices for imbalance experiments | `metrics.json:test_confusion_matrix` | `confusion_matrices/imbalance_grid.png` and single-figure closeups |
| Per-class F1 for imbalance experiments | `metrics.json:per_class_test_f1` | `per_class_f1/` (three figures) |
| Simple representation analysis (C/D-level) | `best_predictions.npz:test_logits` | `representation/` (four figures) |

## Re-running

```bash
python -m scripts.plot_training_curves
python -m scripts.plot_confusion_matrices
python -m scripts.plot_per_class_f1
python -m scripts.plot_lora_params_vs_acc
python -m scripts.plot_representation_analysis
python -m scripts.plot_training_cost
```

The scripts are idempotent. They overwrite existing PNGs but read only from
committed JSON / CSV / NPZ files, so re-running them on a teammate's machine
produces the same figures.

## Known Caveats

- The representation-analysis figures embed 37-dim test logits, not the
  512-dim penultimate features. Penultimate features would require loading
  `best_model.pt` and re-running inference (PyTorch and the Oxford-IIIT Pet
  images, neither of which is needed elsewhere in the visualisation layer).
  Logit-based t-SNE still distinguishes the four strategies visually and is
  reported as such.
- `resnet18_finetune_100_seed42` reports 4418 s of training time because the
  training machine entered sleep mode during two epochs. The trained model
  and all accuracy metrics are unaffected, but `time_vs_acc.png` and
  `efficiency_100.png` carry this inflated time. The figures should be cited
  alongside the per-epoch numbers (~145 s/epoch) when discussing training
  cost in the report.

## Outstanding Items

- Final report and video presentation.
- (Optional) Replace logit-based t-SNE with penultimate-feature t-SNE by
  adding a short feature-extraction step on Colab and re-running
  `scripts/plot_representation_analysis.py`.
- (Optional) Top-5 accuracy table from `best_predictions.npz:test_logits`.
