# Fine-tuning Baselines

Jingmeng Xie is responsible for the fine-tuning baseline experiments.

## Overview

This document describes the training infrastructure and all baseline experiments
completed for Default Project 1, §1.1. The work builds directly on the data
pipeline from Part 1 (Pengyu Wang) and covers:

- Binary cat-vs-dog sanity check
- Linear probing and full fine-tuning across label budgets
- Strategy 1: simultaneous fine-tuning of the last l layers
- Strategy 2: gradual layer unfreezing
- Augmentation and L2 regularization ablations
- Class imbalance experiments

## New Source Files

### `src/metrics.py`

A shared utility module providing five pure functions for classification metrics.
It has no dependencies on PyTorch or GPUs and can be unit-tested instantly.
All three teammates who produce predictions (fine-tuning, LoRA, visualisation)
should import from here to ensure consistent numbers across the report.

Functions:

- `compute_top1_accuracy` — fraction of correctly predicted samples
- `compute_macro_f1` — unweighted average of per-class F1; always pass
  `num_classes=37` so that classes absent from a small validation split are
  counted as zero rather than silently dropped
- `compute_per_class_accuracy` — per-class recall for all 37 breeds
- `compute_per_class_f1` — per-class F1 for all 37 breeds; missing classes
  return 0.0 rather than raising a warning
- `compute_confusion_matrix` — 37×37 matrix returned as a nested Python list so
  it can be passed directly to `json.dump`

All functions accept list, NumPy array, or PyTorch tensor inputs and return plain
Python floats or lists, not NumPy arrays.

### `src/engine.py`

Pure training and evaluation functions with no file I/O. Kept separate from
`src/train.py` so that the LoRA and imbalance experiments can reuse the same
loop without duplicating code or introducing inconsistencies.

Functions:

- `train_one_epoch` — runs one full pass over the training loader; accumulates
  loss and accuracy using sample-weighted averages so the final incomplete batch
  does not skew the mean; returns a dict containing `loss`, `top1`,
  `num_samples`, and `time_seconds`
- `evaluate` — runs inference with gradients disabled and `model.eval()` active;
  always collects predictions internally for metric computation regardless of
  the `return_predictions` flag; when `return_predictions=True`, also returns
  the raw prediction arrays, labels, and logits for downstream figures

### `src/schedulers.py`

Implements `GradualUnfreezingScheduler` for Strategy 2.

The scheduler holds a list of `{epoch, unfreeze_blocks}` entries. Each call to
`step(epoch)` finds the latest entry whose threshold has been reached and
unfreezes the corresponding set of blocks if the state has changed. It returns
`True` when a change occurs.

When `step` returns `True`, the training loop must rebuild the optimizer. AdamW
maintains per-parameter momentum state; newly unfrozen parameters that are not
registered with the optimizer receive gradients but are never updated.

### `src/models.py` (extended from Part 1)

Two additions:

- `RESNET_BLOCKS_OUTPUT_TO_INPUT` — list `["fc", "layer4", "layer3", "layer2",
  "layer1"]` used by both the partial fine-tuning strategy and the scheduler to
  define unfreezing order
- `freeze_all_except_last_n_blocks` — freezes all parameters then unfreezes
  the last n blocks counting from the output; n=1 is equivalent to linear
  probing, n=2 adds layer4, n=3 further adds layer3

`configure_model_for_strategy` gains two new strategy branches:

- `partial_finetune` — reads `unfrozen_layers` from the config and applies the
  corresponding freeze pattern at the start of training
- `gradual_unfreezing` — sets the initial state to linear probing; the scheduler
  handles subsequent unfreezing events during training

The `lora` strategy still raises `NotImplementedError` and is left for the LoRA
lead to implement.

### `src/train.py` (core funciton)

Complete training loop. Called as `python -m src.train --config <path>`.

Reads the config, builds the model and optimizer, runs the epoch loop, selects
the best checkpoint by validation top-1, reloads it, and evaluates on the
official test split. For gradual unfreezing experiments, the scheduler is
initialised before the first epoch and the optimizer is rebuilt whenever a new
layer is unfrozen.

For each experiment, five files are written to `experiments/{name}/`:

```text
config.yaml             resolved hyperparameters for this run  
training_curves.csv     one row per epoch                     
metrics.json            summary + per-class metrics           
best_model.pt           state_dict of the best epoch          
best_predictions.npz    val and test logits / preds / labels  
```

`training_curves.csv` is written before the test evaluation step so that
training data is preserved even if the test step fails.

### `src/binary_data.py`

DataLoader for the binary cat-vs-dog task, used only by the §1.1 sanity check.
Uses `OxfordIIITPet` with `target_types="binary-category"`, which returns 0 for
cats and 1 for dogs without any manual label mapping.

### `scripts/sanity_check_binary.py`

Standalone script for the §1.1 binary sanity check. Freezes all layers except
the two-class output head (1,026 trainable parameters) and trains for 10 epochs.
Does not use a YAML config. Run with `python -m scripts.sanity_check_binary`.

## New Configuration Files

Eleven YAML files added under `configs/`:

```text
partial_finetune_l2_100.yaml      Strategy 1, fc + layer4
partial_finetune_l3_100.yaml      Strategy 1, fc + layer4 + layer3
gradual_unfreeze_100.yaml         Strategy 2, 100% label budget
gradual_unfreeze_10.yaml          Strategy 2, 10% label budget
gradual_unfreeze_1.yaml           Strategy 2, 1% label budget
finetune_10_noaug.yaml            augmentation ablation, 10% data
finetune_1_noaug.yaml             augmentation ablation, 1% data
finetune_10_wdhigh.yaml           L2 ablation, 10% data (weight_decay = 5e-3)
finetune_1_wdhigh.yaml            L2 ablation, 1% data (weight_decay = 5e-3)
imbalanced_baseline.yaml          imbalance experiment, no compensation
```

The two compensation configs (`imbalanced_weighted_ce.yaml` and
`imbalanced_oversampling.yaml`) were provided by Part 1 and used unchanged.

## Experiment Results

### Binary Sanity Check

Linear probing with 1,026 trainable parameters achieves 98.3% test accuracy in
10 epochs. The result confirms that ImageNet pretrained features are strongly
discriminative for the binary task.

### Linear Probing vs Full Fine-tuning

| Strategy        | Budget | test top-1 | test macro F1 | Trainable params |
|-----------------|--------|-----------|--------------|-----------------|
| linear_probe    | 100%   | 0.846     | 0.842        | 18,981          |
| linear_probe    | 10%    | 0.737     | 0.736        | 18,981          |
| linear_probe    | 1%     | 0.388     | 0.377        | 18,981          |
| full_finetuning | 100%   | 0.868     | 0.865        | 11,195,493      |
| full_finetuning | 10%    | 0.713     | 0.708        | 11,195,493      |
| full_finetuning | 1%     | 0.261     | 0.247        | 11,195,493      |

At 100% data, full fine-tuning outperforms linear probing by 2.2 points.
At 10% and 1% data the ranking reverses: linear probing wins because 11 million
parameters overfit severely with 332 or 37 training images.

### Strategy 1: Partial Fine-tuning (100% data)

| l   | Unfrozen blocks        | Trainable params | test top-1 | Time per epoch |
|-----|------------------------|-----------------|-----------|----------------|
| 1   | fc only                | 18,981          | 0.846     | ~34 s          |
| 2   | fc + layer4            | 8,412,709       | 0.866     | ~40 s          |
| 3   | fc + layer4 + layer3   | 10,512,421      | 0.868     | ~47 s          |
| all | all layers             | 11,195,493      | 0.868     | ~71 s          |

The largest gain is from l=1 to l=2 (+2.1 points). Beyond l=2, marginal returns
fall below 0.2 points. Fine-tuning fc and layer4 achieves 99.8% of full
fine-tuning accuracy at 56% of the per-epoch training time.

### Strategy 2: Gradual Unfreezing

| Budget | best epoch | test top-1 | test macro F1 | Training time |
|--------|-----------|-----------|--------------|---------------|
| 100%   | 13        | 0.874     | 0.872        | 1943 s        |
| 10%    | 19        | 0.732     | 0.726        | 1089 s        |
| 1%     | 28        | 0.288     | 0.276        | 868 s         |

At 100% data, gradual unfreezing achieves the highest test top-1 across all
strategies (0.874), 0.6 points above full fine-tuning. The difference is small;
both strategies are viable for the report.

A temporary accuracy dip occurs when layer2 is unfrozen at epoch 15 because the
optimizer is rebuilt and new parameters restart from zero momentum state. Recovery
takes one to two epochs. This is a known limitation of gradual unfreezing and can
be reduced with a learning rate warm-up.

At 10% and 1% data, gradual unfreezing does not outperform linear probing because
the model eventually unfreezes the same 11 million parameters that limited data
cannot constrain.

### Augmentation Ablation

| Budget | aug ON (baseline) | aug OFF | Absolute drop | Relative drop |
|--------|------------------|---------|--------------|---------------|
| 10%    | 0.713            | 0.665   | −4.8 pts     | −6.7%         |
| 1%     | 0.261            | 0.231   | −3.0 pts     | −11.5%        |

The relative benefit of augmentation is greater at smaller label budgets,
consistent with the expectation in §1.1.

### L2 Regularization Ablation

| Budget | wd = 1e-4 (baseline) | wd = 5e-3 | Difference |
|--------|---------------------|-----------|------------|
| 10%    | 0.713               | 0.713     | +0.03 pts  |
| 1%     | 0.261               | 0.261     | +0.06 pts  |

Increasing weight decay by 50× has negligible effect. Augmentation already
provides sufficient regularisation, and AdamW's adaptive learning rates reduce
sensitivity to the weight decay coefficient. This is itself a reportable finding.

### Class Imbalance

Cat breeds are reduced to 20% of their training count. Validation and test splits
remain balanced.

| Experiment                   | test top-1 | test macro F1 |
|------------------------------|-----------|--------------|
| balanced reference (100%)    | 0.868     | 0.865        |
| imbalanced, no compensation  | 0.830     | 0.825        |
| imbalanced + weighted CE     | 0.847     | 0.844        |
| imbalanced + oversampling    | 0.841     | 0.835        |

Macro F1 drops more than overall accuracy, confirming that the per-class F1 of
minority cat breeds is disproportionately harmed. Weighted cross-entropy
outperforms oversampling because oversampling repeats the same minority images,
increasing the risk of overfitting to them. Neither strategy fully closes the gap
to the balanced baseline.

## Output File Reference

### `metrics.json`

Key fields for the report and downstream figures:

```text
best_val_top1, best_val_macro_f1, best_epoch
test_top1, test_macro_f1
trainable_parameters_final, training_time_seconds
per_class_val_accuracy, per_class_val_f1     (length-37 lists)
per_class_test_accuracy, per_class_test_f1   (length-37 lists)
val_confusion_matrix, test_confusion_matrix   (37×37 nested lists)
class_names                                    (length-37 list)
```

### `best_predictions.npz`

Arrays for the best validation epoch:

```text
val_predictions, val_labels     (368 samples)
val_logits                       (368 × 37)
test_predictions, test_labels   (3669 samples)
test_logits                      (3669 × 37)
class_names
```

Load with `numpy.load(..., allow_pickle=True)`. Per-class metrics in
`metrics.json` are pre-computed from these arrays, so most visualisation tasks
can read the JSON directly without loading the full logit arrays.

## Notes for Downstream Tasks

### LoRA lead (Yu Zhang)

Implement `src/lora.py` and add an `elif strategy == "lora"` branch in
`configure_model_for_strategy`. The training loop, metric saving, and output file
structure in `src/train.py` do not need to change. Trainable parameter count and
training time are already written to `metrics.json` automatically.

### Visualisation lead (Jiachen Shi)

All per-class F1 scores and confusion matrices for both validation and test sets
are pre-computed in every `metrics.json`. No retraining is needed for standard
report figures. For figures that require raw logits, such as t-SNE, use the
`test_logits` array from `best_predictions.npz`.

To collect results across all completed experiments, iterate over
`experiments/*/metrics.json`.

## Known Limitations

- The binary sanity check reaches 98.3% rather than ≥99%.

- The best test top-1 across all strategies is 87.4%. The best validation top-1
  is 92.4%. The gap reflects variance from the small validation set (368
  samples).

- Several experiments at 1% data have not converged by the final epoch. Extending
  training to 50 epochs would yield higher final numbers.

- The reported training time for `finetune_100` is 4418 s because the machine
  entered sleep mode during two epochs. The trained model and all metrics are
  unaffected.