# Experiments

This folder contains the output of all fine-tuning baseline experiments (Part 2).
Each subfolder corresponds to one training run and holds up to five files:

```
config.yaml              hyperparameters used for this run
training_curves.csv      per-epoch loss / top-1 / macro F1 / epoch time
metrics.json             final summary + per-class F1 + confusion matrices
best_model.pt            model weights for the best validation epoch 
best_predictions.npz     val and test logits / predictions / labels  
```

All numerical results can be read from `metrics.json`.
Raw logits for t-SNE or top-5 analysis are in `best_predictions.npz`.

---

## Binary Sanity Check

| Folder | Description | test top-1 |
|--------|-------------|-----------|
| `sanity_binary_cat_vs_dog_seed42` | Linear probing, 2-class cat vs dog, 1,026 trainable params | 0.983 |

---

## Linear Probing vs Full Fine-tuning

| Folder | Strategy | Data | test top-1 |
|--------|----------|------|-----------|
| `resnet18_linear_probe_100_seed42` | Linear probe | 100% | 0.846 |
| `resnet18_linear_probe_10_seed42`  | Linear probe | 10%  | 0.737 |
| `resnet18_linear_probe_1_seed42`   | Linear probe | 1%   | 0.388 |
| `resnet18_finetune_100_seed42`     | Full fine-tune | 100% | 0.868 |
| `resnet18_finetune_10_seed42`      | Full fine-tune | 10%  | 0.713 |
| `resnet18_finetune_1_seed42`       | Full fine-tune | 1%   | 0.261 |

Linear probing outperforms full fine-tuning at 10% and 1% data.

---

## Strategy 1 — Partial Fine-tuning (100% data)

| Folder | Unfrozen layers | Trainable params | test top-1 |
|--------|-----------------|-----------------|-----------|
| `resnet18_linear_probe_100_seed42` *(reference)* | fc | 18,981 | 0.846 |
| `resnet18_partial_finetune_l2_100_seed42` | fc + layer4 | 8,412,709 | 0.866 |
| `resnet18_partial_finetune_l3_100_seed42` | fc + layer4 + layer3 | 10,512,421 | 0.868 |
| `resnet18_finetune_100_seed42` *(reference)* | all | 11,195,493 | 0.868 |

Unfreezing layer4 (l=2) provides most of the gain. Further unfreezing adds less than 0.2 points.

---

## Strategy 2 — Gradual Unfreezing

| Folder | Data | Unfreeze schedule | test top-1 |
|--------|------|------------------|-----------|
| `resnet18_gradual_unfreeze_100_seed42` | 100% | fc→+layer4→+layer3→+layer2 at epochs 0/5/10/15 | **0.874** |
| `resnet18_gradual_unfreeze_10_seed42`  | 10%  | same | 0.732 |
| `resnet18_gradual_unfreeze_1_seed42`   | 1%   | same | 0.288 |

Best overall test top-1 across all experiments is 0.874 (gradual unfreezing, 100% data).

---

## Augmentation Ablation

| Folder | Data | Aug | test top-1 | vs baseline |
|--------|------|-----|-----------|-------------|
| `resnet18_finetune_10_seed42` *(baseline)* | 10% | ON  | 0.713 | — |
| `resnet18_finetune_10_noaug_seed42`         | 10% | OFF | 0.665 | −4.8 pts |
| `resnet18_finetune_1_seed42` *(baseline)*  | 1%  | ON  | 0.261 | — |
| `resnet18_finetune_1_noaug_seed42`          | 1%  | OFF | 0.231 | −3.0 pts |

Relative benefit is larger at 1% (−11.5%) than at 10% (−6.7%).

---

## L2 Regularization Ablation

| Folder | Data | weight_decay | test top-1 | vs baseline |
|--------|------|-------------|-----------|-------------|
| `resnet18_finetune_10_seed42` *(baseline)*    | 10% | 1e-4 | 0.713 | — |
| `resnet18_finetune_10_wdhigh_seed42`           | 10% | 5e-3 | 0.713 | +0.0 pts |
| `resnet18_finetune_1_seed42` *(baseline)*     | 1%  | 1e-4 | 0.261 | — |
| `resnet18_finetune_1_wdhigh_seed42`            | 1%  | 5e-3 | 0.261 | +0.0 pts |

Increasing weight decay 50× has negligible effect; augmentation already provides sufficient regularization.

---

## Class Imbalance

Cat breeds are reduced to 20% of their training count; dog breeds are unchanged.

| Folder | Compensation | test top-1 | test macro F1 |
|--------|-------------|-----------|--------------|
| `resnet18_finetune_100_seed42` *(balanced reference)* | — | 0.868 | 0.865 |
| `resnet18_finetune_imbalanced_baseline_seed42`         | None | 0.830 | 0.825 |
| `resnet18_finetune_imbalanced_cat20_weighted_ce_seed42` | Weighted CE | 0.847 | 0.844 |
| `resnet18_finetune_imbalanced_cat20_oversampling_seed42` | Oversampling | 0.841 | 0.835 |

Weighted CE outperforms oversampling. Neither fully recovers to the balanced baseline.

---

---

# Experiments — LoRA Extension (Part 3, A-level)

Each LoRA run follows the same folder structure as Part 2.
A consolidated CSV of all 18 LoRA runs is at
`results/lora_ablation_summary.csv`.

---

## Overview

Two target configurations are compared across three ranks and
three label budgets (18 runs total):

- **FC-LoRA**: LoRA applied to the final linear layer only.
  Trainable params range from 2,196 (r=4) to 8,784 (r=16).
- **Layer4-LoRA**: LoRA applied to all Conv2d layers in
  `layer4` plus the fc head.
  Trainable params range from 94,757 (r=4) to 322,085 (r=16).

Baselines from Part 2 (linear probe, full fine-tuning, gradual
unfreezing) are referenced throughout for direct comparison.

---

## FC-LoRA — Rank Ablation

LoRA applied to `fc` only. Base ResNet-18 backbone is fully
frozen. `alpha = 2 × rank` throughout (scaling = 2).

| Folder | Budget | Rank | Trainable params | test top-1 | test macro F1 |
|--------|--------|------|-----------------|-----------|--------------|
| `resnet18_lora_r4_100_seed42`  | 100% | 4  | 2,196  | 0.608 | 0.605 |
| `resnet18_lora_r8_100_seed42`  | 100% | 8  | 4,392  | 0.792 | 0.790 |
| `resnet18_lora_r16_100_seed42` | 100% | 16 | 8,784  | 0.820 | 0.818 |
| `resnet18_lora_r4_10_seed42`   | 10%  | 4  | 2,196  | 0.210 | 0.171 |
| `resnet18_lora_r8_10_seed42`   | 10%  | 8  | 4,392  | 0.558 | 0.548 |
| `resnet18_lora_r16_10_seed42`  | 10%  | 16 | 8,784  | 0.651 | 0.650 |
| `resnet18_lora_r4_1_seed42`    | 1%   | 4  | 2,196  | 0.054 | 0.039 |
| `resnet18_lora_r8_1_seed42`    | 1%   | 8  | 4,392  | 0.075 | 0.050 |
| `resnet18_lora_r16_1_seed42`   | 1%   | 16 | 8,784  | 0.145 | 0.112 |

FC-LoRA underperforms linear probing at all budgets and ranks.
The update matrix ΔW = BA has rank at most r, which is smaller
than the output dimension (37); this hard capacity limit
explains the gap versus linear probing (full-rank fc, 18,981
params). Higher rank consistently helps, especially at low data
where r=4 nearly fails to learn.

---

## Layer4-LoRA — Rank Ablation

LoRA applied to all Conv2d layers in `layer4` and the fc head.
`train_classifier: true` so the fc base weights are also
updated. `alpha = 2 × rank` throughout (scaling = 2).

| Folder | Budget | Rank | Trainable params | test top-1 | test macro F1 |
|--------|--------|------|-----------------|-----------|--------------|
| `resnet18_lora_layer4_r4_100_seed42`  | 100% | 4  | 94,757  | **0.850** | 0.847 |
| `resnet18_lora_layer4_r8_100_seed42`  | 100% | 8  | 170,533 | 0.840     | 0.837 |
| `resnet18_lora_layer4_r16_100_seed42` | 100% | 16 | 322,085 | 0.845     | 0.843 |
| `resnet18_lora_layer4_r4_10_seed42`   | 10%  | 4  | 94,757  | **0.753** | 0.750 |
| `resnet18_lora_layer4_r8_10_seed42`   | 10%  | 8  | 170,533 | 0.744     | 0.739 |
| `resnet18_lora_layer4_r16_10_seed42`  | 10%  | 16 | 322,085 | 0.729     | 0.725 |
| `resnet18_lora_layer4_r4_1_seed42`    | 1%   | 4  | 94,757  | **0.411** | 0.400 |
| `resnet18_lora_layer4_r8_1_seed42`    | 1%   | 8  | 170,533 | 0.378     | 0.364 |
| `resnet18_lora_layer4_r16_1_seed42`   | 1%   | 16 | 322,085 | 0.351     | 0.343 |

r=4 is best across all three budgets. Lower rank imposes
stronger implicit regularisation, which helps more as data
decreases. At 100%, r=4 uses 0.85% of full fine-tuning's
parameters while losing only 1.8 accuracy points.

---

## Comparison Against Part 2 Baselines

All experiments share the same data splits, test set, and
evaluation script as Part 2, so numbers are directly comparable.

| Method | Params | 100% | 10% | 1% |
|--------|--------|------|-----|----|
| Linear probe *(Part 2)* | 18,981 | 0.846 | 0.737 | 0.388 |
| Full fine-tuning *(Part 2)* | 11,195,493 | 0.868 | 0.713 | 0.261 |
| Gradual unfreezing *(Part 2)* | 11,037,989 | **0.874** | 0.732 | 0.288 |
| FC-LoRA r=16 *(Part 3)* | 8,784 | 0.820 | 0.651 | 0.145 |
| **Layer4-LoRA r=4** *(Part 3)* | **94,757** | 0.850 | **0.753** | **0.411** |

Key findings:

- At **100% data**, Layer4-LoRA r=4 matches linear probing
  and comes within 1.8 points of full fine-tuning, using
  only 0.85% as many trainable parameters.
- At **10% data**, Layer4-LoRA r=4 outperforms both full
  fine-tuning (+4.0 pts) and gradual unfreezing (+2.1 pts).
  Full fine-tuning overfits with 11M parameters on 332
  images; LoRA's low-rank constraint acts as implicit
  regularisation.
- At **1% data**, Layer4-LoRA r=4 outperforms full
  fine-tuning by 15 points and even exceeds linear probing
  (+2.3 pts), showing that a small trainable update to
  layer4 is more valuable than fully freezing the backbone.
