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