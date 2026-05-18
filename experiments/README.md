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

# README Update — LoRA Extension Section Only

> Replace only the existing README section starting from `# Experiments — LoRA Extension (Part 3, A-level)` with the updated section below. Do not change the earlier Part 2 / baseline sections.

---

# Experiments — LoRA Extension (Part 3, A-level)

Each LoRA run follows the same folder structure as Part 2.
The consolidated LoRA results are collected in `results/lora_ablation_summary.csv`, and the final experiment-level summary is in `results/experiment_summary.csv` / `results/experiment_summary.md`.

---

## Overview

The LoRA extension evaluates three questions:

1. **Rank:** how does the LoRA rank affect performance?
2. **Placement:** where should LoRA adapters be inserted?
3. **Learning rate:** are the LoRA gains partly explained by using a larger learning rate?

The original LoRA experiments compared two target configurations across three ranks and three label budgets:

- **FC-LoRA:** LoRA applied to the final linear layer only.
  Trainable parameters range from 2,196 (`r=4`) to 8,784 (`r=16`).
- **Layer4-LoRA:** LoRA applied to all Conv2d layers in `layer4`, with the 37-way classification head trained alongside the adapters.
  Trainable parameters range from 94,757 (`r=4`) to 322,085 (`r=16`).

Additional experiments extend this with placement and learning-rate controls:

- **Layer3-LoRA (`r=4`)** across 1%, 10%, and 100% label budgets.
- **Layer4+Layer3-LoRA (`r=4`)** across 1%, 10%, and 100% label budgets.
- **Layer4-LoRA (`r=4`, lr=`1e-4`)** across 1%, 10%, and 100% label budgets as a learning-rate control.

Main LoRA settings use AdamW, learning rate `1e-3`, weight decay `1e-4`, 30 epochs, batch size 64, and `alpha = 2 × rank` so that the effective LoRA scaling `alpha / rank = 2` is fixed across ranks. The learning-rate control uses lr=`1e-4`, matching dense fine-tuning.

Baselines from Part 2 — linear probing, full fine-tuning, and gradual unfreezing — are referenced for direct comparison.

---

## FC-LoRA — Rank Ablation

LoRA is applied to `fc` only. The ResNet-18 backbone is frozen. Since the 37-way classifier is newly initialized, FC-LoRA should be interpreted as a deliberately capacity-limited classifier baseline rather than a full-capacity linear probe.

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

**Finding:** FC-LoRA underperforms linear probing at all budgets and ranks. The update matrix `ΔW = BA` has rank at most `r`, which is below the classifier's natural rank capacity. Increasing rank helps, but even `r=16` remains below the full linear probe at 100% labels.

---

## Layer4-LoRA — Rank Ablation

LoRA is applied to all Conv2d layers in `layer4`, and the 37-way classification head is trained alongside the adapters.

| Folder | Budget | Rank | Trainable params | test top-1 | test macro F1 |
|--------|--------|------|-----------------|-----------|--------------|
| `resnet18_lora_layer4_r4_100_seed42`  | 100% | 4  | 94,757  | **0.850** | **0.847** |
| `resnet18_lora_layer4_r8_100_seed42`  | 100% | 8  | 170,533 | 0.840 | 0.837 |
| `resnet18_lora_layer4_r16_100_seed42` | 100% | 16 | 322,085 | 0.845 | 0.843 |
| `resnet18_lora_layer4_r4_10_seed42`   | 10%  | 4  | 94,757  | **0.753** | **0.750** |
| `resnet18_lora_layer4_r8_10_seed42`   | 10%  | 8  | 170,533 | 0.744 | 0.739 |
| `resnet18_lora_layer4_r16_10_seed42`  | 10%  | 16 | 322,085 | 0.729 | 0.725 |
| `resnet18_lora_layer4_r4_1_seed42`    | 1%   | 4  | 94,757  | **0.411** | **0.400** |
| `resnet18_lora_layer4_r8_1_seed42`    | 1%   | 8  | 170,533 | 0.378 | 0.364 |
| `resnet18_lora_layer4_r16_1_seed42`   | 1%   | 16 | 322,085 | 0.351 | 0.343 |

**Finding:** Within the Layer4-LoRA family, `r=4` performs best across all three budgets. Increasing rank adds trainable capacity but does not improve generalization, especially under limited labels.

---

## LoRA Placement Ablation (`r=4`, lr=`1e-3`)

This ablation tests whether LoRA should be applied to the final residual stage, an earlier high-level residual stage, or both.

| Target | Trainable params | 1% top-1 | 1% F1 | 10% top-1 | 10% F1 | 100% top-1 | 100% F1 |
|--------|-----------------|---------|------|----------|-------|-----------|--------|
| FC only | 2,196 | 0.054 | 0.039 | 0.210 | 0.171 | 0.608 | 0.605 |
| Layer4 + fc | 94,757 | **0.411** | **0.400** | 0.753 | 0.750 | 0.850 | 0.847 |
| Layer3 + fc | 56,869 | 0.406 | 0.390 | **0.787** | **0.785** | **0.880** | **0.877** |
| Layer4 + Layer3 + fc | 132,645 | 0.381 | 0.372 | 0.754 | 0.753 | 0.861 | 0.858 |

**Finding:** The best LoRA placement is data-dependent.

- At **1% labels**, Layer4-LoRA is slightly best: 0.411 vs. 0.406 for Layer3-LoRA.
- At **10% labels**, Layer3-LoRA is best: 0.787 test top-1.
- At **100% labels**, Layer3-LoRA is also best: 0.880 test top-1.
- Adding both `layer4` and `layer3` does not improve performance, despite increasing trainable parameters to 132,645.

This shows that more LoRA adapters are not automatically better. Adapter placement matters more than simply increasing trainable LoRA capacity.

---

## Learning-Rate Sensitivity — Layer4-LoRA `r=4`

The main LoRA experiments use lr=`1e-3`, whereas dense fine-tuning uses lr=`1e-4`. To control for this difference, Layer4-LoRA `r=4` is rerun with lr=`1e-4`.

| Budget | Setting | Trainable params | Best epoch | test top-1 | test macro F1 |
|--------|---------|-----------------|-----------|-----------|--------------|
| 1% | Layer4-LoRA `r=4`, lr=`1e-3` | 94,757 | 19 | **0.411** | **0.400** |
| 1% | Layer4-LoRA `r=4`, lr=`1e-4` | 94,757 | 29 | 0.078 | 0.054 |
| 10% | Layer4-LoRA `r=4`, lr=`1e-3` | 94,757 | 20 | **0.753** | **0.750** |
| 10% | Layer4-LoRA `r=4`, lr=`1e-4` | 94,757 | 29 | 0.545 | 0.531 |
| 100% | Layer4-LoRA `r=4`, lr=`1e-3` | 94,757 | 14 | 0.850 | 0.847 |
| 100% | Layer4-LoRA `r=4`, lr=`1e-4` | 94,757 | 29 | **0.855** | **0.853** |

**Finding:** LoRA's scarce-data gains are learning-rate sensitive.

- At **1% and 10% labels**, reducing the LoRA learning rate to `1e-4` severely hurts performance.
- At **100% labels**, lr=`1e-4` slightly improves Layer4-LoRA.
- Therefore, LoRA's low-data advantage is not purely architectural; it partly relies on using a larger adapter learning rate for fast adaptation.

---

## Comparison Against Part 2 Baselines

All experiments share the same data splits, official test set, and evaluation script as Part 2, so numbers are directly comparable.

| Method | Params | 100% top-1 | 10% top-1 | 1% top-1 |
|--------|--------|-----------|----------|---------|
| Linear probe *(Part 2)* | 18,981 | 0.846 | 0.737 | 0.388 |
| Full fine-tuning *(Part 2)* | 11,195,493 | 0.868 | 0.713 | 0.261 |
| Gradual unfreezing *(Part 2)* | 11,037,989 | 0.874 | 0.732 | 0.288 |
| FC-LoRA `r=16` *(Part 3)* | 8,784 | 0.820 | 0.651 | 0.145 |
| Layer4-LoRA `r=4` *(Part 3)* | 94,757 | 0.850 | 0.753 | **0.411** |
| Layer3-LoRA `r=4` *(Part 3)* | 56,869 | **0.880** | **0.787** | 0.406 |

Key findings:

- At **1% labels**, Layer4-LoRA `r=4` is best, reaching 0.411 test top-1. It beats linear probing by 2.3 points and full fine-tuning by 15.0 points.
- At **10% labels**, Layer3-LoRA `r=4` is best, reaching 0.787 test top-1. It beats linear probing by 5.0 points and full fine-tuning by 7.4 points.
- At **100% labels**, Layer3-LoRA `r=4` reaches 0.880 test top-1, outperforming full fine-tuning (0.868) and gradual unfreezing (0.874), while using only 56,869 trainable parameters.
- Layer3-LoRA uses about **0.51%** of full fine-tuning's trainable parameters.

Overall, LoRA is not uniformly best by simply adding adapters everywhere. The best placement changes with the amount of labelled data: `layer4` is safest at 1%, while `layer3` becomes stronger at 10% and 100%.

---

# Summary for Teammates

## What I additionally ran

I added **9 new LoRA-related experiments** beyond the original LoRA runs.

### 1. Layer3-LoRA placement experiments

These test whether applying LoRA to `layer3` is better than applying it only to `layer4`.

- `configs/lora_layer3_r4_1.yaml`
- `configs/lora_layer3_r4_10.yaml`
- `configs/lora_layer3_r4_100.yaml`

Results:

| Method | 1% top-1 | 10% top-1 | 100% top-1 |
|--------|---------|----------|-----------|
| Layer3-LoRA `r=4` | 0.406 | **0.787** | **0.880** |

Layer3-LoRA is best at 10% and 100% labels.

### 2. Layer4+Layer3-LoRA placement experiments

These test whether adding LoRA to both `layer4` and `layer3` improves over using only one stage.

- `configs/lora_layer4_layer3_r4_1.yaml`
- `configs/lora_layer4_layer3_r4_10.yaml`
- `configs/lora_layer4_layer3_r4_100.yaml`

Results:

| Method | 1% top-1 | 10% top-1 | 100% top-1 |
|--------|---------|----------|-----------|
| Layer4+Layer3-LoRA `r=4` | 0.381 | 0.754 | 0.861 |

This did **not** improve performance. Even though it has more trainable parameters, it is worse than Layer3-LoRA at 10% and 100%, and worse than Layer4-LoRA at 1%.

### 3. Layer4-LoRA learning-rate control

These test whether Layer4-LoRA's original advantage was partly due to its higher learning rate (`1e-3`) compared with dense fine-tuning (`1e-4`).

- `configs/lora_layer4_r4_lr1e4_1.yaml`
- `configs/lora_layer4_r4_lr1e4_10.yaml`
- `configs/lora_layer4_r4_lr1e4_100.yaml`

Results:

| Budget | Layer4-LoRA lr=`1e-3` | Layer4-LoRA lr=`1e-4` |
|--------|----------------------|----------------------|
| 1% | 0.411 | 0.078 |
| 10% | 0.753 | 0.545 |
| 100% | 0.850 | 0.855 |

This shows that the low-data LoRA gains partly rely on using a larger adapter learning rate. At full data, the lower learning rate slightly helps.

---

## How this changes the original conclusion

### Original conclusion

The previous README/report mainly supported this conclusion:

> Layer4-LoRA `r=4` is the best LoRA variant, especially under 1% and 10% labels.

### Updated conclusion

The new experiments give a more precise conclusion:

> The best LoRA placement is data-dependent. Layer4-LoRA is best at 1% labels, while Layer3-LoRA is best at 10% and 100% labels. Adding both Layer3 and Layer4 does not help, so more LoRA adapters are not automatically better.

### Why this improves the report

These new experiments fill three gaps:

1. **Placement ablation gap**  
   Before, we only compared FC-LoRA and Layer4-LoRA. Now we also test Layer3-LoRA and Layer4+Layer3-LoRA.

2. **“More LoRA layers?” question**  
   Layer4+Layer3-LoRA performs worse than Layer3-LoRA, so adding more adapters does not automatically improve accuracy.

3. **Learning-rate confound**  
   The lr=`1e-4` control shows that Layer4-LoRA's low-data advantage partly depends on using lr=`1e-3`. This makes the analysis more honest and more robust for the report.

The updated report should therefore emphasize:

> We do not only implement LoRA; we systematically analyze rank, adapter placement, label budget, and learning-rate sensitivity.

This makes the A-level extension more convincing.
