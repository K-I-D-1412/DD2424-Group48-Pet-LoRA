# Data Pipeline

Pengyu Wang is responsible for the data pipeline and experiment management setup.

## Dataset

We use the Oxford-IIIT Pet Dataset for 37-class pet breed classification.

The official `trainval` split is used for training and validation. The official `test` split will be used only for final evaluation.

## Validation Split

We create a stratified validation split from the official `trainval` split.

Default settings:

- Random seed: 42
- Validation fraction: 10%
- Stratification: by pet breed class

The generated split files are stored in:

```text
data/splits/
```

## Label Budgets

We create stratified training subsets for the following label budgets:

- 100%
- 10%
- 1%

For very small label budgets, each class keeps at least one sample whenever possible.

## Class Imbalance

We simulate a class-imbalanced setting by reducing cat-breed training samples to 20% while keeping dog-breed samples unchanged.

This split is saved as:

```text
data/splits/train_imbalanced_cat20_seed42.json
```

## Data Augmentation

The training pipeline supports two modes:

1. Without augmentation
2. With augmentation

The default augmentation pipeline includes:

- Random resized crop
- Random horizontal flip
- Random rotation
- ImageNet normalization

Validation and test transforms are deterministic.

## Reproducibility

To make experiments reproducible, we:

- Fix the random seed
- Save all split indices as JSON files
- Use shared split files across all training strategies
- Use shared experiment configuration files