# Task Allocation

## Pengyu Wang

Pengyu Wang will lead data processing and experiment management.

Main responsibilities:

- Set up the Oxford-IIIT Pet data pipeline
- Create the official train/test split interface
- Create a stratified validation split from the training set
- Create 1%, 10%, and 100% label-budget training subsets
- Construct the class-imbalanced training split
- Implement data augmentation pipelines
- Support weighted cross-entropy and oversampling for imbalance experiments
- Maintain experiment configuration and result tracking conventions

## Jingmeng Xie

Jingmeng Xie will lead the fine-tuning baseline experiments.

Main responsibilities:

- Implement the linear probing baseline
- Implement simultaneous fine-tuning
- Implement gradual unfreezing
- Compare different numbers of unfrozen layers
- Run baseline experiments across different label budgets

## Yu Zhang

Yu Zhang will lead the LoRA implementation and ablation study.

Main responsibilities:

- Implement LoRALinear from scratch
- Integrate LoRA into the ResNet classification model
- Verify that gradients flow only to low-rank LoRA parameters
- Run rank ablations for r = 4, 8, and 16
- Compare LoRA with full fine-tuning in terms of accuracy, macro F1, trainable parameters, and training cost

## Jiachen Shi

Jiachen Shi will lead evaluation, visualization, and report writing.

Main responsibilities:

- Implement evaluation metrics
- Compute top-1 accuracy, macro F1, per-class accuracy, and per-class F1
- Generate confusion matrices
- Plot training curves
- Prepare result tables and figures
- Coordinate the final report and video presentation