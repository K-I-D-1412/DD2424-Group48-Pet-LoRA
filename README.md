# DD2424 Group 48: Transfer Learning and LoRA for Pet Breed Classification

This repository contains the code for our DD2424 project: **Transfer Learning and LoRA for Pet Breed Classification on the Oxford-IIIT Pet Dataset**.

## Project Overview

We study transfer learning for 37-class pet breed classification using pretrained ResNet models on the Oxford-IIIT Pet Dataset.

The project compares the following strategies:

- Linear probing
- Simultaneous fine-tuning
- Gradual unfreezing
- LoRA-based parameter-efficient fine-tuning

We also investigate limited-data settings, class imbalance, data augmentation, regularization, and LoRA rank ablations.

## Group Members

- Jiachen Shi
- Jingmeng Xie
- Pengyu Wang
- Yu Zhang

## Repository Structure

```text
configs/          Experiment configuration files
src/              Core source code
scripts/          Helper scripts for running experiments
                  and plotting (scripts/plot_*.py)
experiments/      Experiment logs and lightweight result files
results/          Final tables and figures
results/figures/  Generated PNGs grouped by topic:
                    training_curves/, confusion_matrices/,
                    per_class_f1/, lora_params_vs_acc/,
                    representation/, training_cost/
notebooks/        Exploratory notebooks
report/           Report-related figures and notes
docs/             Planning documents and task allocation
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Example Usage

Train a model with a configuration file:

```bash
python src/train.py --config configs/linear_probe.yaml
```

## Notes

Large files such as datasets, checkpoints, and raw experiment logs should not be committed to GitHub.