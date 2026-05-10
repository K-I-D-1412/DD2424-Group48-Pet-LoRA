| Method | Target | Budget | Rank | Trainable params | Val accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|---|---|---|---|
| Linear probe | fc only | 1% | - | 18,981 | 0.3777 | 0.3876 | 0.3773 |
| Full fine-tuning | full | 1% | - | 11,195,493 | 0.2717 | 0.2608 | 0.2473 |
| Gradual unfreezing | gradual | 1% | - | 11,037,989 | 0.2880 | 0.2881 | 0.2763 |
| Layer4-LoRA | layer4 + fc | 1% | 4 | 94,757 | 0.4049 | 0.4110 | 0.3996 |
| Linear probe | fc only | 10% | - | 18,981 | 0.7446 | 0.7370 | 0.7355 |
| Full fine-tuning | full | 10% | - | 11,195,493 | 0.7418 | 0.7127 | 0.7083 |
| Gradual unfreezing | gradual | 10% | - | 11,037,989 | 0.7636 | 0.7318 | 0.7256 |
| Layer4-LoRA | layer4 + fc | 10% | 4 | 94,757 | 0.7908 | 0.7525 | 0.7500 |
| Linear probe | fc only | 100% | - | 18,981 | 0.8777 | 0.8457 | 0.8422 |
| Full fine-tuning | full | 100% | - | 11,195,493 | 0.9130 | 0.8678 | 0.8646 |
| Gradual unfreezing | gradual | 100% | - | 11,037,989 | 0.9158 | 0.8741 | 0.8724 |
| Layer4-LoRA | layer4 + fc | 100% | 4 | 94,757 | 0.9076 | 0.8496 | 0.8473 |