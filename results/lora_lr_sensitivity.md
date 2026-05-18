| Method | Target | Budget | Rank | Trainable params | Val accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|---|---|---|---|
| Full fine-tuning | full | 1% | - | 11,195,493 | 0.2717 | 0.2608 | 0.2473 |
| Layer4-LoRA | layer4 + fc | 1% | 4 | 94,757 | 0.4049 | 0.4110 | 0.3996 |
| Layer4-LoRA LR1e-4 | layer4 + fc | 1% | 4 | 94,757 | 0.0598 | 0.0777 | 0.0539 |
| Full fine-tuning | full | 10% | - | 11,195,493 | 0.7418 | 0.7127 | 0.7083 |
| Layer4-LoRA | layer4 + fc | 10% | 4 | 94,757 | 0.7908 | 0.7525 | 0.7500 |
| Layer4-LoRA LR1e-4 | layer4 + fc | 10% | 4 | 94,757 | 0.5707 | 0.5451 | 0.5309 |
| Full fine-tuning | full | 100% | - | 11,195,493 | 0.9130 | 0.8678 | 0.8646 |
| Layer4-LoRA | layer4 + fc | 100% | 4 | 94,757 | 0.9076 | 0.8496 | 0.8473 |
| Layer4-LoRA LR1e-4 | layer4 + fc | 100% | 4 | 94,757 | 0.8913 | 0.8550 | 0.8526 |