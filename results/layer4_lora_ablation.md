| Method | Target | Budget | Rank | Trainable params | Val accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|---|---|---|---|
| Layer4-LoRA | layer4 + fc | 1% | 4 | 94,757 | 0.4049 | 0.4110 | 0.3996 |
| Layer4-LoRA | layer4 + fc | 1% | 8 | 170,533 | 0.4049 | 0.3783 | 0.3636 |
| Layer4-LoRA | layer4 + fc | 1% | 16 | 322,085 | 0.3505 | 0.3508 | 0.3431 |
| Layer4-LoRA | layer4 + fc | 10% | 4 | 94,757 | 0.7908 | 0.7525 | 0.7500 |
| Layer4-LoRA | layer4 + fc | 10% | 8 | 170,533 | 0.7935 | 0.7441 | 0.7394 |
| Layer4-LoRA | layer4 + fc | 10% | 16 | 322,085 | 0.7636 | 0.7285 | 0.7246 |
| Layer4-LoRA | layer4 + fc | 100% | 4 | 94,757 | 0.9076 | 0.8496 | 0.8473 |
| Layer4-LoRA | layer4 + fc | 100% | 8 | 170,533 | 0.8967 | 0.8397 | 0.8370 |
| Layer4-LoRA | layer4 + fc | 100% | 16 | 322,085 | 0.8886 | 0.8446 | 0.8427 |