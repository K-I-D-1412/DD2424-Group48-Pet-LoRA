| Budget | Method | Trainable params | Best epoch | Best val acc (%) | Test acc (%) | Test macro F1 (%) | Time | Experiment |
|---|---|---|---|---|---|---|---|---|
| 1% | Layer4-LoRA r=4, lr=1e-3 | 94,757 | 19 | 40.49 | 41.10 | 39.96 | 441.4s | resnet18_lora_layer4_r4_1_seed42 |
| 1% | Layer4-LoRA r=4, lr=1e-4 | 94,757 | 29 | 5.98 | 7.77 | 5.39 | 455.1s | resnet18_lora_layer4_r4_lr1e4_1_seed42 |
| 10% | Layer4-LoRA r=4, lr=1e-3 | 94,757 | 20 | 79.08 | 75.25 | 75.00 | 540.0s | resnet18_lora_layer4_r4_10_seed42 |
| 10% | Layer4-LoRA r=4, lr=1e-4 | 94,757 | 29 | 57.07 | 54.51 | 53.09 | 554.4s | resnet18_lora_layer4_r4_lr1e4_10_seed42 |
| 100% | Layer4-LoRA r=4, lr=1e-3 | 94,757 | 14 | 90.76 | 84.96 | 84.73 | 1479.1s | resnet18_lora_layer4_r4_100_seed42 |
| 100% | Layer4-LoRA r=4, lr=1e-4 | 94,757 | 29 | 89.13 | 85.50 | 85.26 | 1475.8s | resnet18_lora_layer4_r4_lr1e4_100_seed42 |