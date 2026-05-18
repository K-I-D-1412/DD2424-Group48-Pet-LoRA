| Experiment | Method | Target | Budget | Rank | Best epoch | Best val accuracy | Test accuracy | Test macro F1 | Trainable params | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| `resnet18_lora_r4_1_seed42` | FC-LoRA | fc only | 1% | 4 | 29 | 0.0380 | 0.0540 | 0.0385 | 2,196 | 432.4100s |
| `resnet18_lora_r8_1_seed42` | FC-LoRA | fc only | 1% | 8 | 29 | 0.0652 | 0.0752 | 0.0503 | 4,392 | 434.3100s |
| `resnet18_lora_r16_1_seed42` | FC-LoRA | fc only | 1% | 16 | 29 | 0.1196 | 0.1453 | 0.1124 | 8,784 | 433.1600s |
| `resnet18_lora_r4_10_seed42` | FC-LoRA | fc only | 10% | 4 | 29 | 0.2337 | 0.2099 | 0.1714 | 2,196 | 512.0400s |
| `resnet18_lora_r8_10_seed42` | FC-LoRA | fc only | 10% | 8 | 28 | 0.5707 | 0.5576 | 0.5475 | 4,392 | 511.7600s |
| `resnet18_lora_r16_10_seed42` | FC-LoRA | fc only | 10% | 16 | 26 | 0.6467 | 0.6514 | 0.6497 | 8,784 | 511.6300s |
| `resnet18_lora_r4_100_seed42` | FC-LoRA | fc only | 100% | 4 | 29 | 0.6277 | 0.6078 | 0.6047 | 2,196 | 1283.0000s |
| `resnet18_lora_r8_100_seed42` | FC-LoRA | fc only | 100% | 8 | 22 | 0.8207 | 0.7915 | 0.7901 | 4,392 | 1302.0200s |
| `resnet18_lora_r16_100_seed42` | FC-LoRA | fc only | 100% | 16 | 22 | 0.8641 | 0.8204 | 0.8180 | 8,784 | 1296.8400s |
| `resnet18_lora_layer4_r4_1_seed42` | Layer4-LoRA | layer4 + fc | 1% | 4 | 19 | 0.4049 | 0.4110 | 0.3996 | 94,757 | 441.4100s |
| `resnet18_lora_layer4_r8_1_seed42` | Layer4-LoRA | layer4 + fc | 1% | 8 | 21 | 0.4049 | 0.3783 | 0.3636 | 170,533 | 451.5500s |
| `resnet18_lora_layer4_r16_1_seed42` | Layer4-LoRA | layer4 + fc | 1% | 16 | 22 | 0.3505 | 0.3508 | 0.3431 | 322,085 | 438.5000s |
| `resnet18_lora_layer4_r4_10_seed42` | Layer4-LoRA | layer4 + fc | 10% | 4 | 20 | 0.7908 | 0.7525 | 0.7500 | 94,757 | 540.0100s |
| `resnet18_lora_layer4_r8_10_seed42` | Layer4-LoRA | layer4 + fc | 10% | 8 | 25 | 0.7935 | 0.7441 | 0.7394 | 170,533 | 528.7800s |
| `resnet18_lora_layer4_r16_10_seed42` | Layer4-LoRA | layer4 + fc | 10% | 16 | 23 | 0.7636 | 0.7285 | 0.7246 | 322,085 | 534.3800s |
| `resnet18_lora_layer4_r4_100_seed42` | Layer4-LoRA | layer4 + fc | 100% | 4 | 14 | 0.9076 | 0.8496 | 0.8473 | 94,757 | 1479.0900s |
| `resnet18_lora_layer4_r8_100_seed42` | Layer4-LoRA | layer4 + fc | 100% | 8 | 11 | 0.8967 | 0.8397 | 0.8370 | 170,533 | 1463.5800s |
| `resnet18_lora_layer4_r16_100_seed42` | Layer4-LoRA | layer4 + fc | 100% | 16 | 3 | 0.8886 | 0.8446 | 0.8427 | 322,085 | 1466.9900s |
| `resnet18_lora_layer4_r4_lr1e4_1_seed42` | Layer4-LoRA LR1e-4 | layer4 + fc | 1% | 4 | 29 | 0.0598 | 0.0777 | 0.0539 | 94,757 | 455.0700s |
| `resnet18_lora_layer4_r4_lr1e4_10_seed42` | Layer4-LoRA LR1e-4 | layer4 + fc | 10% | 4 | 29 | 0.5707 | 0.5451 | 0.5309 | 94,757 | 554.3900s |
| `resnet18_lora_layer4_r4_lr1e4_100_seed42` | Layer4-LoRA LR1e-4 | layer4 + fc | 100% | 4 | 29 | 0.8913 | 0.8550 | 0.8526 | 94,757 | 1475.8100s |
| `resnet18_lora_layer3_r4_1_seed42` | Layer3-LoRA | layer3 + fc | 1% | 4 | 29 | 0.3913 | 0.4058 | 0.3900 | 56,869 | 433.1900s |
| `resnet18_lora_layer3_r4_10_seed42` | Layer3-LoRA | layer3 + fc | 10% | 4 | 19 | 0.8207 | 0.7871 | 0.7851 | 56,869 | 569.3100s |
| `resnet18_lora_layer3_r4_100_seed42` | Layer3-LoRA | layer3 + fc | 100% | 4 | 16 | 0.9158 | 0.8795 | 0.8774 | 56,869 | 1704.9600s |
| `resnet18_lora_layer4_layer3_r4_1_seed42` | Layer4+3-LoRA | layer4+layer3+fc | 1% | 4 | 20 | 0.3886 | 0.3805 | 0.3721 | 132,645 | 456.8500s |
| `resnet18_lora_layer4_layer3_r4_10_seed42` | Layer4+3-LoRA | layer4+layer3+fc | 10% | 4 | 19 | 0.8071 | 0.7542 | 0.7528 | 132,645 | 577.4200s |
| `resnet18_lora_layer4_layer3_r4_100_seed42` | Layer4+3-LoRA | layer4+layer3+fc | 100% | 4 | 8 | 0.9158 | 0.8607 | 0.8583 | 132,645 | 1752.3500s |
| `resnet18_linear_probe_1_seed42` | Linear probe | fc only | 1% | - | 19 | 0.3777 | 0.3876 | 0.3773 | 18,981 | 585.8300s |
| `resnet18_linear_probe_10_seed42` | Linear probe | fc only | 10% | - | 15 | 0.7446 | 0.7370 | 0.7355 | 18,981 | 709.8500s |
| `resnet18_linear_probe_100_seed42` | Linear probe | fc only | 100% | - | 8 | 0.8777 | 0.8457 | 0.8422 | 18,981 | 1030.0400s |
| `resnet18_finetune_1_noaug_seed42` | Full fine-tuning | full | 1% | - | 23 | 0.2418 | 0.2311 | 0.2130 | 11,195,493 | 867.7500s |
| `resnet18_finetune_1_seed42` | Full fine-tuning | full | 1% | - | 28 | 0.2717 | 0.2608 | 0.2473 | 11,195,493 | 883.1200s |
| `resnet18_finetune_1_wdhigh_seed42` | Full fine-tuning | full | 1% | - | 28 | 0.2717 | 0.2614 | 0.2478 | 11,195,493 | 872.9200s |
| `resnet18_finetune_10_noaug_seed42` | Full fine-tuning | full | 10% | - | 29 | 0.7174 | 0.6645 | 0.6621 | 11,195,493 | 1177.9100s |
| `resnet18_finetune_10_seed42` | Full fine-tuning | full | 10% | - | 26 | 0.7418 | 0.7127 | 0.7083 | 11,195,493 | 1210.7800s |
| `resnet18_finetune_10_wdhigh_seed42` | Full fine-tuning | full | 10% | - | 26 | 0.7391 | 0.7130 | 0.7089 | 11,195,493 | 1170.3800s |
| `resnet18_finetune_100_seed42` | Full fine-tuning | full | 100% | - | 5 | 0.9130 | 0.8678 | 0.8646 | 11,195,493 | 4418.1900s |
| `resnet18_finetune_imbalanced_baseline_seed42` | Full fine-tuning | full | - | - | 6 | 0.8750 | 0.8297 | 0.8252 | 11,195,493 | 2471.3500s |
| `resnet18_finetune_imbalanced_cat20_oversampling_seed42` | Full fine-tuning | full | - | - | 24 | 0.8859 | 0.8408 | 0.8345 | 11,195,493 | 2433.8300s |
| `resnet18_finetune_imbalanced_cat20_weighted_ce_seed42` | Full fine-tuning | full | - | - | 5 | 0.8995 | 0.8466 | 0.8437 | 11,195,493 | 2468.5200s |
| `resnet18_gradual_unfreeze_1_seed42` | Gradual unfreezing | gradual | 1% | - | 28 | 0.2880 | 0.2881 | 0.2763 | 11,037,989 | 868.0600s |
| `resnet18_gradual_unfreeze_10_seed42` | Gradual unfreezing | gradual | 10% | - | 19 | 0.7636 | 0.7318 | 0.7256 | 11,037,989 | 1088.5700s |
| `resnet18_gradual_unfreeze_100_seed42` | Gradual unfreezing | gradual | 100% | - | 13 | 0.9158 | 0.8741 | 0.8724 | 11,037,989 | 1943.0000s |
| `resnet18_partial_finetune_l2_100_seed42` | Partial fine-tuning | partial | 100% | - | 10 | 0.9103 | 0.8662 | 0.8648 | 8,412,709 | 1720.3300s |
| `resnet18_partial_finetune_l3_100_seed42` | Partial fine-tuning | partial | 100% | - | 21 | 0.9239 | 0.8675 | 0.8661 | 10,512,421 | 1931.6700s |
| `sanity_binary_cat_vs_dog_seed42` | Binary sanity check | - | - | - | None | - | 0.9831 | 0.9805 | - | 708.2000s |