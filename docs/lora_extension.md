# LoRA Extension Notes

Yu Zhang implements LoRA from scratch without PEFT libraries.

## Method

For a frozen base weight `W0`, LoRA learns a low-rank update:

`y = W0 x + (alpha / r) B A x`

Only `A` and `B` are trainable. `B` is initialized to zero, so the LoRA-wrapped
module initially behaves exactly like the frozen base module.

## Experiments

1. FC-LoRA rank ablation: `r = 4, 8, 16` over `1% / 10% / 100%` label budgets.
2. Layer4-LoRA extension: apply LoRAConv2d to all convolutions in ResNet-18
   `layer4`, while training the new 37-class classifier head.

The main analysis compares accuracy, macro F1, trainable parameters, and
training time against linear probing, full fine-tuning, and gradual unfreezing.
