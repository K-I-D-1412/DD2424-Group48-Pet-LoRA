"""Sanity checks for Yu Zhang's LoRA implementation.

Run from project root:
    python -m scripts.check_lora
"""

import torch
from torch import nn

from src.lora import LoRALinear, LoRAConv2d


def check_lora_linear():
    torch.manual_seed(0)
    base = nn.Linear(512, 37)
    x = torch.randn(4, 512)
    base_out = base(x).detach()

    layer = LoRALinear(base, rank=8, alpha=16, dropout=0.0)
    out = layer(x)
    assert out.shape == (4, 37)
    assert torch.allclose(out, base_out, atol=1e-6), "B=0 should preserve base output"
    assert not layer.base_layer.weight.requires_grad
    assert layer.lora_A.weight.requires_grad
    assert layer.lora_B.weight.requires_grad

    # Make B nonzero to verify gradients can flow to both A and B.
    nn.init.normal_(layer.lora_B.weight, std=0.01)
    loss = layer(x).pow(2).mean()
    loss.backward()
    assert layer.base_layer.weight.grad is None
    assert layer.lora_A.weight.grad is not None
    assert layer.lora_B.weight.grad is not None
    print("[OK] LoRALinear: shape, initialization, freezing, gradients")


def check_lora_conv2d():
    torch.manual_seed(0)
    base = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
    x = torch.randn(2, 256, 14, 14)
    base_out = base(x).detach()

    layer = LoRAConv2d(base, rank=4, alpha=8, dropout=0.0)
    out = layer(x)
    assert out.shape == base_out.shape
    assert torch.allclose(out, base_out, atol=1e-6), "B=0 should preserve base output"
    assert not layer.base_layer.weight.requires_grad
    assert layer.lora_A.weight.requires_grad
    assert layer.lora_B.weight.requires_grad

    nn.init.normal_(layer.lora_B.weight, std=0.01)
    loss = layer(x).pow(2).mean()
    loss.backward()
    assert layer.base_layer.weight.grad is None
    assert layer.lora_A.weight.grad is not None
    assert layer.lora_B.weight.grad is not None
    print("[OK] LoRAConv2d: shape, initialization, freezing, gradients")


def main():
    check_lora_linear()
    check_lora_conv2d()
    print("All LoRA checks passed.")


if __name__ == "__main__":
    main()
