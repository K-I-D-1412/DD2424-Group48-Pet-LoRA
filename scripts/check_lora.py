"""Sanity checks for Yu Zhang's LoRA implementation.

Run from project root:
    python -m scripts.check_lora
"""

import torch
from torch import nn

from src.lora import LoRALinear, LoRAConv2d, apply_lora_to_resnet18


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


def check_lora_resnet_target_modules():
    # Use a tiny ResNet-like module tree so this check does not need torchvision.
    class TinyResNetLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False))
            self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1, bias=False))
            self.layer3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, bias=False))
            self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, bias=False))
            self.fc = nn.Linear(64, 37)

    model = TinyResNetLike()
    model = apply_lora_to_resnet18(
        model,
        {
            "rank": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["layer4", "layer3"],
            "train_classifier": True,
        },
    )
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any("layer3" in name and "lora_" in name for name in trainable)
    assert any("layer4" in name and "lora_" in name for name in trainable)
    assert any(name.startswith("fc.") for name in trainable)
    assert not any("base_layer" in name and p.requires_grad for name, p in model.named_parameters())
    print("[OK] target_modules: layer3/layer4 LoRA + trainable classifier")


def main():
    check_lora_linear()
    check_lora_conv2d()
    check_lora_resnet_target_modules()
    print("All LoRA checks passed.")


if __name__ == "__main__":
    main()
