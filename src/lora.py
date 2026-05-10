"""
LoRA modules and model surgery utilities for ResNet-18.

Yu Zhang - LoRA implementation and ablation study.

The core LoRA idea is to freeze a pretrained weight W0 and learn only a
low-rank update ΔW = B A, scaled by alpha / rank:

    y = W0 x + (alpha / rank) B A x

This file intentionally does not use any PEFT library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """LoRA wrapper for ``nn.Linear``.

    The wrapped base linear layer is frozen. Only ``lora_A`` and ``lora_B`` are
    trainable. ``lora_B`` is initialized to zero, so the initial output is
    exactly equal to the frozen base layer output.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        # get base_layer input output size
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features


class LoRAConv2d(nn.Module):
    """LoRA wrapper for ``nn.Conv2d``.

    We factorize the convolutional update as:

        Conv_B( Conv_A(x) )

    where Conv_A uses the original kernel size/stride/padding/dilation and maps
    input channels to ``rank`` channels, while Conv_B is a 1x1 convolution that
    maps ``rank`` channels to the original output channels. This keeps the output
    spatial shape identical to the frozen base convolution.

    This implementation supports the standard ResNet-18 convolutions
    (groups=1). It is intentionally simple and transparent for the report.
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError(f"base_layer must be nn.Conv2d, got {type(base_layer)}")
        if base_layer.groups != 1:
            raise ValueError("LoRAConv2d currently supports only groups=1 convolutions.")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity()

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=self.rank,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=1,
            bias=False,
            padding_mode=base_layer.padding_mode,
        )
        self.lora_B = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("fc",)
    train_classifier: bool = False

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "LoRAConfig":
        cfg = cfg or {}
        targets = cfg.get("target_modules", ["fc"])
        if isinstance(targets, str):
            targets = [targets]
        return cls(
            rank=int(cfg.get("rank", cfg.get("r", 8))),
            alpha=float(cfg.get("alpha", 16)),
            dropout=float(cfg.get("dropout", 0.0)),
            target_modules=tuple(str(t) for t in targets),
            train_classifier=bool(cfg.get("train_classifier", False)),
        )


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """Return ``(parent, child_name)`` for a dotted module path."""
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
    parent, child_name = _get_parent_module(root, module_name)
    setattr(parent, child_name, new_module)


def _freeze_all_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_classifier(model: nn.Module):
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True


def apply_lora_to_resnet18(model: nn.Module, lora_cfg: dict | LoRAConfig | None) -> nn.Module:
    """Apply LoRA adapters to selected ResNet-18 modules.

    Supported target modules:
    - ``"fc"``: replace the final classification head with ``LoRALinear``.
    - ``"layer4"``: replace every Conv2d inside ``model.layer4`` with
      ``LoRAConv2d``. For this target, set ``train_classifier: true`` in the
      YAML config so the newly initialized 37-class head can learn normally.

    All existing parameters are frozen first. After replacement, LoRA A/B
    parameters are trainable. The classifier can optionally remain trainable.
    """
    cfg = lora_cfg if isinstance(lora_cfg, LoRAConfig) else LoRAConfig.from_dict(lora_cfg)

    _freeze_all_parameters(model)
    targets = set(t.lower() for t in cfg.target_modules)

    if "fc" in targets:
        model.fc = LoRALinear(model.fc, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)

    if "layer4" in targets:
        for name, module in list(model.layer4.named_modules()):
            if isinstance(module, nn.Conv2d):
                full_name = f"layer4.{name}" if name else "layer4"
                wrapped = LoRAConv2d(
                    module,
                    rank=cfg.rank,
                    alpha=cfg.alpha,
                    dropout=cfg.dropout,
                )
                _replace_module(model, full_name, wrapped)

    unsupported = targets - {"fc", "layer4"}
    if unsupported:
        raise ValueError(
            f"Unsupported LoRA target_modules={sorted(unsupported)}. "
            "Supported values are 'fc' and 'layer4'."
        )

    # Ensure LoRA parameters are trainable even though the base model was frozen.
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            for param in module.lora_A.parameters():
                param.requires_grad = True
            for param in module.lora_B.parameters():
                param.requires_grad = True

    if cfg.train_classifier:
        _unfreeze_classifier(model)

    return model


def iter_lora_parameter_names(model: nn.Module) -> Iterable[str]:
    for name, _ in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            yield name


def lora_parameter_count(model: nn.Module) -> int:
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]
