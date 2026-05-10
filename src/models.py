"""
Model construction utilities.

This module currently supports ResNet-18 for the baseline transfer learning
experiments. LoRA integration will be added later by the LoRA lead.

Jingmeng - freeze_all_except_last_n_blocks + configure_model_for_strategy  (Strategy 1)
"""

import torch
from torch import nn
from torchvision import models


def build_resnet18(num_classes: int = 37, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet-18 model with a replaced classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A ResNet-18 model.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone_for_linear_probe(model: nn.Module):
    """
    Freeze all parameters except the final classification layer.

    Args:
        model: ResNet model.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all_parameters(model: nn.Module):
    """
    Make all model parameters trainable.

    Args:
        model: PyTorch model.
    """
    for param in model.parameters():
        param.requires_grad = True

# Unfreeze ResNet-18 blocks, arranged from output to input
RESNET_BLOCKS_OUTPUT_TO_INPUT = ["fc", "layer4", "layer3", "layer2", "layer1"]


def freeze_all_except_last_n_blocks(model: nn.Module, n: int):
    """
    Strategy 1 Helper Function: Freeze all parameters, then unfreeze the last n blocks.
    When n = 1 → Only fc; when n = 2 → fc + layer4; and so on.
    """
    if n < 1 or n > len(RESNET_BLOCKS_OUTPUT_TO_INPUT):
        raise ValueError(
            f"n must be in [1, {len(RESNET_BLOCKS_OUTPUT_TO_INPUT)}], got {n}"
        )
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last n blocks
    for block_name in RESNET_BLOCKS_OUTPUT_TO_INPUT[:n]:
        block = getattr(model, block_name)
        for param in block.parameters():
            param.requires_grad = True

# Adding partial_finetune branch + unfrozen_layers parameters (Jingmeng)
def configure_model_for_strategy(
    model: nn.Module,
    strategy: str,
    unfrozen_layers=None,
) -> nn.Module:
    """
    Based on the "strategy" field, which parameters can be trained.
    "partial_finetune" requires an additional parameter "unfrozen_layers" (an integer or "all").
    """
    strategy = strategy.lower()

    if strategy == "linear_probe":
        freeze_backbone_for_linear_probe(model)

    elif strategy == "full_finetuning":
        unfreeze_all_parameters(model)

    elif strategy == "partial_finetune":
        if unfrozen_layers is None:
            raise ValueError(
                "Strategy 'partial_finetune' requires model.unfrozen_layers in config."
            )
        if isinstance(unfrozen_layers, str) and unfrozen_layers.strip().lower() == "all":
            unfreeze_all_parameters(model)
        else:
            n = int(unfrozen_layers)
            freeze_all_except_last_n_blocks(model, n=n)

    elif strategy == "gradual_unfreezing":
        # The initial state is the same as that of linear_probe. The scheduler will gradually thaw it.
        freeze_backbone_for_linear_probe(model)

    elif strategy == "lora":
        raise NotImplementedError(
            "The LoRA strategy has not been implemented. Waiting for src/lora.py."
        )

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    return model