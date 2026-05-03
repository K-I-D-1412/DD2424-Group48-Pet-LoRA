"""
Model construction utilities.

This module currently supports ResNet-18 for the baseline transfer learning
experiments. LoRA integration will be added later by the LoRA lead.
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


def configure_model_for_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """
    Configure which parameters should be trainable according to the strategy.

    Args:
        model: PyTorch model.
        strategy: Training strategy from config.

    Returns:
        The configured model.
    """
    if strategy == "linear_probe":
        freeze_backbone_for_linear_probe(model)
    elif strategy == "full_finetuning":
        unfreeze_all_parameters(model)
    elif strategy == "lora":
        raise NotImplementedError(
            "LoRA strategy is not implemented yet. "
            "This will be added in the LoRA module."
        )
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    return model