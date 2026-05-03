"""
Training entry point skeleton.

This script prepares the experiment according to a YAML config file. It does not
yet implement the full training loop, but it verifies that configuration,
dataloaders, model construction, parameter freezing, loss, and optimizer setup
work correctly.

Example:

    python -m src.train --config configs/linear_probe_100.yaml
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.data import get_dataloaders_for_split
from src.models import build_resnet18, configure_model_for_strategy
from src.utils import count_parameters, get_device, load_config, set_seed


def build_optimizer(model: nn.Module, config: dict):
    """
    Build optimizer from config.

    Args:
        model: PyTorch model.
        config: Loaded YAML config.

    Returns:
        A PyTorch optimizer.
    """
    training_cfg = config["training"]
    optimizer_name = training_cfg["optimizer"].lower()
    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg["weight_decay"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_loss(config: dict, class_weights: torch.Tensor, device: torch.device):
    """
    Build loss function from config.

    Args:
        config: Loaded YAML config.
        class_weights: Class weights computed from the training split.
        device: PyTorch device.

    Returns:
        A loss function.
    """
    use_weighted_loss = bool(config["data"].get("use_weighted_loss", False))

    if use_weighted_loss:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))

    return nn.CrossEntropyLoss()


def prepare_experiment(config_path: str):
    """
    Prepare dataloaders, model, loss, and optimizer for an experiment.

    Args:
        config_path: Path to YAML config.
    """
    config = load_config(config_path)

    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    device = get_device()

    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    data = get_dataloaders_for_split(
        train_split_name=data_cfg["train_split"],
        val_split_name=data_cfg["val_split"],
        image_size=int(data_cfg["image_size"]),
        batch_size=int(training_cfg["batch_size"]),
        num_workers=0,
        use_augmentation=bool(data_cfg["use_augmentation"]),
        use_oversampling=bool(data_cfg["use_oversampling"]),
        download=False,
    )

    architecture = model_cfg["architecture"].lower()

    if architecture == "resnet18":
        model = build_resnet18(
            num_classes=int(model_cfg["num_classes"]),
            pretrained=bool(model_cfg["pretrained"]),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = configure_model_for_strategy(
        model=model,
        strategy=model_cfg["strategy"],
    )

    model = model.to(device)

    loss_fn = build_loss(
        config=config,
        class_weights=data["class_weights"],
        device=device,
    )

    optimizer = build_optimizer(model=model, config=config)

    total_params, trainable_params = count_parameters(model)

    print("=" * 80)
    print("Experiment prepared successfully.")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Experiment name: {config['experiment']['name']}")
    print(f"Device: {device}")
    print(f"Architecture: {architecture}")
    print(f"Strategy: {model_cfg['strategy']}")
    print(f"Train split: {data_cfg['train_split']}")
    print(f"Validation split: {data_cfg['val_split']}")
    print(f"Train dataset size: {len(data['train_dataset'])}")
    print(f"Validation dataset size: {len(data['val_dataset'])}")
    print(f"Number of classes: {data['num_classes']}")
    print(f"Batch size: {training_cfg['batch_size']}")
    print(f"Epochs: {training_cfg['epochs']}")
    print(f"Optimizer: {training_cfg['optimizer']}")
    print(f"Learning rate: {training_cfg['learning_rate']}")
    print(f"Weight decay: {training_cfg['weight_decay']}")
    print(f"Use augmentation: {data_cfg['use_augmentation']}")
    print(f"Use weighted loss: {data_cfg['use_weighted_loss']}")
    print(f"Use oversampling: {data_cfg['use_oversampling']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Optimizer object: {optimizer.__class__.__name__}")

    images, labels = next(iter(data["train_loader"]))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(images)
        loss = loss_fn(logits, labels)

    print(f"Sanity-check batch image shape: {tuple(images.shape)}")
    print(f"Sanity-check logits shape: {tuple(logits.shape)}")
    print(f"Sanity-check loss: {loss.item():.4f}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML experiment config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_experiment(args.config)


if __name__ == "__main__":
    main()