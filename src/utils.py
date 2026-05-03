"""
General utility functions for reproducible experiments.
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        A dictionary containing the configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is not a dictionary: {config_path}")

    return config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        A tuple: total parameters, trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params